"""Background-daemon helpers for `mucli --gui`.

Spawns the real server in a detached child process, writes a PID file,
and returns the terminal to the user. `--gui-stop` reads the PID file
and SIGTERMs the child.

Not a full POSIX daemon (no double-fork, no umask reset, no chdir to
``/``) — that's overkill for a per-user CLI tool. We use
``subprocess.Popen(start_new_session=True)`` which:

- Detaches the child from the controlling terminal (new session).
- Survives the parent shell exiting.
- Lets us redirect stdio to a log file so the child doesn't write to
  the user's terminal.

PID file:  ``~/.mucli/gui.pid``
Log file:  ``~/.mucli/logs/gui.log``  (overwritten each boot)
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import utils.config as _config


def _home() -> Path:
    return Path(_config.HISTORY_DIR)


def pid_file() -> Path:
    return _home() / "gui.pid"


def log_file() -> Path:
    return _home() / "logs" / "gui.log"


def _cmdline_is_mucli(pid: int) -> bool | None:
    """Round-27 F1: does the target's command line look like mucli?

    Reads the process command line from procfs (Linux). Returns
    True/False when readable, None when unreadable (EPERM, or the
    process exited between the liveness check and the read). Used to
    avoid SIGTERM/SIGKILL-ing an unrelated process after PID reuse or
    when the port fallback resolves a foreign listener.
    """
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            raw = fh.read()
    except (OSError, ValueError):
        return None
    argv = raw.split(b"\0")
    return any(b"mucli" in arg for arg in argv)


def is_running() -> int | None:
    """Return the PID of the existing daemon, or None if not running."""
    path = pid_file()
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return None
    if pid <= 0:
        return None
    try:
        # signal 0 → ESRCH if process gone, EPERM if not ours but alive.
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            # Stale pid file; remove it.
            try:
                path.unlink()
            except OSError:
                pass
            return None
        # EPERM means it's alive but we can't signal it — still running.
        return pid
    # Round-27 F1: PID reuse guard — the pid file may name a recycled
    # PID now owned by an unrelated process. Verify the cmdline looks
    # like mucli before reporting it as "the daemon"; an unreadable
    # cmdline keeps the conservative old behavior (the pid file is
    # high-trust: we wrote it on launch).
    identity = _cmdline_is_mucli(pid)
    if identity is False:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return pid


def _listener_inodes(port: int) -> set[str]:
    """Socket inodes of LISTEN sockets on `port`, from /proc/net/tcp[6].

    Linux-only. Returns the set of inode strings so the caller can cross-
    reference them against /proc/*/fd socket symlinks to find the owning
    PID. Empty set if /proc isn't available or nothing matches.
    """
    inodes: set[str] = set()
    if not os.path.isdir("/proc"):
        return inodes
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, "r") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]      # "IPHEX:PORTHEX"
            state = parts[3]      # "0A" = LISTEN
            inode = parts[9]
            if state != "0A":
                continue
            try:
                _ip, port_hex = local.rsplit(":", 1)
                if int(port_hex, 16) == int(port):
                    inodes.add(inode)
            except ValueError:
                continue
    return inodes


def pid_for_port(port: int) -> int | None:
    """Find the PID of the process listening on `port` via /proc (Linux).

    Fallback for when the PID file is missing/stale but a server is
    genuinely bound to the port — `--gui-stop` and the start-path
    "already running" check use this so an orphaned server is still
    discoverable. Returns None off-Linux or if nothing is listening.
    """
    inodes = _listener_inodes(port)
    if not inodes:
        return None
    self_pid = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            # target looks like "socket:[<inode>]"
            if target.startswith("socket:[") and target.endswith("]"):
                if target[8:-1] in inodes:
                    return pid
    return None


def spawn_detached(args: list[str], *, port: int) -> int:
    """Spawn the foreground-server invocation as a detached child.

    Caller is responsible for the foreground command shape — typically
    a re-invocation of this script with an internal marker flag so the
    child knows to actually run the server.

    Returns the child PID. Caller should write it to ``pid_file()``.
    """
    log_path = log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "wb", buffering=0)

    # Round-27 F7: the child inherits (dups) the descriptor via
    # stdout/stderr — the parent's own copy must be closed (including
    # on Popen failure) or repeated programmatic launches leak fds.
    try:
        # start_new_session=True detaches the child from the controlling
        # terminal. stdio → log file. Parent exits independently.
        child = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        try:
            log_handle.close()
        except OSError:
            pass
    return child.pid


def write_pid(pid: int) -> None:
    path = pid_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def stop(timeout: float = 5.0, port: int = 30311) -> tuple[bool, str]:
    """SIGTERM the daemon. Returns ``(ok, message)``.

    Resolves the target PID from the PID file first; if that's missing or
    stale (a common failure: the file got removed but the server kept
    running — a prior stop that deleted-then-missed, a direct
    ``--gui-foreground`` launch, or the launcher parent dying before
    writing it), falls back to the process listening on ``port`` via
    ``pid_for_port`` so an orphaned server is still stoppable.
    """
    pid = is_running()
    source = "pid file"
    if pid is None:
        candidate = pid_for_port(port)
        # Round-27 F1: a port-derived PID is low-trust — only signal it
        # when its cmdline identifies it as mucli (never kill a foreign
        # process that happens to hold the port).
        if candidate is not None and _cmdline_is_mucli(candidate) is True:
            pid = candidate
            source = f"port {port}"
    if pid is None:
        return False, "no GUI server is running"
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, f"could not signal pid {pid}: {exc}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                try:
                    pid_file().unlink()
                except OSError:
                    pass
                return True, f"stopped pid {pid} (found via {source})"
        time.sleep(0.1)

    # Still alive after timeout — escalate.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        pid_file().unlink()
    except OSError:
        pass
    return True, f"force-killed pid {pid} (found via {source}, didn't respond to SIGTERM)"


def wait_for_port(port: int, *, host: str = "127.0.0.1", timeout: float = 8.0) -> bool:
    """Poll-connect until the server starts listening, or timeout."""
    import socket as _socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with _socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False
