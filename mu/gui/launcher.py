"""GUI server boot.

Two entry points:
  * ``run_gui(args, build_session)`` — what mucli calls when ``--gui``
    is set. By default it daemonizes (forks itself with a marker flag)
    and returns to the terminal.
  * ``run_server_foreground(args, build_session)`` — what the daemon
    child runs. Stays in the foreground (logs to gui.log) and runs
    uvicorn until killed.

The marker flag is ``--gui-serve`` and is internal — users never set
it. It's how the parent process tells the spawned child "you're the
worker, actually run the server."
"""

from __future__ import annotations

import os
import sys

from utils.logger import logger

from . import daemon
from .app import create_app


DEFAULT_PORT = 30311
DEFAULT_HOST = "127.0.0.1"


def run_gui(args, build_session) -> None:
    """Top-level entry. Daemonizes by default; runs in foreground if
    ``args.gui_foreground`` is set."""
    port = int(getattr(args, "port", None) or DEFAULT_PORT)
    host = getattr(args, "host", None) or DEFAULT_HOST

    # Non-loopback bind requires explicit opt-in (codex round-6 F5): the
    # GUI has no authentication — every LAN-reachable client could delete
    # sessions/files, watch all SSE traffic, and open container shells.
    # MUCLI_GUI_ALLOW_REMOTE=1 documents informed consent at the env level.
    import ipaddress as _ip

    try:
        bind_is_loopback = _ip.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        bind_is_loopback = host in ("localhost", "")
    if not bind_is_loopback and not os.environ.get("MUCLI_GUI_ALLOW_REMOTE"):
        raise SystemExit(
            f"Refusing to bind the GUI to {host}: the GUI has no built-in "
            "authentication (any LAN client could delete sessions/files, "
            "read all event traffic, and open container shells).\n"
            "Keep the default 127.0.0.1, or set MUCLI_GUI_ALLOW_REMOTE=1 "
            "to accept the risk explicitly."
        )

    # gui_foreground marker MUST be checked before is_running — the
    # parent writes the pid file before spawning the child, so the
    # child would otherwise read its OWN pid and exit thinking the GUI
    # is "already running."
    if getattr(args, "gui_foreground", False):
        run_server_foreground(args, build_session, port=port, host=host)
        return

    existing = daemon.is_running()
    if existing is None:
        # PID file missing/stale — but an orphaned server may still be
        # bound to the port (the file got removed while the server kept
        # running). Detect it so we don't spawn a second child that fails
        # to bind with EADDRINUSE.
        existing = daemon.pid_for_port(port)
    if existing is not None:
        url = f"http://{host}:{port}/"
        print(f"  mucli GUI already running at {url} (pid {existing})")
        print(f"  stop with: mucli --gui-stop")
        return

    # Re-invoke ourselves with the internal marker so the child runs
    # the server in foreground while we detach.
    child_args = _build_child_argv(args, port, host)
    pid = daemon.spawn_detached(child_args, port=port)
    daemon.write_pid(pid)

    if not daemon.wait_for_port(port, host=host, timeout=8.0):
        print(
            f"  mucli GUI spawned (pid {pid}) but isn't listening on {host}:{port} yet.\n"
            f"  Tail the log: tail -f {daemon.log_file()}"
        )
        return

    print(f"  mucli GUI → http://{host}:{port}/  (pid {pid})")
    print(f"  log:  {daemon.log_file()}")
    print(f"  stop: mucli --gui-stop")


def run_server_foreground(args, build_session, *, port: int, host: str = DEFAULT_HOST) -> None:
    """Run uvicorn in the current process. Used by the daemon child."""
    app = create_app(args=args, build_session_fn=build_session, port=port)

    import signal
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    # Ensure SIGTERM triggers a clean shutdown. Uvicorn installs its own
    # handlers inside server.run(), but when the process is a detached
    # daemon (start_new_session=True) those handlers can miss the signal
    # if the event loop is blocked on a thread join. This pre-handler
    # sets the flag that makes uvicorn's next loop iteration exit.
    _orig_sigterm = signal.getsignal(signal.SIGTERM)

    def _shutdown_handler(signum, frame):
        server.should_exit = True
        if callable(_orig_sigterm) and _orig_sigterm not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            _orig_sigterm(signum, frame)

    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        server.run()
    finally:
        try:
            app.state.prompts.cancel_all()
        except Exception:
            pass
        session = getattr(app.state, "session", None)
        if session is not None:
            try:
                session.session_manager.save_history(session.folder_context)
            except Exception:
                pass
        try:
            daemon.pid_file().unlink()
        except OSError:
            pass
        logger.info("GUI: server stopped")


def stop_gui(port: int | None = None) -> int:
    """`mucli --gui-stop` entry. Returns shell exit code.

    ``port`` is the port the GUI was started on (defaults to 30311) — used
    as a fallback to locate an orphaned server whose PID file is missing.
    """
    ok, msg = daemon.stop(port=int(port or DEFAULT_PORT))
    print(f"  {msg}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------


def _build_child_argv(args, port: int, host: str = DEFAULT_HOST) -> list[str]:
    """Construct the argv for the daemonized child.

    Forwards the user's flags (session, provider, model, workspace,
    yolo, debug, system) and tags the invocation with ``--gui
    --gui-foreground --port <port> --host <host>`` so the child runs
    the server in foreground (no further forking).
    """
    py = sys.executable or "python3.11"
    script = _resolve_mucli_script()
    argv = [py, script, "--gui", "--gui-foreground", "--port", str(port)]

    # Only forward --host when the user explicitly overrode the default.
    # This keeps the child's behavior predictable when no flag was given.
    if host and host != DEFAULT_HOST:
        argv += ["--host", str(host)]

    if getattr(args, "session", None):
        argv += ["--session", str(args.session)]
    if getattr(args, "provider", None):
        argv += ["--provider", str(args.provider)]
    if getattr(args, "model", None):
        argv += ["--model", str(args.model)]
    for workspace in getattr(args, "workspace", None) or []:
        argv += ["--workspace", str(workspace)]
    if getattr(args, "yolo", False):
        argv += ["--yolo"]
    if getattr(args, "debug", False):
        argv += ["--debug"]
    return argv


def _resolve_mucli_script() -> str:
    """Locate mucli.py on disk for child re-invocation."""
    # The repository layout is fixed: mucli.py lives at the tools root.
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(here, "mucli.py")
    if os.path.exists(candidate):
        return candidate
    # Fallback: $0 from argv if available.
    if sys.argv and os.path.exists(sys.argv[0]):
        return sys.argv[0]
    return "mucli.py"
