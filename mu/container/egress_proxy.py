"""Minimal allowlist HTTP/HTTPS proxy for MuCLI container workers.

The worker is attached only to an ``--internal`` Docker network.  This proxy is
attached to that network and to a separate ordinary bridge, making it the sole
egress path.  Policy enforcement therefore needs no host firewall changes,
root process, ``sudo``, or elevated container capabilities.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

_MAX_HEADER_BYTES = 64 * 1024
_COPY_CHUNK = 64 * 1024


def _normalise_host(value: str) -> str:
    host = str(value or "").strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _host_authority(host: str, port: int) -> str:
    """RFC-correct Host header authority (codex round-9 F6).

    IPv6 literals are bracketed; the port is included whenever it is not
    the scheme default implied by the connection. Callers pass the port
    they actually opened, so a non-default port survives synthesis instead
    of silently routing to the origin's default vhost.
    """
    if ":" in host:  # bare IPv6 literal (already unbracketed by normalise)
        authority = f"[{host}]"
    else:
        authority = host
    if port not in (80, 443):
        authority += f":{port}"
    return authority


def _normalise_rule(value: str) -> str:
    return _normalise_host(value)


def _matches_rule(host: str, rule: str) -> bool:
    host = _normalise_host(host)
    rule = _normalise_rule(rule)
    if not host or not rule:
        return False
    if rule.startswith("*."):
        suffix = rule[1:]
        return host.endswith(suffix) and host != suffix[1:]
    try:
        network = ipaddress.ip_network(rule, strict=False)
        return ipaddress.ip_address(host) in network
    except ValueError:
        return host == rule


def _parse_host_port(value: str, default_port: int) -> tuple[str, int]:
    target = str(value or "").strip()
    if target.startswith("["):
        closing = target.find("]")
        if closing < 0:
            raise ValueError("invalid IPv6 target")
        host = target[1:closing]
        remainder = target[closing + 1 :]
        port = int(remainder[1:]) if remainder.startswith(":") else default_port
        return _normalise_host(host), port
    if target.count(":") == 1:
        host, raw_port = target.rsplit(":", 1)
        if raw_port.isdigit():
            return _normalise_host(host), int(raw_port)
    return _normalise_host(target), int(default_port)


@dataclass(frozen=True)
class EgressPolicy:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    host_allow: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> "EgressPolicy":
        def load_list(key: str) -> tuple[str, ...]:
            try:
                value = json.loads(os.getenv(key, "[]"))
            except (TypeError, ValueError):
                value = []
            if not isinstance(value, list):
                value = []
            return tuple(
                dict.fromkeys(
                    rule
                    for item in value
                    if (rule := _normalise_rule(str(item or "")))
                )
            )

        try:
            host_value = json.loads(os.getenv("MUCLI_PROXY_HOST_ALLOW", "{}"))
        except (TypeError, ValueError):
            host_value = {}
        host_allow: dict[str, tuple[int, ...]] = {}
        if isinstance(host_value, dict):
            for raw_host, raw_ports in host_value.items():
                host = _normalise_host(str(raw_host or ""))
                if not host:
                    continue
                values = raw_ports if isinstance(raw_ports, list) else [raw_ports]
                ports = tuple(
                    sorted(
                        {
                            int(port)
                            for port in values
                            if str(port).isdigit() and 0 < int(port) <= 65535
                        }
                    )
                )
                if ports:
                    host_allow[host] = ports
        return cls(
            allow=load_list("MUCLI_PROXY_ALLOW"),
            deny=load_list("MUCLI_PROXY_DENY"),
            host_allow=host_allow,
        )

    def permits(self, host: str, port: int) -> bool:
        host = _normalise_host(host)
        port = int(port)
        # Control-plane access is explicit and separate from user egress rules.
        if port in self.host_allow.get(host, ()):
            return True
        if port not in {80, 443}:
            return False
        if any(_matches_rule(host, rule) for rule in self.deny):
            return False
        return any(_matches_rule(host, rule) for rule in self.allow)

    def is_control_plane(self, host: str, port: int) -> bool:
        return int(port) in self.host_allow.get(_normalise_host(host), ())


async def _public_addresses(host: str, port: int, *, allow_private: bool) -> list[tuple]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=__import__("socket").SOCK_STREAM)
    accepted: list[tuple] = []
    seen: set[tuple] = set()
    for family, socktype, proto, _canonname, sockaddr in infos:
        address = sockaddr[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not allow_private and (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
            or parsed.is_unspecified
        ):
            continue
        key = (family, socktype, proto, sockaddr)
        if key not in seen:
            seen.add(key)
            accepted.append(key)
    return accepted


async def _open_target(policy: EgressPolicy, host: str, port: int):
    addresses = await _public_addresses(
        host,
        port,
        allow_private=policy.is_control_plane(host, port),
    )
    if not addresses:
        raise OSError("target resolved only to disallowed addresses")
    last_error: Exception | None = None
    for family, _socktype, _proto, sockaddr in addresses:
        try:
            return await asyncio.open_connection(sockaddr[0], sockaddr[1], family=family)
        except OSError as exc:
            last_error = exc
    raise last_error or OSError("could not connect to target")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(_COPY_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.write_eof()
        except (AttributeError, OSError, RuntimeError):
            pass


async def _tunnel(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    remote_reader: asyncio.StreamReader,
    remote_writer: asyncio.StreamWriter,
) -> None:
    tasks = [
        asyncio.create_task(_pipe(client_reader, remote_writer)),
        asyncio.create_task(_pipe(remote_reader, client_writer)),
    ]
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    remote_writer.close()
    client_writer.close()
    await asyncio.gather(
        remote_writer.wait_closed(), client_writer.wait_closed(), return_exceptions=True
    )


def _error_response(status: int, reason: str) -> bytes:
    body = f"{status} {reason}\n".encode("utf-8")
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body


async def _read_headers(reader: asyncio.StreamReader) -> bytes:
    data = await reader.readuntil(b"\r\n\r\n")
    if len(data) > _MAX_HEADER_BYTES:
        raise ValueError("request headers too large")
    return data


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    policy: EgressPolicy,
) -> None:
    peer = writer.get_extra_info("peername")
    try:
        raw_headers = await _read_headers(reader)
        header_text = raw_headers.decode("iso-8859-1")
        lines = header_text.split("\r\n")
        method, target, version = lines[0].split(" ", 2)
        method_upper = method.upper()

        if method_upper == "CONNECT":
            host, port = _parse_host_port(target, 443)
            if not policy.permits(host, port):
                writer.write(_error_response(403, "Forbidden"))
                await writer.drain()
                return
            remote_reader, remote_writer = await _open_target(policy, host, port)
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            print(f"allow CONNECT {host}:{port} peer={peer}", flush=True)
            await _tunnel(reader, writer, remote_reader, remote_writer)
            return

        parsed = urlsplit(target)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            writer.write(_error_response(400, "Bad Request"))
            await writer.drain()
            return
        host = _normalise_host(parsed.hostname)
        port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
        if parsed.scheme == "https" or not policy.permits(host, port):
            writer.write(_error_response(403, "Forbidden"))
            await writer.drain()
            return
        remote_reader, remote_writer = await _open_target(policy, host, port)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        forwarded = [f"{method} {path} {version}"]
        for line in lines[1:]:
            lowered = line.lower()
            if lowered.startswith("proxy-connection:"):
                continue
            # Host header synthesis (codex round-8 F1): a client can send
            # a Host header that differs from the allowlisted target,
            # causing the origin (or an intermediary) to route the request
            # to a vhost outside the allowlist. Drop incoming Host headers
            # and synthesize one from the validated authority.
            if lowered.startswith("host:"):
                continue
            forwarded.append(line)
        forwarded.append(f"Host: {_host_authority(host, port)}")
        remote_writer.write("\r\n".join(forwarded).encode("iso-8859-1"))
        await remote_writer.drain()
        print(f"allow {method_upper} {host}:{port} peer={peer}", flush=True)
        await _tunnel(reader, writer, remote_reader, remote_writer)
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    except Exception as exc:  # keep malformed requests isolated to one client
        try:
            writer.write(_error_response(502, "Bad Gateway"))
            await writer.drain()
        except Exception:
            pass
        print(f"proxy error peer={peer}: {exc}", flush=True)
    finally:
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


async def serve(host: str, port: int, policy: EgressPolicy) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: _handle_client(reader, writer, policy),
        host,
        port,
        limit=_MAX_HEADER_BYTES,
    )
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(
        f"MuCLI egress proxy listening on {addresses}; "
        f"allow={list(policy.allow)} deny={list(policy.deny)}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="MuCLI allowlist egress proxy")
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3128)
    args = parser.parse_args()
    asyncio.run(serve(args.listen, args.port, EgressPolicy.from_environment()))


if __name__ == "__main__":
    main()
