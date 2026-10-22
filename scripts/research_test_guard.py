"""Opt-in pytest guard for this advanced CPU regression run.

Denies external networking and known local service ports. High loopback ports
remain available for Windows asyncio socket-pair wakeups/test servers.
"""
import ipaddress
import json
from pathlib import Path
import socket

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo
_denied = 0


def _allowed(address):
    if isinstance(address, str):
        return False
    try:
        host, port = address[:2]
        return ipaddress.ip_address(host).is_loopback and 49152 <= int(port) <= 65535
    except (ValueError, TypeError):
        return False


def _check(address):
    global _denied
    if not _allowed(address):
        _denied += 1
        raise AssertionError("Advanced CPU tests deny external networking and application service calls")


def pytest_sessionstart(session):
    def guarded_connect(instance, address):
        _check(address)
        return _connect(instance, address)
    def guarded_connect_ex(instance, address):
        _check(address)
        return _connect_ex(instance, address)
    def guarded_dns(host, port, *args, **kwargs):
        if host not in (None, "localhost", "127.0.0.1", "::1"):
            _check((host, port or 0))
        return _getaddrinfo(host, port, *args, **kwargs)
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_dns


def pytest_sessionfinish(session, exitstatus):
    path = Path(__file__).resolve().parents[1] / "reports/research_execution_2026-10-16/network_guard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scope": "external network and local service ports denied; ephemeral loopback allowed",
                                "denied_attempts": _denied, "pytest_exitstatus": int(exitstatus)}, indent=2), encoding="utf-8")
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.getaddrinfo = _getaddrinfo
