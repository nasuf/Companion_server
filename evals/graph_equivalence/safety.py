"""Fail-closed network fence for opt-in live-model evaluation processes."""

from __future__ import annotations

import socket
from contextlib import ExitStack
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx


class DeniedIO:
    def __getattr__(self, name):
        raise RuntimeError(f"Evaluation attempted forbidden business IO: {name}")


class ModelFence(ExitStack):
    def __init__(self):
        super().__init__()
        self.violations: list[str] = []

    def deny(self, kind: str):
        message = f"Evaluation blocked a non-model {kind}"
        self.violations.append(message)
        raise RuntimeError(message)


def provider_origin(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Evaluation requires an HTTPS model provider origin")
    if parsed.port not in (None, 443):
        raise ValueError("Evaluation requires model provider port 443")
    return parsed.hostname.lower(), 443


def model_network_fence(urls: list[str]) -> ModelFence:
    """Allow only declared provider HTTPS hosts and their resolved addresses.

    HTTP guards prevent redirects/URLs reaching another service on a shared IP.
    Socket guards also block production PG/Redis and non-httpx libraries. DNS
    resolution is restricted after the provider addresses have been resolved.
    No wildcard domains, localhost, business API, proxy or Ollama access.
    """
    origins = {provider_origin(url) for url in urls}
    if not origins:
        raise ValueError("No model providers configured")
    resolve = socket.getaddrinfo
    pinned = {(host, port): resolve(host, port, type=socket.SOCK_STREAM) for host, port in origins}
    destinations = {(result[4][0], port) for (_, port), results in pinned.items() for result in results}
    stack = ModelFence()

    def guarded_resolve(host, port, *args, **kwargs):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if (str(host).lower(), int(port)) not in origins | destinations:
            stack.deny("DNS request")
        key = str(host).lower(), int(port)
        results = pinned.get(key)
        if results is None:
            results = [row for rows in pinned.values() for row in rows if row[4][0] == host]
        # Keep provider IPs fixed for the run. Re-resolving round-robin DNS here
        # could yield an unapproved address and create an evaluation-only retry.
        family = kwargs.get("family", args[0] if args else 0)
        socktype = kwargs.get("type", args[1] if len(args) > 1 else 0)
        return [row for row in results if (not family or row[0] == family) and
                (not socktype or row[1] == socktype)]

    def connect_guard(original):
        def connect(sock, address):
            if not isinstance(address, tuple) or (address[0], address[1]) not in destinations:
                stack.deny("socket")
            return original(sock, address)
        return connect

    def request_allowed(request):
        if (request.url.scheme != "https" or
            (request.url.host.lower(), request.url.port or 443) not in origins):
            stack.deny("HTTP request")

    # Fence every hop, including redirects and pooled connections; guarding
    # public send() alone would inspect only the first request in a redirect.
    async_send = httpx.AsyncClient._send_single_request
    sync_send = httpx.Client._send_single_request

    async def send_async(client, request, *args, **kwargs):
        request_allowed(request)
        return await async_send(client, request, *args, **kwargs)

    def send_sync(client, request, *args, **kwargs):
        request_allowed(request)
        return sync_send(client, request, *args, **kwargs)

    stack.enter_context(patch.object(socket, "getaddrinfo", guarded_resolve))
    stack.enter_context(patch.object(socket.socket, "connect", connect_guard(socket.socket.connect)))
    stack.enter_context(patch.object(socket.socket, "connect_ex", connect_guard(socket.socket.connect_ex)))
    stack.enter_context(patch.object(httpx.AsyncClient, "_send_single_request", send_async))
    stack.enter_context(patch.object(httpx.Client, "_send_single_request", send_sync))
    return stack
