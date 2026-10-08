"""Bounded public-web fetching for untrusted link pages and cover URLs.

The public HTTPcore network-backend API pins each TCP connection to a checked
IP. The original hostname remains in HTTPcore's origin for Host and TLS/SNI.
HTTPX handles redirects, each of which passes through the same transport.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import ipaddress
import socket
import ssl
from typing import AsyncIterable
import zlib

import certifi
import httpcore
import httpx

PAGE_MAX_BYTES = 6_000_000  # Existing HTML parser accepts at most 1.5M characters.
COVER_MAX_BYTES = 10 * 1024 * 1024
_NAT64_NETWORKS = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


def _public_address(raw: str) -> str:
    try:
        address = ipaddress.ip_address(raw)
    except ValueError as exc:
        raise httpcore.ConnectError("link destination has an invalid address") from exc
    if not address.is_global or address.is_multicast or address.is_reserved:
        raise httpcore.ConnectError("link destination must be a public address")
    if isinstance(address, ipaddress.IPv6Address):
        if (address.scope_id or address.ipv4_mapped or address.sixtofour or address.teredo
                or any(address in network for network in _NAT64_NETWORKS)):
            raise httpcore.ConnectError("link destination cannot use an IPv6 translation or scope")
    return str(address)


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve once, reject mixed public/private answers, connect by IP literal."""

    def __init__(self) -> None:
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(
        self, host: str, port: int, timeout: float | None = None,
        local_address: str | None = None, socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        loop = asyncio.get_running_loop()
        budget = timeout if timeout is not None else 12.0
        deadline = loop.time() + budget
        try:
            async with asyncio.timeout(budget):
                try:
                    literal = ipaddress.ip_address(host)
                except ValueError:
                    literal = None
                if literal is not None:
                    addresses = [_public_address(host)]
                else:
                    results = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                    addresses = list(dict.fromkeys(_public_address(result[4][0]) for result in results))
                if not addresses:
                    raise httpcore.ConnectError("link destination has no public addresses")
                ipv6 = next((value for value in addresses if ":" in value), None)
                ipv4 = next((value for value in addresses if ":" not in value), None)
                if ipv6 is not None and ipv4 is not None:
                    addresses = [ipv6, ipv4, *(value for value in addresses if value not in {ipv6, ipv4})]
                return await self._connect_addresses(
                    addresses, port, deadline, local_address, socket_options,
                )
        except TimeoutError as exc:
            raise httpcore.ConnectTimeout("link connection timed out") from exc
        except OSError as exc:
            raise httpcore.ConnectError("link destination could not be resolved") from exc

    async def _connect_addresses(
        self, addresses: list[str], port: int, deadline: float,
        local_address: str | None, socket_options,
    ) -> httpcore.AsyncNetworkStream:
        # Race checked IPs with the same 250ms stagger as ordinary Happy Eyeballs.
        # A blackholed IPv6 answer must not consume the entire IPv4 connect budget.
        async def attempt(index, address):
            if index:
                await asyncio.sleep(0.25 * index)
            return await self._backend.connect_tcp(
                address, port, timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                local_address=local_address, socket_options=socket_options,
            )

        tasks = [asyncio.create_task(attempt(index, address)) for index, address in enumerate(addresses)]
        selected = None
        last_error = None
        try:
            try:
                pending = set(tasks)
                while pending and selected is None:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in tasks:
                        if task not in done:
                            continue
                        try:
                            candidate = task.result()
                        except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                            last_error = exc
                        else:
                            selected = candidate
                            break
                if selected is None:
                    raise last_error or httpcore.ConnectError("link connection failed")
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, httpcore.AsyncNetworkStream) and result is not selected:
                        await result.aclose()
        except BaseException:
            if selected is not None:
                await selected.aclose()
            raise
        return selected

    async def connect_unix_socket(self, *args, **kwargs) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("link fetch cannot use a Unix socket")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


@contextmanager
def _httpx_errors(request: httpx.Request):
    # Map specific subclasses first; no private HTTPX/httpcore APIs are used.
    errors = (
        (httpcore.ConnectTimeout, httpx.ConnectTimeout),
        (httpcore.ReadTimeout, httpx.ReadTimeout),
        (httpcore.WriteTimeout, httpx.WriteTimeout),
        (httpcore.PoolTimeout, httpx.PoolTimeout),
        (httpcore.ConnectError, httpx.ConnectError),
        (httpcore.ReadError, httpx.ReadError),
        (httpcore.WriteError, httpx.WriteError),
        (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
        (httpcore.LocalProtocolError, httpx.LocalProtocolError),
        (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    )
    try:
        yield
    except tuple(source for source, _ in errors) as exc:
        for source, target in errors:
            if isinstance(exc, source):
                # Do not echo untrusted URLs or DNS answers into a card/log error.
                raise target("link fetch transport failed", request=request) from exc
        raise


async def _bounded_body(stream: AsyncIterable[bytes], headers: httpx.Headers, limit: int) -> bytes:
    encoding = headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"identity", "gzip", "deflate"}:
        raise httpx.DecodingError("unsupported link content encoding")
    wire_limit = limit + 64 * 1024
    length = headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError as exc:
            raise httpx.RemoteProtocolError("invalid link content length") from exc
        if declared < 0 or declared > (limit if encoding == "identity" else wire_limit):
            raise httpx.DecodingError("link response exceeds the download limit")
    body = bytearray()
    wire_size = 0
    decoder = zlib.decompressobj(31) if encoding == "gzip" else None
    # Deflate's two common formats are selected from the first two wire bytes.
    prefix = bytearray()
    async for chunk in stream:
        wire_size += len(chunk)
        if wire_size > wire_limit:
            raise httpx.DecodingError("link response exceeds the download limit")
        if encoding == "identity":
            if len(body) + len(chunk) > limit:
                raise httpx.DecodingError("link response exceeds the download limit")
            body.extend(chunk)
            continue
        if encoding == "deflate" and decoder is None:
            prefix.extend(chunk)
            if len(prefix) < 2:
                continue
            wrapped = (prefix[0] & 15 == 8 and (prefix[0] * 256 + prefix[1]) % 31 == 0)
            decoder = zlib.decompressobj(15 if wrapped else -15)
            chunk = bytes(prefix)
            prefix.clear()
        try:
            if decoder.eof:
                # Concatenated gzip members are valid; trailing deflate data is not.
                if encoding != "gzip":
                    raise httpx.DecodingError("invalid link compressed body")
                decoder = zlib.decompressobj(31)
            while chunk:
                body.extend(decoder.decompress(chunk, limit - len(body) + 1))
                if len(body) > limit:
                    raise httpx.DecodingError("link response exceeds the download limit")
                if decoder.unused_data:
                    if encoding != "gzip":
                        raise httpx.DecodingError("invalid link compressed body")
                    chunk = decoder.unused_data
                    decoder = zlib.decompressobj(31)
                else:
                    chunk = decoder.unconsumed_tail
        except zlib.error as exc:
            raise httpx.DecodingError("invalid link compressed body") from exc
    if encoding != "identity" and (decoder is None or not decoder.eof):
        raise httpx.DecodingError("incomplete link compressed body")
    return bytes(body)


class PublicFetchTransport(httpx.AsyncBaseTransport):
    """Small, bounded GET/HEAD bridge using the pinned public HTTPcore API."""

    def __init__(self, *, max_bytes: int, total_timeout: float = 36.0) -> None:
        if max_bytes <= 0 or total_timeout <= 0:
            raise ValueError("positive link fetch limits are required")
        self._limit = max_bytes
        self._total_timeout = total_timeout
        self._deadline: float | None = None
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(cafile=certifi.where()),
            network_backend=PublicNetworkBackend(),
            max_connections=4, max_keepalive_connections=2, retries=0,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if (request.method not in {"GET", "HEAD"} or url.scheme not in {"http", "https"}
                or not url.host or url.userinfo or "%" in url.host):
            raise httpx.UnsupportedProtocol("invalid public link request", request=request)
        loop = asyncio.get_running_loop()
        if self._deadline is None:
            self._deadline = loop.time() + self._total_timeout
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(scheme=url.raw_scheme, host=url.raw_host, port=url.port, target=url.raw_path),
            headers=request.headers.raw, content=request.stream, extensions=request.extensions,
        )
        try:
            with _httpx_errors(request):
                async with asyncio.timeout(max(0.0, self._deadline - loop.time())):
                    response = await self._pool.handle_async_request(core_request)
                    try:
                        headers = httpx.Headers(response.headers)
                        body = (b"" if request.method == "HEAD" or response.status in {204, 304} else
                                await _bounded_body(response.stream, headers, self._limit))
                    finally:
                        await response.aclose()
        except TimeoutError as exc:
            raise httpx.ReadTimeout("link fetch exceeded its total time budget", request=request) from exc
        headers.pop("content-encoding", None)
        headers.pop("transfer-encoding", None)
        headers["content-length"] = str(len(body))
        return httpx.Response(response.status, headers=headers, content=body, extensions=response.extensions)

    async def aclose(self) -> None:
        await self._pool.aclose()


def public_link_client(
    *, timeout: float | httpx.Timeout = 12.0, headers: dict[str, str] | None = None,
    max_bytes: int = PAGE_MAX_BYTES, follow_redirects: bool = True,
) -> httpx.AsyncClient:
    """Dedicated client: environment proxies never bypass destination checks."""
    fetch_headers = {**(headers or {}), "accept-encoding": "gzip, deflate"}
    return httpx.AsyncClient(
        transport=PublicFetchTransport(max_bytes=max_bytes),
        timeout=timeout, headers=fetch_headers, follow_redirects=follow_redirects,
        max_redirects=10, trust_env=False,
    )
