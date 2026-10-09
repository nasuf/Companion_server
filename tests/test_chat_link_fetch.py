"""Offline public-link protocol tests; no provider or private-network probes."""
import asyncio
import gzip
import socket
import zlib

import httpcore
import httpx
import pytest

from app.services.chat_links import fetch, extraction, covers


class WireStream(httpcore.AsyncNetworkStream):
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.writes = []
        self.closed = False
        self.tls_host = None

    async def read(self, max_bytes, timeout=None):
        return self.chunks.pop(0) if self.chunks else b""

    async def write(self, buffer, timeout=None):
        self.writes.append(buffer)

    async def aclose(self):
        self.closed = True

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        assert ssl_context.check_hostname
        assert ssl_context.verify_mode != 0
        self.tls_host = server_hostname
        return self

    def get_extra_info(self, info):
        return None


class WireBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.streams = []

    async def connect_tcp(self, host, port, **kwargs):
        self.calls.append((host, port, kwargs))
        if not self.responses:
            raise AssertionError("unexpected connection")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        stream = WireStream(response)
        self.streams.append(stream)
        return stream

    async def sleep(self, seconds):
        pass


def wire(body=b"ok", *, status=200, headers=()):
    rows = [(b"content-length", str(len(body)).encode()), (b"connection", b"close"), *headers]
    return [f"HTTP/1.1 {status} Reply\r\n".encode() + b"".join(k + b": " + v + b"\r\n" for k, v in rows) + b"\r\n", body]


@pytest.fixture
async def harness(monkeypatch):
    dns_calls = []
    dns = {}

    async def resolve(host, port, **kwargs):
        dns_calls.append(host)
        addresses = dns.get(host, ["93.184.215.14"])
        if isinstance(addresses, Exception):
            raise addresses
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in addresses]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)

    def install(responses):
        backend = WireBackend(responses)
        monkeypatch.setattr(fetch.httpcore, "AnyIOBackend", lambda: backend)
        return backend

    return install, dns, dns_calls


@pytest.mark.parametrize("address", [
    "127.0.0.1", "0.0.0.0", "10.0.0.1", "172.16.1.2", "192.168.1.2", "169.254.169.254",
    "100.64.0.1", "192.0.2.1", "198.18.0.1", "224.0.0.1", "255.255.255.255",
    "::", "::1", "fc00::1", "fe80::1", "ff02::1", "2001:db8::1", "fe80::1%eth0",
    "::ffff:127.0.0.1", "::ffff:8.8.8.8", "2002:7f00:1::", "2001:0:4136:e378:8000:63bf:3fff:fdd2",
    "64:ff9b::7f00:1", "64:ff9b:1::a00:1", "invalid",
])
def test_non_public_or_translated_addresses_are_rejected(address):
    with pytest.raises(httpcore.ConnectError):
        fetch._public_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.215.14", "2606:4700:4700::1111"])
def test_public_addresses_remain_supported(address):
    assert fetch._public_address(address) == address


async def test_tcp_uses_validated_ip_but_preserves_host_and_tls_name(harness):
    install, dns, dns_calls = harness
    backend = install([wire(b"hello")])
    dns["public.example"] = ["93.184.215.14"]
    async with fetch.public_link_client() as client:
        response = await client.get("https://public.example/article?a=1")
    assert response.text == "hello"
    assert backend.calls[0][:2] == ("93.184.215.14", 443)
    assert dns_calls == ["public.example"]
    assert backend.streams[0].tls_host == "public.example"
    request = b"".join(backend.streams[0].writes)
    assert b"Host: public.example" in request
    assert b"GET /article?a=1 HTTP/1.1" in request
    assert backend.streams[0].closed


@pytest.mark.parametrize("addresses", [["10.0.0.1"], ["93.184.215.14", "127.0.0.1"], []])
async def test_dns_mixed_or_private_answers_never_connect(harness, addresses):
    install, dns, _ = harness
    backend = install([])
    dns["unsafe.example"] = addresses
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://unsafe.example/")
    assert backend.calls == []


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://2130706433/", "http://localhost/", "http://[::1]/"])
async def test_literal_and_legacy_numeric_local_hosts_are_blocked(harness, url):
    install, dns, _ = harness
    backend = install([])
    dns[httpx.URL(url).host] = ["127.0.0.1"]
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(url)
    assert not backend.calls


async def test_dns_rebinding_cannot_change_destination_between_check_and_connect(harness):
    install, dns, dns_calls = harness
    backend = install([wire()])
    original = backend.connect_tcp

    async def connect(host, port, **kwargs):
        dns["public.example"] = ["127.0.0.1"]
        return await original(host, port, **kwargs)

    backend.connect_tcp = connect
    async with fetch.public_link_client() as client:
        assert (await client.get("http://public.example/")).status_code == 200
    assert dns_calls == ["public.example"]
    assert backend.calls[0][0] == "93.184.215.14"


async def test_public_short_link_redirects_still_work(harness):
    install, dns, dns_calls = harness
    backend = install([wire(status=302, headers=[(b"location", b"https://www.zhihu.com/question/1")]), wire(b"<title>question</title>")])
    async with fetch.public_link_client() as client:
        response = await client.get("https://short.example/a")
    assert str(response.url) == "https://www.zhihu.com/question/1"
    assert len(response.history) == 1
    assert dns_calls == ["short.example", "www.zhihu.com"]
    assert len(backend.calls) == 2


@pytest.mark.parametrize("target", ["http://127.0.0.1/secret", "http://internal.example/", "http://[::1]/", "https://user:pass@public.example/"])
async def test_every_redirect_is_checked_before_connect(harness, target):
    install, dns, _ = harness
    dns["internal.example"] = ["10.0.0.1"]
    backend = install([wire(status=302, headers=[(b"location", target.encode())])])
    async with fetch.public_link_client() as client:
        with pytest.raises((httpx.ConnectError, httpx.UnsupportedProtocol)):
            await client.get("https://short.example/a")
    assert len(backend.calls) == 1


async def test_environment_proxy_cannot_bypass_checks(monkeypatch, harness):
    install, _, _ = harness
    backend = install([])
    monkeypatch.setenv("HTTP_PROXY", "http://93.184.215.14:8080")
    monkeypatch.setenv("ALL_PROXY", "http://93.184.215.14:8080")
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://127.0.0.1/")
    assert not backend.calls


@pytest.mark.parametrize("url", ["https://u:p@public.example/", "ftp://public.example/file", "http://[fe80::1%25eth0]/"])
async def test_invalid_url_credentials_and_scope_fail_before_connection(harness, url):
    install, _, _ = harness
    backend = install([])
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.UnsupportedProtocol):
            await client.get(url)
    assert not backend.calls


async def test_public_ipv6_and_fallback_addresses(harness):
    install, dns, _ = harness
    dns["public.example"] = ["2606:4700:4700::1111", "93.184.215.14"]
    backend = install([httpcore.ConnectError("first unavailable"), wire()])
    async with fetch.public_link_client() as client:
        assert (await client.get("http://public.example/")).text == "ok"
    assert [c[0] for c in backend.calls] == dns["public.example"]


async def test_dns_errors_are_bounded_and_sanitized(harness):
    install, dns, _ = harness
    dns["public.example"] = socket.gaierror("sensitive resolver detail")
    backend = install([])
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.ConnectError, match="transport failed"):
            await client.get("http://public.example/")
    assert not backend.calls


async def test_dns_resolution_uses_connect_timeout(monkeypatch, harness):
    install, _, _ = harness
    backend = install([])

    async def slow(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", slow)
    async with fetch.public_link_client(timeout=httpx.Timeout(0.01)) as client:
        with pytest.raises(httpx.ConnectTimeout):
            await client.get("http://public.example/")
    assert not backend.calls


async def chunks(*values):
    for value in values:
        yield value


@pytest.mark.parametrize("body", [b"", b"abc", b"a" * 16])
async def test_identity_bodies_within_limit(body):
    assert await fetch._bounded_body(chunks(body[:1], body[1:]), httpx.Headers(), 16) == body


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "raw-deflate", "concat-gzip"])
async def test_compressed_bodies_decode_with_split_headers_and_no_double_decoding(harness, encoding):
    install, _, _ = harness
    body = "正常的链接正文".encode()
    if encoding == "gzip":
        encoded = gzip.compress(body)
    elif encoding == "deflate":
        encoded = zlib.compress(body)
    elif encoding == "raw-deflate":
        encoder = zlib.compressobj(wbits=-15)
        encoded = encoder.compress(body) + encoder.flush()
    else:
        encoded = gzip.compress(body[:3]) + gzip.compress(body[3:])
    header = "gzip" if "gzip" in encoding else "deflate"
    backend = install([wire(encoded, headers=[(b"content-encoding", header.encode())])])
    async with fetch.public_link_client(max_bytes=len(body)) as client:
        response = await client.get("https://public.example/")
    assert response.content == body
    assert "content-encoding" not in response.headers
    assert response.headers["content-length"] == str(len(body))
    assert backend.streams[0].closed
    assert await fetch._bounded_body(chunks(*[bytes([v]) for v in encoded]), httpx.Headers({"content-encoding": header}), len(body)) == body


@pytest.mark.parametrize("encoding", ["identity", "gzip", "deflate"])
async def test_download_and_decompression_limit_interrupts_and_closes(harness, encoding):
    install, _, _ = harness
    body = b"a" * 2_000_000
    encoded = gzip.compress(body) if encoding == "gzip" else zlib.compress(body) if encoding == "deflate" else body
    backend = install([wire(encoded, headers=[(b"content-encoding", encoding.encode())])])
    async with fetch.public_link_client(max_bytes=1024) as client:
        with pytest.raises(httpx.DecodingError, match="download limit"):
            await client.get("https://public.example/")
    assert backend.streams[0].closed


async def test_chunked_oversize_without_content_length_closes_stream(harness):
    install, _, _ = harness
    backend = install([[b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"4\r\nabcd\r\n", b"4\r\nefgh\r\n", b"0\r\n\r\n"]])
    async with fetch.public_link_client(max_bytes=6) as client:
        with pytest.raises(httpx.DecodingError):
            await client.get("http://public.example/")
    assert backend.streams[0].closed
    assert backend.streams[0].chunks  # stopped before reading the terminator


@pytest.mark.parametrize("header,body", [
    ("gzip", gzip.compress(b"abc")[:-2]), ("gzip", b"invalid"),
    ("deflate", zlib.compress(b"abc")[:-2]), ("deflate", zlib.compress(b"abc") + b"junk"),
    ("br", b"x"), ("gzip, deflate", b"x"),
])
async def test_corrupt_truncated_and_unsupported_encoding_fails_closed(header, body):
    with pytest.raises(httpx.DecodingError):
        await fetch._bounded_body(chunks(body), httpx.Headers({"content-encoding": header}), 20)


@pytest.mark.parametrize("declared", ["-1", "9999999", "invalid"])
async def test_declared_content_length_checked_before_read(declared):
    async def unread():
        raise AssertionError("must not start reading")
        yield b""

    with pytest.raises((httpx.RemoteProtocolError, httpx.DecodingError)):
        await fetch._bounded_body(unread(), httpx.Headers({"content-length": declared}), 20)


async def test_compressed_wire_budget_is_separate_from_decoded_size():
    with pytest.raises(httpx.DecodingError, match="download limit"):
        await fetch._bounded_body(chunks(b"x" * 70_000), httpx.Headers({"content-encoding": "gzip"}), 10)


async def test_head_advisory_size_does_not_block_short_link_resolution(harness):
    install, _, _ = harness
    backend = install([[b"HTTP/1.1 302 Found\r\nLocation: https://www.bilibili.com/video/BV1abc\r\nContent-Length: 99999999\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\n"]])
    result = await extraction._resolve_bilibili_final_url("https://b23.tv/a", timeout=12)
    assert result == "https://www.bilibili.com/video/BV1abc"
    assert b"HEAD /a HTTP/1.1" in b"".join(backend.streams[0].writes)


async def test_total_deadline_covers_slow_body_and_closes_stream(harness):
    install, _, _ = harness
    backend = install([wire()])
    original = backend.connect_tcp

    async def connect(*args, **kwargs):
        stream = await original(*args, **kwargs)
        read = stream.read

        async def slow_read(*args, **kwargs):
            await asyncio.sleep(0.03)
            return await read(*args, **kwargs)

        stream.read = slow_read
        return stream

    backend.connect_tcp = connect
    transport = fetch.PublicFetchTransport(max_bytes=1024, total_timeout=0.05)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ReadTimeout, match="total time"):
            await client.get("https://public.example/")
    assert backend.streams[0].closed


@pytest.mark.parametrize("host", ["xhslink.com", "v.douyin.com", "weibo.com", "www.toutiao.com", "www.zhihu.com"])
async def test_platform_parsing_through_actual_httpcore_transport(harness, host):
    install, _, _ = harness
    # Weibo/Toutiao attempt an API fallback; it may fail without affecting HTML.
    install([wire(b'<html><title>safe title</title><meta property="og:description" content="body"></html>')])
    result = await extraction.extract_link_metadata(url=f"https://{host}/article", shared_text="看看这个")
    assert result.title == "safe title"
    assert result.status == "ready"


async def test_bilibili_api_uses_guarded_transport(harness):
    install, _, dns_calls = harness
    install([wire(b'{"code":0,"data":{"title":"video","desc":"body"}}')])
    result = await extraction.extract_link_metadata(url="https://www.bilibili.com/video/BV1abc", shared_text="video")
    assert result.title == "video"
    assert dns_calls == ["api.bilibili.com"]


async def test_unsafe_link_keeps_shared_text_as_partial_card_without_network(harness):
    install, _, _ = harness
    backend = install([])
    result = await extraction.extract_link_metadata(url="http://127.0.0.1/", shared_text="分享说明")
    assert result.status == "partial"
    assert result.content_text == "分享说明"
    assert len(backend.calls) == 0


async def test_cover_fetch_is_guarded_and_keeps_nonfatal_fallback(harness, monkeypatch):
    install, _, _ = harness
    backend = install([])
    metadata = extraction.LinkMetadata(source_url="https://public.example/", final_url="https://public.example/", platform="网页", title="cover", image_url="http://169.254.169.254/latest/")
    result = await covers.cache_link_cover(user_id="u1", metadata=metadata)
    assert result.metadata is metadata
    assert result.extra_metadata == {"remote_image_url": metadata.image_url}
    assert not backend.calls


async def test_valid_image_download_preserves_mime_and_bytes(harness, monkeypatch):
    install, _, _ = harness
    image = b"image bytes"
    install([wire(image, headers=[(b"content-type", b"image/png")])])
    monkeypatch.setattr(covers.storage, "validate_image_size", lambda blob: None)
    blob, mime = await covers._download_image("https://images.example/a.png")
    assert (blob, mime) == (image, "image/png")


async def test_redirect_limit_remains_bounded(harness):
    install, _, _ = harness
    backend = install([wire(status=302, headers=[(b"location", b"/again")])] * 12)
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.TooManyRedirects):
            await client.get("https://public.example/")
    assert len(backend.calls) == 11


async def test_real_socket_http_redirect_cookie_and_body_flow(monkeypatch):
    """Real HTTPcore/AnyIO/H11 over isolated loopback, after the public-IP guard.

    Only the test dialer maps its already-validated public IP to a local server;
    production has no mapping or bypass option. No external socket is opened.
    """
    requests = []
    public_dials = []
    connections = []

    async def serve(reader, writer):
        connections.append(writer)
        try:
            data = await reader.readuntil(b"\r\n\r\n")
            requests.append(data)
            if data.startswith(b"GET /short "):
                reply = wire(status=302, headers=[(b"location", b"/article"), (b"set-cookie", b"visitor=ok; Path=/")])
            else:
                assert b"Cookie: visitor=ok" in data
                body = gzip.compress("本地协议测试正文".encode())
                reply = wire(body, headers=[(b"content-encoding", b"gzip"), (b"content-type", b"text/html; charset=utf-8")])
            for chunk in reply:
                writer.write(chunk)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    real_backend = httpcore.AnyIOBackend()

    class LocalDialBackend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **kwargs):
            assert host == "93.184.215.14"
            public_dials.append(host)
            return await real_backend.connect_tcp("127.0.0.1", port, **kwargs)

        async def sleep(self, seconds):
            await real_backend.sleep(seconds)

    async def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))]

    monkeypatch.setattr(fetch.httpcore, "AnyIOBackend", LocalDialBackend)
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    # AnyIO resolves literals through getaddrinfo too; permit only this local
    # test dialer's explicit 127.0.0.1 without changing PublicNetworkBackend.
    resolver = asyncio.get_running_loop().getaddrinfo

    async def resolve_dial(host, port, **kwargs):
        if host == "127.0.0.1":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]
        return await resolver(host, port, **kwargs)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve_dial)
    try:
        async with fetch.public_link_client() as client:
            response = await client.get(f"http://public.example:{port}/short")
        assert response.text == "本地协议测试正文"
        assert len(response.history) == 1
        assert len(requests) == len(public_dials) == 2
        assert all(f"Host: public.example:{port}".encode() in data for data in requests)
    finally:
        server.close()
        await server.wait_closed()
        for connection in connections:
            connection.close()


async def test_cancelled_body_closes_stream(harness):
    install, _, _ = harness
    backend = install([wire()])
    original = backend.connect_tcp
    entered = asyncio.Event()

    async def connect(*args, **kwargs):
        stream = await original(*args, **kwargs)
        read = stream.read
        count = 0

        async def waiting_read(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                entered.set()
                await asyncio.Event().wait()
            return await read(*args, **kwargs)

        stream.read = waiting_read
        return stream

    backend.connect_tcp = connect
    async with fetch.public_link_client() as client:
        task = asyncio.create_task(client.get("https://public.example/"))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert backend.streams[0].closed


async def test_total_budget_is_shared_across_redirects(harness):
    install, _, _ = harness
    backend = install([wire(status=302, headers=[(b"location", b"/next")]), wire()])
    original = backend.connect_tcp

    async def connect(*args, **kwargs):
        await asyncio.sleep(0.03)
        return await original(*args, **kwargs)

    backend.connect_tcp = connect
    transport = fetch.PublicFetchTransport(max_bytes=1024, total_timeout=0.05)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        with pytest.raises(httpx.ReadTimeout, match="total time"):
            await client.get("https://public.example/")
    assert len(backend.calls) == 1
    assert backend.streams[0].closed


async def test_cover_limit_applies_to_chunked_download_before_storage(harness, monkeypatch):
    install, _, _ = harness
    # No declared length: the read limit, rather than a metadata hint, must stop it.
    size = covers._MAX_COVER_BYTES + 1
    backend = install([[b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: image/png\r\n\r\n", f"{size:x}\r\n".encode(), b"a" * size, b"\r\n0\r\n\r\n"]])
    def forbidden(*args, **kwargs):
        raise AssertionError("oversize data must not reach image storage")
    monkeypatch.setattr(covers.storage, "validate_image_size", forbidden)
    with pytest.raises(httpx.DecodingError, match="download limit"):
        await covers._download_image("https://images.example/large.png")
    assert backend.streams[0].closed


async def test_blackholed_ipv6_does_not_block_ipv4_connection(harness):
    install, dns, _ = harness
    dns["public.example"] = ["2606:4700:4700::1111", "2606:4700:4700::1001", "93.184.215.14"]
    backend = install([wire()])
    original = backend.connect_tcp
    ipv6_cancelled = asyncio.Event()

    async def connect(host, port, **kwargs):
        if ":" in host:
            try:
                await asyncio.Event().wait()
            finally:
                ipv6_cancelled.set()
        return await original(host, port, **kwargs)

    backend.connect_tcp = connect
    async with fetch.public_link_client(timeout=httpx.Timeout(0.4)) as client:
        assert (await client.get("http://public.example/")).text == "ok"
    assert ipv6_cancelled.is_set()
    assert backend.calls[0][0] == "93.184.215.14"


@pytest.mark.parametrize("status", [204, 304])
async def test_bodyless_status_does_not_decode_advisory_gzip(harness, status):
    install, _, _ = harness
    backend = install([[f"HTTP/1.1 {status} Reply\r\nContent-Length: 9999999\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\n".encode()]])
    async with fetch.public_link_client(max_bytes=1024) as client:
        response = await client.get("https://public.example/")
    assert response.status_code == status
    assert response.content == b""
    assert backend.streams[0].closed


@pytest.mark.parametrize("certificate_host", ["public.example", "wrong.example"])
async def test_real_tls_checks_original_hostname_with_ip_pinned_dial(monkeypatch, tmp_path, certificate_host):
    from datetime import datetime, timedelta, timezone
    import ssl
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, certificate_host)])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(certificate_host)]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    real_backend = httpcore.AnyIOBackend()
    dials, requests = [], []

    async def serve(reader, writer):
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"".join(wire(b"tls verified")))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    class DialBackend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **kwargs):
            assert host == "93.184.215.14"
            dials.append(host)
            return await real_backend.connect_tcp("127.0.0.1", port, **kwargs)
        async def sleep(self, seconds):
            await real_backend.sleep(seconds)
    async def resolve(host, port, **kwargs):
        address = "127.0.0.1" if host == "127.0.0.1" else "93.184.215.14"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    # This local test alone supplies a disposable trusted root. Production
    # always uses certifi's checked bundle with certificate/hostname validation.
    monkeypatch.setattr(fetch.certifi, "where", lambda: str(cert_path))
    monkeypatch.setattr(fetch.httpcore, "AnyIOBackend", DialBackend)
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    try:
        async with fetch.public_link_client() as client:
            if certificate_host == "public.example":
                response = await client.get(f"https://public.example:{port}/")
                assert response.text == "tls verified"
            else:
                with pytest.raises(httpx.ConnectError):
                    await client.get(f"https://public.example:{port}/")
        assert len(dials) == 1
        assert len(requests) == (1 if certificate_host == "public.example" else 0)
    finally:
        server.close()
        await server.wait_closed()


async def test_large_dns_answer_has_bounded_connection_attempts(harness, monkeypatch):
    install, dns, _ = harness
    # A domain may publish far more addresses than the pool's connection budget.
    dns["many.example"] = [f"93.184.215.{n}" for n in range(1, 65)] + ["2606:4700:4700::1111"]
    backend = install([httpcore.ConnectError("synthetic unavailable")] * 65)

    async def no_stagger(_):
        pass

    monkeypatch.setattr(fetch.asyncio, "sleep", no_stagger)
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("https://many.example/")
    assert len(backend.calls) == 8
    assert backend.calls[0][0] == "2606:4700:4700::1111"
    assert backend.calls[1][0] == "93.184.215.1"
    assert not backend.streams


async def test_private_dns_answer_after_connection_cap_still_rejects_all(harness):
    install, dns, _ = harness
    dns["mixed.example"] = [f"93.184.215.{n}" for n in range(1, 65)] + ["127.0.0.1"]
    backend = install([])
    async with fetch.public_link_client() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("https://mixed.example/")
    assert not backend.calls
