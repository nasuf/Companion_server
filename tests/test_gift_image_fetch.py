"""后台礼物图片链接转存：直链图片、商品页主图、非图片/过大/内网的拒绝。"""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException

from app.models.admin_gifts import AdminGiftUpdateRequest
from app.services.offline import gift_admin_service as svc
from app.services.offline import gift_image_fetch as fetcher

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    # 传输层是假的，域名校验单独测；这里让所有域名都当作公网。
    monkeypatch.setattr(fetcher, "_ensure_public_http_url", AsyncMock())


def _transport(routes):
    def handler(request: httpx.Request) -> httpx.Response:
        return routes[str(request.url)]()
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_direct_image_link_is_downloaded():
    transport = _transport({
        "https://cdn.example/a.png": lambda: httpx.Response(200, content=PNG, headers={"content-type": "image/png"}),
    })
    assert await fetcher.fetch_gift_image("https://cdn.example/a.png", transport=transport) == (PNG, "image/png")


@pytest.mark.asyncio
async def test_product_page_falls_back_to_its_main_image():
    page = (
        '<html><head><meta name="twitter:card" content="x">'
        '<meta property="og:image" content="/media/memo.jpg?w=800&amp;q=80"></head></html>'
    )
    transport = _transport({
        "https://shop.example/products/memo-2402": lambda: httpx.Response(
            200, text=page, headers={"content-type": "text/html; charset=utf-8"},
        ),
        "https://shop.example/media/memo.jpg?w=800&q=80": lambda: httpx.Response(
            200, content=PNG, headers={"content-type": "image/jpeg"},
        ),
    })
    body, mime = await fetcher.fetch_gift_image("https://shop.example/products/memo-2402", transport=transport)
    assert (body, mime) == (PNG, "image/jpeg")


@pytest.mark.asyncio
async def test_redirects_are_followed_and_rechecked(monkeypatch):
    checked = AsyncMock()
    monkeypatch.setattr(fetcher, "_ensure_public_http_url", checked)
    transport = _transport({
        "https://short.example/x": lambda: httpx.Response(302, headers={"location": "https://cdn.example/a.png"}),
        "https://cdn.example/a.png": lambda: httpx.Response(200, content=PNG, headers={"content-type": "image/png"}),
    })
    await fetcher.fetch_gift_image("https://short.example/x", transport=transport)
    assert [c.args[0] for c in checked.await_args_list] == ["https://short.example/x", "https://cdn.example/a.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("response,message", [
    (lambda: httpx.Response(200, text="<html>no meta</html>", headers={"content-type": "text/html"}), "没找到商品主图"),
    (lambda: httpx.Response(200, content=b"{}", headers={"content-type": "application/json"}), "不是图片"),
    (lambda: httpx.Response(404), "HTTP 404"),
    (lambda: httpx.Response(200, content=b"x" * (fetcher.MAX_IMAGE_BYTES + 1),
                            headers={"content-type": "image/png"}), "10MB"),
    (lambda: httpx.Response(302), "跳转异常"),
])
async def test_unusable_links_explain_why(response, message):
    transport = _transport({"https://shop.example/x": response})
    with pytest.raises(fetcher.GiftImageFetchError, match=message):
        await fetcher.fetch_gift_image("https://shop.example/x", transport=transport)


@pytest.mark.asyncio
async def test_network_errors_become_friendly_failures():
    def boom(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(fetcher.GiftImageFetchError, match="打不开"):
        await fetcher.fetch_gift_image("https://shop.example/x", transport=httpx.MockTransport(boom))


@pytest.mark.asyncio
async def test_only_public_http_addresses_are_allowed(monkeypatch):
    monkeypatch.undo()  # 用真实的地址校验
    for url in ("http://127.0.0.1/a.png", "http://10.0.0.8/a.png", "http://169.254.169.254/latest", "ftp://x/a"):
        with pytest.raises(fetcher.GiftImageFetchError):
            await fetcher._ensure_public_http_url(url)
    await fetcher._ensure_public_http_url("https://8.8.8.8/a.png")


@pytest.mark.asyncio
async def test_admin_maps_fetch_failures_and_formats_to_422(monkeypatch):
    monkeypatch.setattr(svc.admin_repo, "get_gift", AsyncMock(return_value={
        "id": "g", "user_id": "user-1", "status": "shipping", "product_image_url": None,
    }))
    monkeypatch.setattr(svc.admin_repo, "update_gift_details", AsyncMock(return_value={"id": "g"}))
    monkeypatch.setattr(svc, "gift_detail", AsyncMock(return_value={}))

    monkeypatch.setattr(svc, "fetch_gift_image", AsyncMock(side_effect=fetcher.GiftImageFetchError("链接返回的不是图片")))
    with pytest.raises(HTTPException) as not_image:
        await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(image_url="https://shop.example/x"))
    assert not_image.value.status_code == 422 and "不是图片" in not_image.value.detail

    monkeypatch.setattr(svc, "fetch_gift_image", AsyncMock(return_value=(b"GIF89a", "image/gif")))
    with pytest.raises(HTTPException) as gif:
        await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(image_url="https://cdn.example/a.gif"))
    assert gif.value.status_code == 422 and "JPG" in gif.value.detail

    # 已经转存过的本站图片原样保留，不再下载。
    fetch = AsyncMock()
    monkeypatch.setattr(svc, "fetch_gift_image", fetch)
    save = Mock()
    monkeypatch.setattr(svc.activity_media_storage, "save_image_blob", save)
    await svc.update_gift("admin-1", "g", AdminGiftUpdateRequest(image_url="/offline/media/user-1_gift_a.png"))
    fetch.assert_not_awaited()
    save.assert_not_called()


@pytest.mark.asyncio
async def test_redirect_loops_are_cut_off():
    transport = _transport({
        "https://loop.example/a": lambda: httpx.Response(302, headers={"location": "https://loop.example/a"}),
    })
    with pytest.raises(fetcher.GiftImageFetchError, match="跳转次数过多"):
        await fetcher.fetch_gift_image("https://loop.example/a", transport=transport)
