import httpx
import pytest

from app.web_fetch import PublicWebFetcher, WebFetchError


def public_resolver(*_args, **_kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def test_web_fetch_extracts_visible_html_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/article"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><title>港口资料</title>"
                "<script>ignore()</script></head>"
                "<body><article><h1>灯塔</h1><p>建于 1932 年。</p>"
                "</article></body></html>"
            ),
        )

    fetcher = PublicWebFetcher(
        transport=httpx.MockTransport(handler),
        resolver=public_resolver,
    )

    result = fetcher.fetch("https://example.com/article")

    assert result.title == "港口资料"
    assert "灯塔" in result.text
    assert "建于 1932 年" in result.text
    assert "ignore" not in result.text


def test_web_fetch_rejects_private_resolution():
    fetcher = PublicWebFetcher(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, text="secret")
        ),
        resolver=lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("127.0.0.1", 80))
        ],
    )

    with pytest.raises(WebFetchError, match="本机或内网"):
        fetcher.fetch("http://internal.example/path")


def test_web_fetch_revalidates_redirect_targets():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "http://localhost/admin"}
        )

    fetcher = PublicWebFetcher(
        transport=httpx.MockTransport(handler),
        resolver=public_resolver,
    )

    with pytest.raises(WebFetchError, match="本机或内网"):
        fetcher.fetch("https://example.com/start")
