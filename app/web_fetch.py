from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx


class WebFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebFetchResult:
    title: str
    url: str
    text: str
    content_type: str

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "content_type": self.content_type,
        }


Resolver = Callable[..., Iterable[tuple[Any, ...]]]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in {
            "p",
            "div",
            "article",
            "section",
            "br",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "tr",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "article", "section", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        clean = str(data or "")
        if self._in_title:
            self.title_parts.append(clean)
        self.parts.append(clean)

    def result(self) -> tuple[str, str]:
        title = re.sub(r"\s+", " ", "".join(self.title_parts)).strip()
        lines = []
        for line in "".join(self.parts).splitlines():
            clean = re.sub(r"[ \t\f\v]+", " ", line).strip()
            if clean:
                lines.append(clean)
        return title, "\n".join(lines)


class PublicWebFetcher:
    """Fetch public text pages while rejecting local and private targets."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver = socket.getaddrinfo,
        timeout_seconds: float = 20,
        max_redirects: int = 3,
        max_bytes: int = 1_000_000,
    ):
        self.transport = transport
        self.resolver = resolver
        self.timeout_seconds = timeout_seconds
        self.max_redirects = max_redirects
        self.max_bytes = max_bytes

    def fetch(self, url: str, *, max_chars: int = 16_000) -> WebFetchResult:
        current_url = self._validate_url(url)
        limit = min(48_000, max(1_000, int(max_chars)))
        try:
            with httpx.Client(
                transport=self.transport,
                timeout=self.timeout_seconds,
                follow_redirects=False,
                headers={
                    "User-Agent": "Readraft/0.1 (+public-web-fetch)",
                    "Accept": "text/html,text/plain,application/json;q=0.9",
                },
            ) as client:
                for redirect_count in range(self.max_redirects + 1):
                    with client.stream("GET", current_url) as response:
                        if response.status_code in {
                            301,
                            302,
                            303,
                            307,
                            308,
                        }:
                            location = str(
                                response.headers.get("location") or ""
                            ).strip()
                            if not location:
                                raise WebFetchError("网页重定向缺少目标地址")
                            if redirect_count >= self.max_redirects:
                                raise WebFetchError("网页重定向次数过多")
                            current_url = self._validate_url(
                                urljoin(current_url, location)
                            )
                            continue
                        if response.status_code >= 400:
                            raise WebFetchError(
                                "网页读取失败"
                                f"（HTTP {response.status_code}）"
                            )
                        content_type = str(
                            response.headers.get("content-type") or ""
                        ).lower()
                        if not any(
                            value in content_type
                            for value in (
                                "text/",
                                "application/json",
                                "application/xhtml+xml",
                            )
                        ):
                            raise WebFetchError("网页不是可读取的文本内容")
                        chunks: list[bytes] = []
                        byte_count = 0
                        for chunk in response.iter_bytes():
                            byte_count += len(chunk)
                            if byte_count > self.max_bytes:
                                raise WebFetchError("网页正文超过读取上限")
                            chunks.append(chunk)
                        raw = b"".join(chunks)
                        encoding = response.encoding or "utf-8"
                        try:
                            decoded = raw.decode(encoding, errors="replace")
                        except LookupError:
                            decoded = raw.decode("utf-8", errors="replace")
                        return self._extract(
                            current_url,
                            decoded,
                            content_type,
                            limit,
                        )
        except WebFetchError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise WebFetchError("网页连接失败") from exc
        raise WebFetchError("网页没有返回内容")

    def _validate_url(self, value: str) -> str:
        clean = str(value or "").strip()
        if not clean or len(clean) > 2048:
            raise WebFetchError("网页地址无效")
        try:
            parsed = urlsplit(clean)
            port = parsed.port
        except ValueError as exc:
            raise WebFetchError("网页地址格式不正确") from exc
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or not parsed.netloc
        ):
            raise WebFetchError("只支持完整的 HTTP(S) 网页地址")
        if parsed.username is not None or parsed.password is not None:
            raise WebFetchError("网页地址不能包含用户名或密码")
        if port is not None and not 1 <= port <= 65535:
            raise WebFetchError("网页端口无效")
        hostname = parsed.hostname.rstrip(".").lower()
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
            or hostname.endswith(".internal")
        ):
            raise WebFetchError("不能读取本机或内网页面")
        try:
            addresses = self.resolver(
                hostname,
                port or (443 if parsed.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise WebFetchError("无法解析网页地址") from exc
        resolved = {
            str(item[4][0])
            for item in addresses
            if len(item) >= 5 and item[4]
        }
        if not resolved:
            raise WebFetchError("无法解析网页地址")
        for address in resolved:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as exc:
                raise WebFetchError("网页地址解析结果无效") from exc
            if not parsed_address.is_global:
                raise WebFetchError("不能读取本机或内网页面")
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path or "/",
                parsed.query,
                "",
            )
        )

    @staticmethod
    def _extract(
        url: str,
        decoded: str,
        content_type: str,
        max_chars: int,
    ) -> WebFetchResult:
        if "html" in content_type:
            parser = _HTMLTextExtractor()
            parser.feed(decoded)
            title, text = parser.result()
        else:
            title = ""
            text = decoded.strip()
        if not text:
            raise WebFetchError("网页没有可读取的正文")
        return WebFetchResult(
            title=title[:300] or url,
            url=url,
            text=text[:max_chars],
            content_type=content_type.split(";", 1)[0],
        )
