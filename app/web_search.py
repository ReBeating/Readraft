from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


class ExaWebSearch:
    """Use Exa's hosted MCP search service.

    The hosted endpoint has a free unauthenticated tier. Deployments that
    outgrow it may set EXA_API_KEY without changing the user-facing workflow.
    """

    endpoint = "https://mcp.exa.ai/mcp"
    protocol_version = "2025-03-26"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 25,
    ):
        self.api_key = str(api_key or "").strip()
        self.transport = transport
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        clean_query = str(query or "").strip()
        if not clean_query:
            raise WebSearchError("联网搜索关键词不能为空")
        if len(clean_query) > 500:
            raise WebSearchError("联网搜索关键词不能超过 500 个字符")
        result_limit = min(6, max(1, int(max_results)))
        base_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
            "x-exa-source": "readraft",
        }
        if self.api_key:
            base_headers["x-api-key"] = self.api_key

        try:
            with httpx.Client(
                transport=self.transport,
                timeout=self.timeout_seconds,
                follow_redirects=False,
            ) as client:
                initialized = client.post(
                    self.endpoint,
                    headers=base_headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": self.protocol_version,
                            "capabilities": {},
                            "clientInfo": {
                                "name": "readraft",
                                "version": "0.1",
                            },
                        },
                    },
                )
                _raise_for_exa_response(initialized)
                initialize_payload = _parse_mcp_response(initialized)
                _raise_for_mcp_error(initialize_payload)
                session_id = str(
                    initialized.headers.get("mcp-session-id") or ""
                ).strip()
                if not session_id:
                    raise WebSearchError("Exa 联网搜索未建立有效会话")

                session_headers = {
                    **base_headers,
                    "Mcp-Session-Id": session_id,
                }
                acknowledged = client.post(
                    self.endpoint,
                    headers=session_headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    },
                )
                _raise_for_exa_response(acknowledged)

                response = client.post(
                    self.endpoint,
                    headers=session_headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "web_search_exa",
                            "arguments": {
                                "query": clean_query,
                                "numResults": result_limit,
                            },
                        },
                    },
                )
        except WebSearchError:
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise WebSearchError("Exa 联网搜索连接失败") from exc

        _raise_for_exa_response(response)
        payload = _parse_mcp_response(response)
        _raise_for_mcp_error(payload)
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise WebSearchError("Exa 联网搜索返回结构不正确")
        content = result.get("content")
        if not isinstance(content, list):
            raise WebSearchError("Exa 联网搜索返回结构不正确")
        text = "\n".join(
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") == "text"
            and str(item.get("text") or "").strip()
        )
        if result.get("isError"):
            _raise_exa_message(text)
        return _parse_exa_results(text, limit=result_limit)


def _raise_for_exa_response(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise WebSearchError(
            "免费联网搜索暂时达到限额，请稍后再试"
        )
    if response.status_code in {401, 403}:
        raise WebSearchError("Exa 联网搜索认证失败")
    if response.status_code >= 400:
        raise WebSearchError(
            f"Exa 联网搜索请求失败（HTTP {response.status_code}）"
        )


def _parse_mcp_response(response: httpx.Response) -> Mapping[str, Any]:
    if response.status_code == 202 and not response.content:
        return {}
    content_type = str(response.headers.get("content-type") or "").lower()
    try:
        if "text/event-stream" not in content_type:
            payload = response.json()
            if isinstance(payload, Mapping):
                return payload
            raise ValueError("not an object")

        data_lines: list[str] = []
        payloads: list[Mapping[str, Any]] = []
        for line in response.text.splitlines():
            if not line:
                if data_lines:
                    parsed = json.loads("\n".join(data_lines))
                    if isinstance(parsed, Mapping):
                        payloads.append(parsed)
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            parsed = json.loads("\n".join(data_lines))
            if isinstance(parsed, Mapping):
                payloads.append(parsed)
        if payloads:
            return payloads[-1]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WebSearchError(
            "Exa 联网搜索返回了无法解析的响应"
        ) from exc
    raise WebSearchError("Exa 联网搜索没有返回结果")


def _raise_for_mcp_error(payload: Mapping[str, Any]) -> None:
    error = payload.get("error")
    if not error:
        return
    if isinstance(error, Mapping):
        message = str(error.get("message") or "")
    else:
        message = str(error)
    _raise_exa_message(message)


def _raise_exa_message(message: str) -> None:
    clean = str(message or "").strip()
    lowered = clean.lower()
    if "429" in lowered or "rate limit" in lowered:
        raise WebSearchError(
            "免费联网搜索暂时达到限额，请稍后再试"
        )
    if "401" in lowered or "unauthorized" in lowered:
        raise WebSearchError("Exa 联网搜索认证失败")
    raise WebSearchError(
        "Exa 联网搜索执行失败"
        + (f"：{clean[:300]}" if clean else "")
    )


def _parse_exa_results(
    text: str,
    *,
    limit: int,
) -> list[WebSearchResult]:
    blocks = re.split(r"\n\s*---\s*\n", str(text or "").strip())
    results: list[WebSearchResult] = []
    seen_urls: set[str] = set()
    for block in blocks:
        title_match = re.search(r"(?m)^Title:\s*(.+?)\s*$", block)
        url_match = re.search(r"(?m)^URL:\s*(\S+)\s*$", block)
        if not title_match or not url_match:
            continue
        title = title_match.group(1).strip()
        url = url_match.group(1).strip()
        parsed_url = urlparse(url)
        if (
            not title
            or parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or url in seen_urls
        ):
            continue
        highlight_match = re.search(
            r"(?ms)^Highlights:\s*\n?(.*)$",
            block,
        )
        snippet = (
            highlight_match.group(1).strip()
            if highlight_match
            else block[url_match.end() :].strip()
        )
        snippet = re.sub(
            r"(?m)^(?:Published|Author):\s*.*$\n?",
            "",
            snippet,
        ).strip()
        if not snippet:
            continue
        seen_urls.add(url)
        results.append(
            WebSearchResult(
                title=title[:300],
                url=url[:2048],
                snippet=snippet[:4000],
            )
        )
        if len(results) >= limit:
            break
    if not results:
        raise WebSearchError("Exa 联网搜索没有返回可用结果")
    return results
