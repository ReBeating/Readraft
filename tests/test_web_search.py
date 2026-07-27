import json

import httpx
import pytest

from app.agent_capabilities import agent_manifest
from app.agent_tools import AgentToolExecutor, available_agent_tools
from app.db import Database
from app.web_search import ExaWebSearch, WebSearchError


def _sse(payload):
    return "event: message\ndata: " + json.dumps(payload) + "\n\n"


def test_exa_web_search_uses_unauthenticated_hosted_mcp():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.setdefault("methods", []).append(payload["method"])
        if payload["method"] == "initialize":
            assert "x-api-key" not in request.headers
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "mcp-session-id": "exa-test-session",
                },
                text=_sse(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": "exa-search-server",
                                "version": "test",
                            },
                        },
                    }
                ),
            )
        assert request.headers["mcp-session-id"] == "exa-test-session"
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        arguments = payload["params"]["arguments"]
        seen["query"] = arguments["query"]
        seen["count"] = arguments["numResults"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Title: 官方资料\n"
                                    "URL: https://example.com/official\n"
                                    "Published: 2026-07-01\n"
                                    "Author: 测试作者\n"
                                    "Highlights:\n"
                                    "- 第一段摘要。\n"
                                    "- 第二段摘要。\n\n"
                                    "---\n\n"
                                    "Title: 重复链接\n"
                                    "URL: https://example.com/official\n"
                                    "Published: N/A\n"
                                    "Author: N/A\n"
                                    "Highlights:\n"
                                    "- 不应重复。\n\n"
                                    "---\n\n"
                                    "Title: 无效协议\n"
                                    "URL: ftp://example.com/file\n"
                                    "Highlights:\n"
                                    "- 不应返回。"
                                ),
                            }
                        ]
                    },
                }
            ),
        )

    search = ExaWebSearch(transport=httpx.MockTransport(handler))
    results = search.search("2026 年资料", max_results=4)
    assert seen["methods"] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert seen["query"] == "2026 年资料"
    assert seen["count"] == 4
    assert len(results) == 1
    assert results[0].title == "官方资料"
    assert results[0].url == "https://example.com/official"
    assert results[0].snippet == "- 第一段摘要。\n- 第二段摘要。"


def test_exa_web_search_reports_free_tier_rate_limit():
    search = ExaWebSearch(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(429)
        ),
    )
    with pytest.raises(WebSearchError, match="免费联网搜索暂时达到限额"):
        search.search("测试")


def test_agent_web_search_tool_is_opt_in_and_returns_citable_sources(
    tmp_path,
):
    database = Database(tmp_path / "test.db")
    database.initialize()
    calls = []
    executor = AgentToolExecutor(
        database,
        web_search=lambda user_id, query, limit: calls.append(
            (user_id, query, limit)
        )
        or [
            {
                "title": "资料页",
                "url": "https://example.com/source",
                "snippet": "可引用的搜索摘要。",
            }
        ],
    )
    context = {
        "scope": "novel_project",
        "agent": agent_manifest("advisor"),
        "web_search_available": False,
    }
    assert "search_web" not in {
        item.name for item in available_agent_tools(context)
    }

    context["web_search_available"] = True
    assert "search_web" in {
        item.name for item in available_agent_tools(context)
    }
    execution = executor.execute(
        user_id=7,
        tool_name="search_web",
        arguments={"query": "海港停电史", "max_results": 3},
        context=context,
        sources=[],
        selected_quote="",
    )
    assert calls == [(7, "海港停电史", 3)]
    assert execution.result["matched_count"] == 1
    assert execution.result["matches"][0]["source_id"].startswith("web:")
    assert execution.accessed_sources == [
        {
            "source_id": execution.result["matches"][0]["source_id"],
            "kind": "web",
            "label": "资料页",
            "text": "可引用的搜索摘要。",
            "base_offset": 0,
            "url": "https://example.com/source",
        }
    ]
