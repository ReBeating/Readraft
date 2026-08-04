from app.agent_capabilities import WRITE_CHAPTER
from app.assistant_chat_schema import (
    AssistantChatResponse,
    AssistantChatResult,
    AssistantCitationProposal,
    AssistantDraftProposal,
)
from app.assistant_result import decode_message, normalize_assistant_response


def _response(*, with_draft: bool = False) -> AssistantChatResponse:
    return AssistantChatResponse(
        result=AssistantChatResult(
            answer="已经按要求完成。",
            citations=[
                AssistantCitationProposal(
                    source_id="chapter:1",
                    quote="旧收音机",
                    note="来自当前正文",
                )
            ],
            draft=(
                AssistantDraftProposal(
                    content="这是完整的新正文。",
                    rationale="保留线索并推进冲突。",
                )
                if with_draft
                else None
            ),
        ),
        raw_response="{}",
        input_tokens=10,
        output_tokens=20,
        provider="test-provider",
        model="test-model",
        agent_trace=[{"tool": "compose", "status": "completed"}],
    )


def test_normalize_response_exposes_one_full_draft_contract():
    normalized = normalize_assistant_response(
        response=_response(with_draft=True),
        sources=[
            {
                "source_id": "chapter:1",
                "kind": "novel_version",
                "label": "第 1 章",
                "text": "开头写着旧收音机。",
                "base_offset": 100,
                "url": "/source",
            }
        ],
        context={
            "scope": "novel_chapter",
            "agent": {
                "role": "writer",
                "capabilities": [WRITE_CHAPTER],
            },
            "assistant_boundaries": {"auto_advance_main_head": True},
        },
    )

    assert normalized["draft"] == {
        "content": "这是完整的新正文。",
        "rationale": "保留线索并推进冲突。",
    }
    assert normalized["draft_status"] == "candidate"
    assert normalized["auto_commit"]["status"] == "pending"
    assert normalized["citations"][0]["start_offset"] == 104
    assert normalized["citations"][0]["url"] == "/source?start=104&end=108"


def test_normalize_response_drops_draft_without_writer_capability():
    normalized = normalize_assistant_response(
        response=_response(with_draft=True),
        sources=[],
        context={
            "scope": "novel_chapter",
            "agent": {"role": "advisor", "capabilities": []},
        },
    )

    assert normalized["draft"] is None
    assert normalized["auto_commit"] == {}


def test_decode_message_restores_response_and_quote():
    decoded = decode_message(
        {
            "response_json": '{"agent":{"role":"writer"}}',
            "source_type": "novel_version",
            "quote_text": "一段选区",
            "quote_project_id": "project-1",
            "quote_start_offset": 3,
            "quote_end_offset": 7,
        }
    )

    assert decoded["response"]["agent"]["label"] == "创作"
    assert decoded["quote"]["project_id"] == "project-1"
    assert decoded["quote"]["quote_text"] == "一段选区"
