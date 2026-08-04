from __future__ import annotations

from typing import Any, Mapping, Sequence

from .agent_capabilities import (
    CREATE_TECHNIQUE_CARD,
    WRITE_CHAPTER,
    MANAGE_CHAPTERS,
    MANAGE_NOTES,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    agent_manifest,
)
from .assistant_chat_schema import AssistantChatResponse
from .db import utc_now
from .json_support import load_json
from .structured_settings import preview_structured_edits


def compact_analysis(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    return {
        key: result.get(key)
        for key in (
            "chapter_title",
            "summary",
            "characters",
            "scenes",
            "key_events",
            "foreshadowing",
            "conflicts",
            "ending_hook",
            "techniques",
        )
        if result.get(key) not in (None, "", [])
    }


def normalize_assistant_response(
    *,
    response: AssistantChatResponse,
    sources: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one native Agent result into the persisted UI contract."""
    source_map = {
        str(source.get("source_id") or ""): source
        for source in sources
        if source.get("source_id")
    }
    citations = []
    for proposal in response.result.citations:
        source = source_map.get(proposal.source_id)
        if not source:
            continue
        source_text = str(source.get("text") or "")
        local_start = source_text.find(proposal.quote)
        if local_start < 0:
            continue
        absolute_start = int(source.get("base_offset") or 0) + local_start
        absolute_end = absolute_start + len(proposal.quote)
        url = str(source.get("url") or "")
        external = str(source.get("kind") or "") == "web"
        separator = "&" if "?" in url else "?"
        if url and not external:
            url = f"{url}{separator}start={absolute_start}&end={absolute_end}"
        citations.append(
            {
                "source_id": proposal.source_id,
                "label": str(source.get("label") or "来源"),
                "quote": proposal.quote,
                "note": proposal.note,
                "url": url,
                "external": external,
                "start_offset": absolute_start,
                "end_offset": absolute_end,
            }
        )

    capabilities = {
        str(item) for item in ((context.get("agent") or {}).get("capabilities") or [])
    }
    settings_patch = None
    settings_patch_preview = None
    if (
        PROPOSE_SETTINGS_PATCH in capabilities
        and str(context.get("scope") or "") in {"novel_project", "novel_chapter"}
        and response.result.settings_patch is not None
    ):
        settings_patch = response.result.settings_patch.model_dump(
            mode="json", exclude_none=True
        )
        structured_edits = list(response.result.settings_patch.structured_edits or [])
        if structured_edits:
            try:
                settings_patch_preview = {
                    "items": preview_structured_edits(
                        structured_edits,
                        dict(context.get("structured_settings") or {}),
                    )
                }
            except ValueError as exc:
                settings_patch_preview = {"items": [], "error": str(exc)}

    story_plan = None
    if (
        PROPOSE_STORY_PLAN in capabilities
        and str(context.get("scope") or "") in {"novel_project", "novel_chapter"}
        and response.result.story_plan is not None
    ):
        story_plan = response.result.story_plan.model_dump(mode="json")

    chapter_patch = None
    if (
        MANAGE_CHAPTERS in capabilities
        and str(context.get("scope") or "")
        in {"novel_project", "novel_chapter"}
        and response.result.chapter_patch is not None
    ):
        chapter_patch = response.result.chapter_patch.model_dump(mode="json")

    note_patch = None
    if (
        MANAGE_NOTES in capabilities
        and str(context.get("scope") or "")
        in {"novel_project", "novel_chapter"}
        and response.result.note_patch is not None
    ):
        note_patch = response.result.note_patch.model_dump(mode="json")

    version_restore = None
    if (
        WRITE_CHAPTER in capabilities
        and str(context.get("scope") or "")
        in {"novel_project", "novel_chapter"}
        and response.result.version_restore is not None
    ):
        version_restore = response.result.version_restore.model_dump(
            mode="json"
        )

    technique_patch = None
    if (
        CREATE_TECHNIQUE_CARD in capabilities
        and str(context.get("scope") or "")
        in {"document", "reference_chapter", "novel_project", "novel_chapter"}
        and response.result.technique_patch is not None
    ):
        technique_patch = response.result.technique_patch.model_dump(
            mode="json"
        )

    chapter_workflow = None
    if (
        WRITE_CHAPTER in capabilities
        and str(context.get("scope") or "")
        in {"novel_project", "novel_chapter"}
        and response.result.chapter_workflow is not None
    ):
        chapter_workflow = response.result.chapter_workflow.model_dump(
            mode="json"
        )

    draft = None
    if (
        WRITE_CHAPTER in capabilities
        and str(context.get("scope") or "") == "novel_chapter"
        and response.result.draft is not None
    ):
        draft = response.result.draft.model_dump(mode="json")

    agent = dict(context.get("agent") or {})
    boundaries = context.get("assistant_boundaries") or {}
    auto_apply_settings = bool(boundaries.get("auto_apply_settings"))
    auto_apply_story_plan = bool(boundaries.get("auto_apply_story_plan"))
    auto_apply_chapter_metadata = bool(
        boundaries.get("auto_apply_chapter_metadata")
    )
    auto_apply_notes = bool(boundaries.get("auto_apply_notes"))
    auto_apply_techniques = bool(boundaries.get("auto_apply_techniques"))
    auto_commit: dict[str, Any] = {}
    if bool(boundaries.get("auto_advance_main_head")) and draft:
        auto_commit = {
            "status": "pending",
            "kind": "draft",
            "version_id": None,
            "reverted_version_id": None,
            "error": None,
            "updated_at": utc_now(),
        }
    return {
        "answer": response.result.answer,
        "citations": citations,
        "draft": draft,
        "draft_status": "candidate" if draft else None,
        "settings_patch": settings_patch,
        "settings_patch_preview": settings_patch_preview,
        "settings_patch_status": (
            ("pending" if auto_apply_settings else "candidate")
            if settings_patch
            else None
        ),
        "story_plan": story_plan,
        "story_plan_status": (
            ("pending" if auto_apply_story_plan else "candidate")
            if story_plan
            else None
        ),
        "chapter_patch": chapter_patch,
        "chapter_patch_status": (
            ("pending" if auto_apply_chapter_metadata else "candidate")
            if chapter_patch
            else None
        ),
        "note_patch": note_patch,
        "note_patch_status": (
            ("pending" if auto_apply_notes else "candidate")
            if note_patch
            else None
        ),
        "version_restore": version_restore,
        "version_restore_status": (
            "pending"
            if version_restore
            and bool(boundaries.get("auto_advance_main_head"))
            else ("candidate" if version_restore else None)
        ),
        "technique_patch": technique_patch,
        "technique_patch_status": (
            ("pending" if auto_apply_techniques else "candidate")
            if technique_patch
            else None
        ),
        "chapter_workflow": chapter_workflow,
        "chapter_workflow_status": (
            chapter_workflow.get("status") if chapter_workflow else None
        ),
        "auto_commit": auto_commit,
        "provider": response.provider,
        "model": response.model,
        "agent": {
            "role": str(agent.get("role") or "advisor"),
            "label": str(agent.get("label") or "讨论"),
            "capabilities": sorted(capabilities),
        },
        "agent_run": {
            "dispatch": dict(context.get("dispatch") or {}),
            "tool_calls": list(response.agent_trace or []),
        },
        "boundary": {
            "canon_unchanged": not bool(
                chapter_workflow
                and int(chapter_workflow.get("completed_count") or 0)
            ),
            "story_memory_unchanged": not bool(
                chapter_workflow
                and int(chapter_workflow.get("completed_count") or 0)
            ),
            "task_card_unchanged": True,
            "project_settings_unchanged": True,
            "story_plan_unchanged": True,
            "chapter_metadata_unchanged": True,
            "author_notes_unchanged": True,
            "main_head_unchanged": True,
            "technique_library_unchanged": True,
            "chapter_workflow_unchanged": not bool(chapter_workflow),
        },
    }


def decode_message(message: dict[str, Any]) -> dict[str, Any]:
    message["response"] = load_json(message.get("response_json"), {})
    agent = message["response"].get("agent")
    if isinstance(agent, dict):
        role = str(agent.get("role") or "")
        try:
            agent["label"] = agent_manifest(role)["label"]
        except ValueError:
            pass
    if message.get("quote_text"):
        message["quote"] = {
            "source_type": message.get("source_type"),
            "project_id": message.get("quote_project_id"),
            "document_id": message.get("quote_document_id"),
            "novel_chapter_id": message.get("quote_novel_chapter_id"),
            "version_id": message.get("quote_version_id"),
            "reference_chapter_id": message.get("quote_reference_chapter_id"),
            "start_offset": message.get("quote_start_offset"),
            "end_offset": message.get("quote_end_offset"),
            "quote_text": message.get("quote_text"),
            "content_hash": message.get("quote_content_hash"),
            "source_label": message.get("quote_source_label"),
        }
    else:
        message["quote"] = None
    return message


def decode_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_call["arguments"] = load_json(tool_call.get("arguments_json"), {})
    tool_call["result"] = load_json(tool_call.get("result_json"), {})
    tool_call["read_only"] = bool(tool_call.get("read_only"))
    return tool_call


def decode_agent_step(step: dict[str, Any]) -> dict[str, Any]:
    step["available_tools"] = load_json(step.get("available_tools_json"), [])
    step["decision"] = load_json(step.get("decision_json"), {})
    return step
