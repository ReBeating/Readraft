from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

from pydantic import BaseModel, ConfigDict, Field

from .agent_capabilities import (
    ANALYZE_REFERENCE,
    CREATE_CANDIDATE_DRAFT,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    PROPOSE_TEXT_PATCH,
    READ_CHAPTER,
    READ_PROJECT,
    READ_REFERENCE,
    SEARCH_PROJECT,
    SEARCH_REFERENCE,
    WEB_SEARCH,
)
from .assistant_chat_schema import (
    AssistantDraftProposal,
    AssistantRewriteProposal,
    AssistantSettingsPatch,
    AssistantStoryPlanProposal,
)
from .db import Database


MAX_TOOL_RESULT_CHARS = 48_000
MUTATING_PROPOSAL_TOOLS = frozenset(
    {
        "propose_settings_patch",
        "propose_story_plan",
        "create_chapter_draft",
        "replace_selected_text",
    }
)


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=6, ge=1, le=10)


class WebSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=6)


class ReferenceChapterArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chapter_id: Optional[str] = Field(default=None, max_length=80)
    position: Optional[int] = Field(default=None, ge=0, le=100_000)


class ReferenceAnalysisArguments(ReferenceChapterArguments):
    pass


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    label: str
    description: str
    capability: str
    scopes: frozenset[str]
    input_model: Type[BaseModel]
    read_only: bool = True

    def public_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "read_only": self.read_only,
            "parameters": self.input_model.model_json_schema(),
        }


AGENT_TOOL_SPECS: dict[str, AgentToolSpec] = {
    "read_book_settings": AgentToolSpec(
        name="read_book_settings",
        label="读取作品设定",
        description=(
            "读取本书已保存设定、人物、确认规划和章节目录；不会修改任何内容。"
        ),
        capability=READ_PROJECT,
        scopes=frozenset({"novel_project", "novel_chapter"}),
        input_model=EmptyArguments,
    ),
    "read_chapter": AgentToolSpec(
        name="read_chapter",
        label="读取当前章节",
        description=(
            "读取当前章节工作稿、前章摘录、任务卡和已确认写作约束。"
        ),
        capability=READ_CHAPTER,
        scopes=frozenset({"novel_chapter"}),
        input_model=EmptyArguments,
    ),
    "search_story_memory": AgentToolSpec(
        name="search_story_memory",
        label="检索故事记忆",
        description=(
            "在本书冻结的正史记忆、确认规划和近期章节来源中检索相关事实。"
        ),
        capability=SEARCH_PROJECT,
        scopes=frozenset({"novel_project", "novel_chapter"}),
        input_model=SearchArguments,
    ),
    "read_reference_chapter": AgentToolSpec(
        name="read_reference_chapter",
        label="读取参考章节",
        description=(
            "读取当前参考书中指定章节；结果只能用于证据化分析，不能进入正文生成。"
        ),
        capability=READ_REFERENCE,
        scopes=frozenset({"reference_document", "reference_chapter"}),
        input_model=ReferenceChapterArguments,
    ),
    "search_reference": AgentToolSpec(
        name="search_reference",
        label="检索参考书",
        description=(
            "检索参考书原文并返回带章节位置的短证据，不复制整书。"
        ),
        capability=SEARCH_REFERENCE,
        scopes=frozenset({"reference_document", "reference_chapter"}),
        input_model=SearchArguments,
    ),
    "search_web": AgentToolSpec(
        name="search_web",
        label="联网搜索",
        description=(
            "搜索当前互联网并返回标题、链接和可引用摘要。用户明确要求"
            "搜索、核实或来源，问题依赖近期变化，或者缺少必要的外部现实"
            "资料时调用；纯创作、改写、概括已有材料时不要调用。"
        ),
        capability=WEB_SEARCH,
        scopes=frozenset(
            {
                "novel_project",
                "novel_chapter",
                "reference_document",
                "reference_chapter",
            }
        ),
        input_model=WebSearchArguments,
    ),
    "read_reference_analysis": AgentToolSpec(
        name="read_reference_analysis",
        label="读取拆书分析",
        description=(
            "读取指定章节已有的结构化拆书结果，用于提炼抽象技法。"
        ),
        capability=ANALYZE_REFERENCE,
        scopes=frozenset({"reference_document", "reference_chapter"}),
        input_model=ReferenceAnalysisArguments,
    ),
    "propose_settings_patch": AgentToolSpec(
        name="propose_settings_patch",
        label="提出设定候选",
        description=(
            "创建等待作者确认的设定候选；不会直接写入作品设定。"
        ),
        capability=PROPOSE_SETTINGS_PATCH,
        scopes=frozenset({"novel_project", "novel_chapter"}),
        input_model=AssistantSettingsPatch,
        read_only=False,
    ),
    "propose_story_plan": AgentToolSpec(
        name="propose_story_plan",
        label="提出故事规划",
        description=(
            "创建完整、可版本化的全书规划候选；不会直接修改正文。"
        ),
        capability=PROPOSE_STORY_PLAN,
        scopes=frozenset({"novel_project", "novel_chapter"}),
        input_model=AssistantStoryPlanProposal,
        read_only=False,
    ),
    "create_chapter_draft": AgentToolSpec(
        name="create_chapter_draft",
        label="创建章节工作稿",
        description=(
            "创建整章或续写候选；服务端随后按当前策略提交到可撤回工作稿。"
        ),
        capability=CREATE_CANDIDATE_DRAFT,
        scopes=frozenset({"novel_chapter"}),
        input_model=AssistantDraftProposal,
        read_only=False,
    ),
    "replace_selected_text": AgentToolSpec(
        name="replace_selected_text",
        label="修订正文选区",
        description=(
            "只为作者已引用且经过版本校验的正文选区创建替换候选。"
        ),
        capability=PROPOSE_TEXT_PATCH,
        scopes=frozenset({"novel_chapter"}),
        input_model=AssistantRewriteProposal,
        read_only=False,
    ),
}


def available_agent_tools(
    context: Mapping[str, Any],
) -> list[AgentToolSpec]:
    scope = str(context.get("scope") or "")
    capabilities = {
        str(value)
        for value in (
            (context.get("agent") or {}).get("capabilities") or []
        )
    }
    return [
        spec
        for spec in AGENT_TOOL_SPECS.values()
        if spec.capability in capabilities
        and scope in spec.scopes
        and (
            spec.name != "search_web"
            or bool(context.get("web_search_available"))
        )
    ]


@dataclass
class AgentToolExecution:
    result: dict[str, Any]
    accessed_sources: list[dict[str, Any]] = field(default_factory=list)
    settings_patch: AssistantSettingsPatch | None = None
    story_plan: AssistantStoryPlanProposal | None = None
    draft: AssistantDraftProposal | None = None
    rewrite: AssistantRewriteProposal | None = None


WebSearchCallable = Callable[
    [int, str, int], Sequence[Mapping[str, Any]]
]


class AgentToolExecutor:
    def __init__(
        self,
        database: Database,
        *,
        web_search: WebSearchCallable | None = None,
    ):
        self.database = database
        self.web_search = web_search

    def execute(
        self,
        *,
        user_id: int,
        tool_name: str,
        arguments: Mapping[str, Any],
        context: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        selected_quote: str,
    ) -> AgentToolExecution:
        spec = AGENT_TOOL_SPECS.get(tool_name)
        allowed = {item.name for item in available_agent_tools(context)}
        if not spec or tool_name not in allowed:
            raise PermissionError("当前协作角色没有调用该工具的权限")
        parsed = spec.input_model.model_validate(dict(arguments))
        handler = getattr(self, f"_execute_{tool_name}")
        execution = handler(
            user_id=user_id,
            arguments=parsed,
            context=context,
            sources=sources,
            selected_quote=selected_quote,
        )
        execution.result = _bounded_mapping(execution.result)
        return execution

    def _execute_read_book_settings(
        self,
        *,
        arguments: EmptyArguments,
        context: Mapping[str, Any],
        **_: Any,
    ) -> AgentToolExecution:
        del arguments
        result = {
            key: context.get(key)
            for key in (
                "project",
                "chapter",
                "characters",
                "chapter_plan",
                "confirmed_story_blueprint",
                "confirmed_plot_arcs",
                "confirmed_story_plan",
                "confirmed_voice_profile",
                "confirmed_editing_preferences",
                "active_techniques",
                "planned_causal_links",
            )
            if context.get(key) not in (None, "", [], {})
        }
        return AgentToolExecution(result={"settings": result})

    def _execute_read_chapter(
        self,
        *,
        arguments: EmptyArguments,
        context: Mapping[str, Any],
        selected_quote: str,
        **_: Any,
    ) -> AgentToolExecution:
        del arguments
        result = {
            key: context.get(key)
            for key in (
                "chapter",
                "confirmed_task_card",
                "characters",
                "canonical_memory",
                "confirmed_story_plan",
                "planned_causal_links",
                "confirmed_voice_profile",
                "confirmed_editing_preferences",
                "active_techniques",
                "previous_chapter_excerpt",
                "current_chapter_excerpt",
                "current_version_id",
                "current_chapter_hash",
            )
            if context.get(key) not in (None, "", [], {})
        }
        if selected_quote:
            result["selected_quote"] = selected_quote
        return AgentToolExecution(result={"chapter_context": result})

    def _execute_search_story_memory(
        self,
        *,
        arguments: SearchArguments,
        context: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        **_: Any,
    ) -> AgentToolExecution:
        searchable = {
            "canonical_memory": context.get("canonical_memory"),
            "confirmed_story_plan": (
                context.get("confirmed_story_plan")
                or context.get("confirmed_story_blueprint")
            ),
            "confirmed_plot_arcs": context.get("confirmed_plot_arcs"),
            "recent_memory": context.get("canonical_recent_memory"),
            "planned_causal_links": context.get("planned_causal_links"),
        }
        matches = _search_structured(
            searchable,
            query=arguments.query,
            max_results=arguments.max_results,
        )
        accessed_sources: list[dict[str, Any]] = []
        if len(matches) < arguments.max_results:
            source_matches, accessed_sources = _search_source_items(
                sources,
                query=arguments.query,
                max_results=arguments.max_results - len(matches),
                allowed_kinds={"novel_version"},
            )
            matches.extend(source_matches)
        return AgentToolExecution(
            result={
                "query": arguments.query,
                "matches": matches,
                "matched_count": len(matches),
            },
            accessed_sources=accessed_sources,
        )

    def _execute_read_reference_chapter(
        self,
        *,
        user_id: int,
        arguments: ReferenceChapterArguments,
        context: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        **_: Any,
    ) -> AgentToolExecution:
        chapter_id = str(arguments.chapter_id or "")
        if not chapter_id and arguments.position is None:
            existing = next(
                (
                    dict(item)
                    for item in sources
                    if str(item.get("kind")) == "reference_chapter"
                ),
                None,
            )
            if existing:
                return AgentToolExecution(
                    result={
                        "chapter": {
                            "source_id": existing.get("source_id"),
                            "label": existing.get("label"),
                            "text": existing.get("text"),
                        }
                    },
                    accessed_sources=[existing],
                )
        row = self._reference_chapter_row(
            user_id=user_id,
            context=context,
            chapter_id=chapter_id or None,
            position=arguments.position,
        )
        text = _read_text(Path(str(row["content_path"])))
        source = _reference_source(row, text)
        return AgentToolExecution(
            result={
                "chapter": {
                    "source_id": source["source_id"],
                    "label": source["label"],
                    "text": source["text"],
                }
            },
            accessed_sources=[source],
        )

    def _execute_search_reference(
        self,
        *,
        user_id: int,
        arguments: WebSearchArguments,
        context: Mapping[str, Any],
        **_: Any,
    ) -> AgentToolExecution:
        document_id = str((context.get("document") or {}).get("id") or "")
        if not document_id:
            raise ValueError("参考书上下文缺少文档标识")
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.document_id, c.position, c.title,
                       c.content_path, d.title AS document_title
                FROM chapters c
                JOIN documents d ON d.id=c.document_id
                WHERE c.document_id=? AND d.user_id=?
                ORDER BY c.position
                LIMIT 500
                """,
                (document_id, user_id),
            ).fetchall()
        terms = _query_terms(arguments.query)
        matches: list[dict[str, Any]] = []
        accessed_sources: list[dict[str, Any]] = []
        for row in rows:
            text = _read_text(Path(str(row["content_path"])))
            position = _best_match_position(text, terms)
            if position < 0:
                continue
            source = _reference_source(
                row, text, selected_position=position
            )
            start = max(0, position - 180)
            end = min(len(text), position + 420)
            quote = text[start:end].strip()
            matches.append(
                {
                    "source_id": source["source_id"],
                    "label": source["label"],
                    "quote": quote,
                    "start_offset": start,
                    "end_offset": end,
                }
            )
            accessed_sources.append(source)
            if len(matches) >= arguments.max_results:
                break
        return AgentToolExecution(
            result={
                "query": arguments.query,
                "matches": matches,
                "matched_count": len(matches),
            },
            accessed_sources=accessed_sources,
        )

    def _execute_search_web(
        self,
        *,
        user_id: int,
        arguments: SearchArguments,
        **_: Any,
    ) -> AgentToolExecution:
        if self.web_search is None:
            raise ValueError("联网搜索尚未配置")
        raw_results = self.web_search(
            user_id,
            arguments.query,
            arguments.max_results,
        )
        matches: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        for raw in raw_results[: arguments.max_results]:
            title = str(raw.get("title") or "").strip()
            url = str(raw.get("url") or "").strip()
            snippet = str(raw.get("snippet") or "").strip()
            if not title or not url or not snippet:
                continue
            source_id = (
                "web:"
                + hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
            )
            source = {
                "source_id": source_id,
                "kind": "web",
                "label": title,
                "text": snippet,
                "base_offset": 0,
                "url": url,
            }
            sources.append(source)
            matches.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
        return AgentToolExecution(
            result={
                "query": arguments.query,
                "matches": matches,
                "matched_count": len(matches),
                "source_notice": (
                    "搜索摘要来自外部网页，必须交叉核对并引用来源。"
                ),
            },
            accessed_sources=sources,
        )

    def _execute_read_reference_analysis(
        self,
        *,
        user_id: int,
        arguments: ReferenceAnalysisArguments,
        context: Mapping[str, Any],
        **_: Any,
    ) -> AgentToolExecution:
        row = self._reference_chapter_row(
            user_id=user_id,
            context=context,
            chapter_id=arguments.chapter_id,
            position=arguments.position,
        )
        with self.database.connection() as connection:
            analysis = connection.execute(
                """
                SELECT a.id, a.result_json
                FROM chapter_analyses a
                JOIN analysis_jobs j ON j.id=a.job_id
                WHERE a.chapter_id=? AND a.status='completed'
                  AND j.user_id=?
                ORDER BY a.finished_at DESC, a.rowid DESC
                LIMIT 1
                """,
                (row["id"], user_id),
            ).fetchone()
        if not analysis:
            raise ValueError("该参考章节还没有完成的拆书分析")
        try:
            result = json.loads(str(analysis["result_json"] or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("拆书分析数据损坏") from exc
        source = {
            "source_id": f"analysis:{analysis['id']}",
            "kind": "reference_analysis",
            "label": f"拆书分析 · 第 {row['position']} 章《{row['title']}》",
            "text": json.dumps(result, ensure_ascii=False, indent=2)[
                :MAX_TOOL_RESULT_CHARS
            ],
            "base_offset": 0,
            "url": f"/analyses/{analysis['id']}",
            "document_id": str(row["document_id"]),
            "reference_chapter_id": str(row["id"]),
        }
        return AgentToolExecution(
            result={
                "analysis": result,
                "source_id": source["source_id"],
            },
            accessed_sources=[source],
        )

    def _execute_propose_settings_patch(
        self,
        *,
        arguments: AssistantSettingsPatch,
        **_: Any,
    ) -> AgentToolExecution:
        return AgentToolExecution(
            result={
                "accepted": True,
                "kind": "settings_patch",
                "fields": sorted(
                    arguments.model_dump(exclude_none=True).keys()
                ),
                "requires_author_confirmation": True,
            },
            settings_patch=arguments,
        )

    def _execute_propose_story_plan(
        self,
        *,
        arguments: AssistantStoryPlanProposal,
        **_: Any,
    ) -> AgentToolExecution:
        return AgentToolExecution(
            result={
                "accepted": True,
                "kind": "story_plan",
                "major_turn_count": len(
                    arguments.blueprint.major_turns
                ),
                "payoff_count": len(
                    arguments.blueprint.must_payoffs
                ),
                "requires_author_confirmation": True,
            },
            story_plan=arguments,
        )

    def _execute_create_chapter_draft(
        self,
        *,
        arguments: AssistantDraftProposal,
        **_: Any,
    ) -> AgentToolExecution:
        return AgentToolExecution(
            result={
                "accepted": True,
                "kind": "chapter_draft",
                "mode": arguments.mode,
                "character_count": len(arguments.content),
                "target": "revertible_working_copy",
            },
            draft=arguments,
        )

    def _execute_replace_selected_text(
        self,
        *,
        arguments: AssistantRewriteProposal,
        selected_quote: str,
        **_: Any,
    ) -> AgentToolExecution:
        if not selected_quote:
            raise ValueError("修订正文前必须先引用经过校验的正文选区")
        return AgentToolExecution(
            result={
                "accepted": True,
                "kind": "selected_text_rewrite",
                "selected_character_count": len(selected_quote),
                "replacement_character_count": len(
                    arguments.replacement_text
                ),
                "target": "revertible_working_copy",
            },
            rewrite=arguments,
        )

    def _reference_chapter_row(
        self,
        *,
        user_id: int,
        context: Mapping[str, Any],
        chapter_id: str | None,
        position: int | None,
    ) -> Mapping[str, Any]:
        document_id = str((context.get("document") or {}).get("id") or "")
        if not document_id:
            raise ValueError("参考书上下文缺少文档标识")
        if not chapter_id and position is None:
            chapter_items = list(context.get("chapters_and_analysis") or [])
            if len(chapter_items) == 1:
                chapter_id = str(chapter_items[0].get("id") or "")
            else:
                raise ValueError("请指定需要读取的参考章节")
        with self.database.connection() as connection:
            if chapter_id:
                row = connection.execute(
                    """
                    SELECT c.id, c.document_id, c.position, c.title,
                           c.content_path, d.title AS document_title
                    FROM chapters c
                    JOIN documents d ON d.id=c.document_id
                    WHERE c.id=? AND c.document_id=? AND d.user_id=?
                    """,
                    (chapter_id, document_id, user_id),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT c.id, c.document_id, c.position, c.title,
                           c.content_path, d.title AS document_title
                    FROM chapters c
                    JOIN documents d ON d.id=c.document_id
                    WHERE c.position=? AND c.document_id=? AND d.user_id=?
                    """,
                    (position, document_id, user_id),
                ).fetchone()
        if not row:
            raise ValueError("参考章节不存在或不属于当前用户")
        return row


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("无法读取工具所需的文本来源") from exc


def _reference_source(
    row: Mapping[str, Any],
    text: str,
    *,
    selected_position: int | None = None,
) -> dict[str, Any]:
    base_offset = 0
    if (
        selected_position is not None
        and len(text) > MAX_TOOL_RESULT_CHARS
    ):
        base_offset = max(
            0, selected_position - MAX_TOOL_RESULT_CHARS // 2
        )
    bounded = text[
        base_offset : base_offset + MAX_TOOL_RESULT_CHARS
    ]
    return {
        "source_id": f"reference-chapter:{row['id']}",
        "kind": "reference_chapter",
        "label": f"参考书第 {row['position']} 章《{row['title']}》",
        "text": bounded,
        "base_offset": base_offset,
        "url": (
            f"/documents/{row['document_id']}/chapters/"
            f"{row['id']}/source"
        ),
        "document_id": str(row["document_id"]),
        "reference_chapter_id": str(row["id"]),
    }


def _bounded_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    encoded = json.dumps(result, ensure_ascii=False, default=str)
    if len(encoded) <= MAX_TOOL_RESULT_CHARS:
        return result
    return {
        "truncated": True,
        "preview": encoded[:MAX_TOOL_RESULT_CHARS],
    }


def _query_terms(query: str) -> list[str]:
    chunks = [
        item
        for item in re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", query)
        if item not in {"什么", "怎么", "这个", "那个", "可以", "需要", "希望"}
    ]
    terms: list[str] = []
    for chunk in chunks:
        if chunk not in terms:
            terms.append(chunk)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk) and len(chunk) > 4:
            for index in range(0, len(chunk) - 1):
                pair = chunk[index : index + 2]
                if pair not in terms:
                    terms.append(pair)
    return sorted(terms, key=len, reverse=True)[:32] or [query[:80]]


def _best_match_position(text: str, terms: Sequence[str]) -> int:
    lowered = text.casefold()
    positions = [
        lowered.find(term.casefold())
        for term in terms
        if term and lowered.find(term.casefold()) >= 0
    ]
    return min(positions) if positions else -1


def _search_structured(
    value: Any,
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    leaves: list[tuple[str, str]] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
        elif item not in (None, ""):
            leaves.append((path, str(item)))

    walk(value, "")
    scored: list[tuple[int, str, str]] = []
    for path, text in leaves:
        lowered = text.casefold()
        score = sum(
            max(1, len(term))
            for term in terms
            if term.casefold() in lowered
        )
        if score:
            scored.append((score, path, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"path": path, "text": text[:1200]}
        for _score, path, text in scored[:max_results]
    ]


def _search_source_items(
    sources: Sequence[Mapping[str, Any]],
    *,
    query: str,
    max_results: int,
    allowed_kinds: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    terms = _query_terms(query)
    matches: list[dict[str, Any]] = []
    accessed: list[dict[str, Any]] = []
    for raw_source in sources:
        if str(raw_source.get("kind") or "") not in allowed_kinds:
            continue
        source = dict(raw_source)
        text = str(source.get("text") or "")
        position = _best_match_position(text, terms)
        if position < 0:
            continue
        start = max(0, position - 160)
        end = min(len(text), position + 360)
        matches.append(
            {
                "source_id": source.get("source_id"),
                "label": source.get("label"),
                "quote": text[start:end].strip(),
            }
        )
        accessed.append(source)
        if len(matches) >= max_results:
            break
    return matches, accessed
