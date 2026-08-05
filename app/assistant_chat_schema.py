from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .story_planning_schema import StoryBlueprint
from .technique_schema import TechniqueObservation


class AssistantCitationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=160)
    quote: str = Field(min_length=1, max_length=800)
    note: str = Field(default="", max_length=500)


class AssistantDraftProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1)
    rationale: str = Field(min_length=2, max_length=800)


class ChapterDraftAuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal[
        "instruction",
        "continuity",
        "state",
        "time",
        "causality",
        "point_of_view",
        "repetition",
        "exposition",
        "scene_change",
        "character_motivation",
        "dialogue",
        "specificity",
        "rhythm",
        "style",
    ] = "instruction"
    description: str = Field(min_length=2, max_length=800)
    evidence: str = Field(
        default="候选正文未满足对应硬约束",
        min_length=1,
        max_length=500,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_compact_issue(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {
                "description": value,
                "evidence": value,
            }
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        description = str(
            normalized.get("description")
            or normalized.get("issue")
            or normalized.get("summary")
            or ""
        ).strip()
        if description and not str(
            normalized.get("evidence") or ""
        ).strip():
            normalized["evidence"] = description
        normalized["description"] = description
        raw_category = str(
            normalized.get("category")
            or normalized.get("type")
            or ""
        ).strip()
        allowed_categories = {
            "instruction",
            "continuity",
            "state",
            "time",
            "causality",
            "point_of_view",
            "repetition",
            "exposition",
            "scene_change",
            "character_motivation",
            "dialogue",
            "specificity",
            "rhythm",
            "style",
        }
        return {
            "category": (
                raw_category
                if raw_category in allowed_categories
                else "instruction"
            ),
            "description": normalized.get("description"),
            "evidence": normalized.get("evidence"),
        }


class ChapterDraftAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    verdict: Literal["pass", "revised"]
    issues: list[ChapterDraftAuditIssue] = Field(default_factory=list)
    revised_content: Optional[str] = Field(
        default=None,
        min_length=1,
    )
    summary: str = Field(min_length=2, max_length=1200)

    @model_validator(mode="after")
    def validate_revision(self) -> "ChapterDraftAuditResult":
        if self.verdict == "revised":
            if not self.issues:
                raise ValueError("修订正文时必须列出至少一个明确问题")
            if not self.revised_content:
                raise ValueError("修订正文时必须返回完整 revised_content")
        elif self.revised_content is not None:
            raise ValueError("通过审校时 revised_content 必须为空")
        return self


class AssistantArchiveRuleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal[
        "core", "world", "character", "structure", "style"
    ] = Field(
        description=(
            "资料归属：作品概览、世界、人物、剧情与结构、叙事与文风之一"
        )
    )
    title: str = Field(default="", max_length=120)
    content: str = Field(min_length=2)


SettingFieldValue = str | int | list[str]


class AssistantStructuredSettingEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    entity_type: Literal[
        "world_entry",
        "character",
        "relationship",
        "story_blueprint",
        "plot_arc",
        "voice_profile",
        "archive_rule",
    ] = Field(
        description=(
            "要编辑的结构化资料类型：世界资料卡、人物卡、人物关系、"
            "全书蓝图、剧情线、叙事与文风或已确认规则"
        )
    )
    action: Literal["create", "update", "delete"] = Field(
        description="新增、局部修改或删除（剧情线的 delete 会安全停用）"
    )
    target_id: Optional[str] = Field(
        default=None,
        max_length=160,
        description=(
            "修改或删除时优先填写 read_book_settings 返回的对象 id"
        ),
    )
    target_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "供作者识别并在缺少 id 时精确匹配的名称；不要使用模糊描述"
        ),
    )
    changes: dict[str, SettingFieldValue] = Field(
        default_factory=dict,
        max_length=24,
        description=(
            "只放需要改变的字段，不能重写未要求修改的字段。"
            "人物字段：name、role、external_goal、internal_need、"
            "central_conflict、secret、traits、speech_style、background、"
            "initial_state、character_arc；世界资料字段：entry_type、name、"
            "description、constraints；关系字段：character_a_name、"
            "character_b_name、relationship、tension、change_direction；"
            "蓝图字段：central_question、protagonist_goal、core_conflict、"
            "stakes、opening_state、ending_state、major_turns、must_payoffs、"
            "forbidden_shortcuts、author_notes；剧情线字段：arc_type、title、"
            "dramatic_question、promise、start_state、target_payoff、"
            "involved_characters、planned_turns、lifecycle_status、priority、"
            "author_notes；文风字段：narrative_tense、narrative_distance、"
            "tone、narration_rules、sentence_rhythm、dialogue_voice、"
            "sensory_palette、metaphor_policy、allowed_omissions、"
            "preferred_patterns、banned_expressions、style_examples、"
            "author_notes；规则字段：category、title、content。"
            "类型要求：major_turns、must_payoffs、forbidden_shortcuts、"
            "involved_characters、planned_turns、preferred_patterns、"
            "banned_expressions、style_examples 必须是字符串数组；priority"
            " 必须是 1–5 的整数；其余字段必须是字符串。entry_type 只能是"
            " background、rule、faction、location、element；arc_type 只能"
            "是 main、subplot、character、relationship、mystery、world；"
            "lifecycle_status 只能是 planned、active、paused、resolved、"
            "abandoned；category 只能是 core、world、character、structure、"
            "style。新增剧情线时，title、dramatic_question、promise、"
            "target_payoff 都必须提供且不能为空"
        ),
    )
    reason: str = Field(
        default="",
        max_length=800,
        description="这次局部修改解决什么创作问题",
    )

    @model_validator(mode="after")
    def validate_target_and_changes(self) -> "AssistantStructuredSettingEdit":
        if self.action in {"update", "delete"} and not (
            self.target_id or self.target_name
        ):
            raise ValueError("修改或删除结构化资料时必须指定目标")
        if self.action in {"create", "update"} and not self.changes:
            raise ValueError("新增或修改结构化资料时必须包含 changes")
        if self.action == "delete" and self.changes:
            raise ValueError("删除结构化资料时 changes 必须为空")
        return self


class AssistantSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="作品正式书名",
    )
    genre: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="作品题材或类型",
    )
    premise: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="概括全书核心因果的一句话故事",
    )
    theme: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="作品希望持续探索的主题",
    )
    story_promise: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="作品向读者承诺的主要阅读体验",
    )
    target_audience: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="目标读者",
    )
    core_appeal: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="作品最核心、可持续兑现的吸引力",
    )
    ending_constraint: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="结局必须满足的全局约束",
    )
    world_setting: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=20_000,
        description=(
            "只用于全书级时代、空间、社会秩序和世界运行逻辑的概述；"
            "不得塞入人物卡、具体情节、章节计划或文风要求"
        ),
    )
    style_guide: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=10_000,
        description="全书通用的叙事与语言规范",
    )
    point_of_view: Optional[
        Literal["第三人称限知", "第一人称", "第三人称全知", "多视角"]
    ] = None
    archive_rules: Optional[list[AssistantArchiveRuleProposal]] = Field(
        default=None,
        min_length=1,
        description=(
            "无法准确放入上述全局字段的具体作品资料。每条必须归入作品概览、"
            "世界、人物、剧情与结构、叙事与文风之一"
        ),
    )
    structured_edits: Optional[list[AssistantStructuredSettingEdit]] = Field(
        default=None,
        min_length=1,
        description=(
            "对具体资料对象的可确认局部编辑。已存在的人物、世界资料、"
            "关系、蓝图、剧情线和文风必须优先使用这里，不得退化成"
            "重复的 archive_rules"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def omit_blank_optional_text(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        cleaned = dict(value)
        for field_name in (
            "title",
            "genre",
            "premise",
            "theme",
            "story_promise",
            "target_audience",
            "core_appeal",
            "ending_constraint",
            "world_setting",
            "style_guide",
        ):
            field_value = cleaned.get(field_name)
            if isinstance(field_value, str) and not field_value.strip():
                cleaned[field_name] = None
        return cleaned

    @model_validator(mode="after")
    def ensure_not_empty(self) -> "AssistantSettingsPatch":
        if not self.model_dump(exclude_none=True):
            raise ValueError("settings_patch 至少需要包含一个设定字段")
        return self


class AssistantStoryPlanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    blueprint: StoryBlueprint
    rationale: str = Field(min_length=2, max_length=1200)

    @model_validator(mode="after")
    def ensure_confirmable_blueprint(self) -> "AssistantStoryPlanProposal":
        self.blueprint.ensure_confirmable()
        return self


class AssistantChapterEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chapter_id: str = Field(min_length=1, max_length=64)
    expected_revision: str = Field(min_length=64, max_length=64)
    title: Optional[str] = Field(default=None, max_length=200)
    outline: Optional[str] = None
    key_points: Optional[str] = None
    position: Optional[int] = Field(default=None, ge=1)
    delete: Optional[bool] = None

    @model_validator(mode="after")
    def ensure_change(self) -> "AssistantChapterEdit":
        if self.delete is False:
            raise ValueError("delete 只能在明确删除章节时设为 true")
        if not self.model_dump(
            exclude={"chapter_id", "expected_revision"},
            exclude_none=True,
        ):
            raise ValueError("章节修改至少需要包含一个字段")
        if self.delete and self.model_dump(
            exclude={"chapter_id", "expected_revision", "delete"},
            exclude_none=True,
        ):
            raise ValueError("删除章节不能同时修改其他章节字段")
        return self


class AssistantChapterPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    edits: list[AssistantChapterEdit] = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_unique_chapters(self) -> "AssistantChapterPatch":
        chapter_ids = [edit.chapter_id for edit in self.edits]
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("同一章节在一次修改中只能出现一次")
        return self


class AssistantNoteEdit(BaseModel):
    """One optimistic, project-scoped author-note mutation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["create", "update", "delete"]
    note_key: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[^/\\\s][^/\\]*$",
        description="book/notes/author/ 下稳定且不含路径分隔符的文件名（不含 .md）",
    )
    note_id: Optional[str] = Field(default=None, max_length=64)
    expected_revision: str = Field(min_length=3, max_length=128)
    title: Optional[str] = Field(default=None, max_length=200)
    content: Optional[str] = None
    rationale: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def validate_note_mutation(self) -> "AssistantNoteEdit":
        if self.note_key in {".", ".."} or self.note_key.startswith("."):
            raise ValueError("笔记文件名不能是隐藏路径")
        if self.action == "create":
            if self.expected_revision != "new" or self.note_id is not None:
                raise ValueError("新建笔记必须使用 revision=new 且不能指定 note_id")
            if not str(self.content or "").strip():
                raise ValueError("新建笔记内容不能为空")
        elif self.action == "update":
            if not self.note_id:
                raise ValueError("更新笔记必须指定 note_id")
            if not str(self.content or "").strip():
                raise ValueError("更新笔记内容不能为空")
        else:
            if not self.note_id:
                raise ValueError("删除笔记必须指定 note_id")
            if self.content is not None or self.title is not None:
                raise ValueError("删除笔记不能同时提供内容")
        return self


class AssistantNotePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    edits: list[AssistantNoteEdit] = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_unique_notes(self) -> "AssistantNotePatch":
        keys = [edit.note_key for edit in self.edits]
        if len(keys) != len(set(keys)):
            raise ValueError("同一笔记在一次修改中只能出现一次")
        return self


class AssistantVersionRestoreProposal(BaseModel):
    """Restore immutable history by creating a new main HEAD version."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chapter_id: str = Field(min_length=1, max_length=64)
    version_id: str = Field(min_length=1, max_length=64)
    source_expected_revision: str = Field(min_length=64, max_length=64)
    head_version_id: Optional[str] = Field(default=None, max_length=64)
    head_expected_revision: str = Field(min_length=64, max_length=64)
    rationale: str = Field(min_length=2, max_length=800)


class AssistantTechniqueCardProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_document_id: str = Field(min_length=1, max_length=64)
    source_chapter_id: str = Field(min_length=1, max_length=64)
    source_expected_revision: str = Field(min_length=64, max_length=64)
    observation: TechniqueObservation
    author_note: str = Field(default="", max_length=2000)


class AssistantTechniquePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    cards: list[AssistantTechniqueCardProposal] = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_unique_cards(self) -> "AssistantTechniquePatch":
        identities = [
            (card.source_chapter_id, card.observation.name)
            for card in self.cards
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("同一来源章节不能重复创建同名技法卡")
        return self


class AssistantChapterWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    workflow_id: str = Field(min_length=1, max_length=64)
    status: Literal["completed", "paused"]
    total_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    chapter_ids: list[str] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)
    error: str = ""


class AssistantChatResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    citations: list[AssistantCitationProposal] = Field(default_factory=list)
    draft: Optional[AssistantDraftProposal] = None
    settings_patch: Optional[AssistantSettingsPatch] = None
    story_plan: Optional[AssistantStoryPlanProposal] = None
    chapter_patch: Optional[AssistantChapterPatch] = None
    note_patch: Optional[AssistantNotePatch] = None
    version_restore: Optional[AssistantVersionRestoreProposal] = None
    technique_patch: Optional[AssistantTechniquePatch] = None
    chapter_workflow: Optional[AssistantChapterWorkflowResult] = None


@dataclass(frozen=True)
class AssistantChatResponse:
    result: AssistantChatResult
    raw_response: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    accessed_sources: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ChapterDraftAuditResponse:
    result: ChapterDraftAuditResult
    raw_response: str
    input_tokens: int
    output_tokens: int
