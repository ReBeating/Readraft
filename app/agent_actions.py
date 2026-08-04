"""High-level actions chosen by the Agent after it gathers evidence."""

from __future__ import annotations

from typing import Collection, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_callables import AgentCallableSpec
from .agent_capabilities import (
    WRITE_CHAPTER,
    CREATE_CHAPTER,
    RUN_BOUNDED_TASK,
)


class ComposeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(min_length=1, max_length=500)
    instruction: str = Field(min_length=2, max_length=6000)
    expected_revision: str = Field(min_length=8, max_length=128)
    mode: Literal["replace", "append"] = "replace"
    target_chars: int | None = Field(default=None, ge=80, le=20_000)


class CreateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource: Literal["chapter"] = "chapter"
    title: str = Field(default="", max_length=200)
    outline: str = Field(default="", max_length=6000)
    key_points: str = Field(default="", max_length=6000)


class TaskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal[
        "general",
        "continuity",
        "structure",
        "character",
        "style",
        "research",
    ] = "general"
    objective: str = Field(min_length=2, max_length=2000)
    paths: list[str] = Field(min_length=1, max_length=12)


class SeriesChapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(default="", max_length=200)
    outline: str = Field(default="", max_length=6000)
    key_points: str = Field(default="", max_length=6000)
    instruction: str = Field(min_length=2, max_length=6000)
    target_chars: int | None = Field(default=None, ge=500, le=20_000)


class SeriesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chapters: list[SeriesChapterSpec] = Field(
        default_factory=list,
        max_length=12,
    )
    resume_latest: bool = False

    @model_validator(mode="after")
    def validate_start_or_resume(self) -> "SeriesArguments":
        if self.resume_latest == bool(self.chapters):
            raise ValueError("series 必须二选一：提供章节列表，或恢复最近工作流")
        return self


AGENT_ACTION_SPECS: dict[str, AgentCallableSpec] = {
    "compose": AgentCallableSpec(
        name="compose",
        label="创作章节正文",
        description=(
            "在你已经确定写作目标并读取必要章节与设定后，执行章节正文"
            "创作、续写或大范围重写。提供目标 path、revision、写作要求和"
            "replace/append 模式；短段追加可按作者要求设置较小 target_chars。"
            "不要在参数中自行生成整章正文。"
        ),
        input_model=ComposeArguments,
        category="agent_action",
        read_only=False,
    ),
    "create": AgentCallableSpec(
        name="create",
        label="创建下一章",
        description=(
            "在当前 main 作品末尾创建一个新的空白章节，并把本轮工作范围切换"
            "到该章节。应在本轮任何正文或设定修改之前调用；返回新章节路径后"
            "先 read，再按需使用 compose。人物、关系等结构化资料仍使用 write。"
        ),
        input_model=CreateArguments,
        category="agent_action",
        read_only=False,
    ),
    "task": AgentCallableSpec(
        name="task",
        label="委托专项分析",
        description=(
            "把一个边界明确的只读分析委托给专项 Agent。调用者必须先确定"
            "目标并提供最多 12 个准确资源路径；专项 Agent 只能阅读这些资源，"
            "不能调用工具、修改作品或继续委托。适合连续性、结构、人物、文风"
            "和资料核对；简单问题直接自行处理。"
        ),
        input_model=TaskArguments,
        category="agent_action",
        read_only=True,
    ),
    "series": AgentCallableSpec(
        name="series",
        label="连续创作多章",
        description=(
            "仅在作者明确要求连续创作多章时使用。首次传入按顺序排列的章节"
            "目标（最多 12 章）；服务端逐章创建、生成和独立提交，任一章失败"
            "会暂停并保留已完成章节。之后传 resume_latest=true 从失败断点继续，"
            "不会重写已完成章节。单章任务继续使用 create/compose。"
        ),
        input_model=SeriesArguments,
        category="agent_action",
        read_only=False,
    ),
}


def available_agent_actions(
    *,
    capabilities: Collection[str],
    main_writable: bool,
    has_writable_chapter: bool,
) -> list[AgentCallableSpec]:
    """Grant high-level actions from the resolved Agent capability set."""

    actions: list[AgentCallableSpec] = []
    if RUN_BOUNDED_TASK in capabilities:
        actions.append(AGENT_ACTION_SPECS["task"])
    if main_writable and CREATE_CHAPTER in capabilities:
        actions.append(AGENT_ACTION_SPECS["create"])
    if (
        main_writable
        and CREATE_CHAPTER in capabilities
        and WRITE_CHAPTER in capabilities
    ):
        actions.append(AGENT_ACTION_SPECS["series"])
    if (
        main_writable
        and has_writable_chapter
        and WRITE_CHAPTER in capabilities
    ):
        actions.append(AGENT_ACTION_SPECS["compose"])
    return actions
