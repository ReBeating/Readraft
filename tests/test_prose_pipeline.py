import asyncio
from types import SimpleNamespace

from app.agent_model import AssistantModelTurn, MockAgentModel
from app.prose_pipeline import ProseDraftPipeline
from app.prose_craft import PROSE_WRITING_SYSTEM_PROMPT
from app.writing import WRITING_SYSTEM_PROMPT


class PlainProseModel(MockAgentModel):
    def __init__(self):
        self.request = None

    async def native_turn(self, **kwargs):
        self.request = kwargs
        callback = kwargs.get("on_text_delta")
        if callback is not None:
            callback("海雾压低了灯塔的轮廓。")
        return AssistantModelTurn(
            content="海雾压低了灯塔的轮廓。\n\n林岚把信收进口袋。",
            reasoning="内部规划不会进入正文",
            tool_calls=(),
            finish_reason="stop",
            input_tokens=30,
            output_tokens=20,
        )


class ThinkingProseModel(PlainProseModel):
    settings = SimpleNamespace(model_thinking=True)


class ContinuedProseModel(MockAgentModel):
    def __init__(self):
        self.requests = []

    async def native_turn(self, **kwargs):
        self.requests.append(kwargs)
        first = len(self.requests) == 1
        content = "海雾压住灯塔。" if first else "林岚沿堤岸继续向前。"
        callback = kwargs.get("on_text_delta")
        if callback is not None:
            result = callback(content)
            if asyncio.iscoroutine(result):
                await result
        return AssistantModelTurn(
            content=content,
            reasoning="",
            tool_calls=(),
            finish_reason="length" if first else "stop",
            input_tokens=20,
            output_tokens=10,
        )


def test_all_prose_paths_share_one_canonical_writing_contract():
    assert PROSE_WRITING_SYSTEM_PROMPT in WRITING_SYSTEM_PROMPT
    assert "required_change 是本轮验收条件" in PROSE_WRITING_SYSTEM_PROMPT
    assert "未提供的重大事实保持未知" in PROSE_WRITING_SYSTEM_PROMPT
    assert "定向改写时只改变作者要求的表达维度" in PROSE_WRITING_SYSTEM_PROMPT


def test_prose_pipeline_uses_plain_text_turn_without_tools():
    model = PlainProseModel()
    deltas = []

    async def scenario():
        return await ProseDraftPipeline().generate(
            model=model,
            packet={
                "instruction": "让林岚决定去灯塔",
                "target_chars": 2000,
                "chapter": {"current_text": ""},
            },
            provider_user_id="user-1",
            on_text_delta=lambda value: deltas.append(value),
        )

    result = asyncio.run(scenario())

    assert model.request["tools"] == []
    assert model.request["max_tokens"] is None
    system_prompt = model.request["messages"][0]["content"]
    assert "完整场景必须" in system_prompt
    assert "不再用旁白复述同一含义" in system_prompt
    assert result.content.startswith("海雾压低")
    assert "内部规划" not in result.content
    assert deltas == ["海雾压低了灯塔的轮廓。"]


def test_prose_pipeline_selects_only_relevant_craft_modules():
    model = PlainProseModel()

    async def scenario():
        return await ProseDraftPipeline().generate(
            model=model,
            packet={
                "instruction": "写一场审问式对话，藏住真正线索。",
                "genre": "悬疑",
                "target_chars": 1200,
                "chapter": {"current_text": ""},
                "scene_contract": {
                    "central_conflict": "林岚试探守塔人是否撒谎",
                    "must_happen": ["守塔人无意说出潮汐时间"],
                },
            },
            provider_user_id="user-1",
        )

    result = asyncio.run(scenario())

    assert "dialogue_subtext" in result.craft_modules
    assert "suspense_information" in result.craft_modules
    assert 2 <= len(result.craft_modules) <= 4
    system_prompt = model.request["messages"][0]["content"]
    assert "【对话与潜台词】" in system_prompt
    assert "【悬念与信息控制】" in system_prompt
    assert PROSE_WRITING_SYSTEM_PROMPT in system_prompt


def test_prose_pipeline_leaves_output_budget_automatic_for_reasoning_models():
    model = ThinkingProseModel()

    async def scenario():
        return await ProseDraftPipeline().generate(
            model=model,
            packet={
                "instruction": "让人物完成决定",
                "target_chars": 1000,
                "chapter": {"current_text": ""},
            },
            provider_user_id="user-1",
        )

    asyncio.run(scenario())

    assert model.request["max_tokens"] is None


def test_prose_pipeline_continues_provider_length_stop_without_duplication():
    model = ContinuedProseModel()
    streamed = []

    async def scenario():
        return await ProseDraftPipeline().generate(
            model=model,
            packet={
                "instruction": "让林岚走到灯塔",
                "target_chars": 2000,
                "chapter": {"current_text": ""},
            },
            provider_user_id="user-1",
            on_text_delta=lambda value: streamed.append(value),
        )

    result = asyncio.run(scenario())

    assert result.content == "海雾压住灯塔。林岚沿堤岸继续向前。"
    assert result.input_tokens == 40
    assert result.output_tokens == 20
    assert len(model.requests) == 2
    assert model.requests[0]["max_tokens"] is None
    assert model.requests[1]["messages"][-2]["role"] == "assistant"
    assert streamed == [
        "海雾压住灯塔。",
        "海雾压住灯塔。林岚沿堤岸继续向前。",
    ]
