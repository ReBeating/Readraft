import asyncio

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


def test_all_prose_paths_share_one_canonical_writing_contract():
    assert PROSE_WRITING_SYSTEM_PROMPT in WRITING_SYSTEM_PROMPT


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
    assert model.request["max_tokens"] == 6000
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
