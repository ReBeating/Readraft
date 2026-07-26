from app.model_smoke import (
    SAFE_SMOKE_MODEL_ADAPTER_PROMPT,
    synthetic_writing_context,
)


def test_synthetic_smoke_fixture_is_small_and_self_contained():
    context = synthetic_writing_context()
    serialized = str(context)

    assert context["chapter"]["project_title"] == "纸灯塔"
    assert context["chapter"]["target_chapter_chars"] == 260
    assert len(serialized) < 2_000
    assert "合成" in SAFE_SMOKE_MODEL_ADAPTER_PROMPT
    assert "真实作品" in SAFE_SMOKE_MODEL_ADAPTER_PROMPT
