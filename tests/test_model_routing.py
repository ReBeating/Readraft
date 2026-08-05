from app.model_routing import normalize_quality_mode, route_model_task


def test_low_forces_fast_model_without_thinking_for_every_task():
    for task in ("fast", "discussion", "reasoning", "deep", "prose"):
        route = route_model_task("low", task)
        assert route.model_role == "fast"
        assert route.reasoning_policy == "fast"


def test_standard_splits_light_discussion_and_quality_work():
    assert route_model_task("standard", "fast") == (
        route_model_task("low", "fast")
    )
    discussion = route_model_task("standard", "discussion")
    assert discussion.model_role == "fast"
    assert discussion.reasoning_policy == "reasoning"
    writing = route_model_task("standard", "reasoning")
    assert writing.model_role == "quality"
    assert writing.reasoning_policy == "reasoning"
    planning = route_model_task("standard", "deep")
    assert planning.model_role == "quality"
    assert planning.reasoning_policy == "deep"
    prose = route_model_task("standard", "prose")
    assert prose.model_role == "quality"
    assert prose.reasoning_policy == "fast"


def test_max_forces_quality_model_and_deep_reasoning_for_every_task():
    for task in ("fast", "discussion", "reasoning", "deep", "prose"):
        route = route_model_task("max", task)
        assert route.model_role == "quality"
        assert route.reasoning_policy == "deep"


def test_unknown_quality_mode_falls_back_to_standard():
    assert normalize_quality_mode("unexpected") == "standard"
