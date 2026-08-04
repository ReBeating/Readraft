import asyncio

import pytest

from app.agent_runtime import (
    AgentProgressTracker,
    AgentRun,
    AgentRunCancelled,
    AgentRunPhase,
    semantic_tool_fingerprint,
)


def test_agent_run_persists_valid_transitions_and_checks_cancellation():
    events = []
    cancelled = False

    def transition_callback(**payload):
        events.append(payload)
        return True

    def cancellation_check(**_payload):
        return cancelled

    runtime = AgentRun(
        message_id="message-1",
        claim_token="claim-1",
        transition_callback=transition_callback,
        cancellation_check=cancellation_check,
    )

    async def scenario():
        nonlocal cancelled
        await runtime.transition(
            AgentRunPhase.ROUTING,
            event_type="run.routing",
            label="正在理解请求",
        )
        await runtime.transition(
            AgentRunPhase.PREPARING_CONTEXT,
            event_type="context.preparing",
            label="正在准备上下文",
        )
        cancelled = True
        with pytest.raises(AgentRunCancelled):
            await runtime.checkpoint()

    asyncio.run(scenario())
    assert [event["phase"] for event in events] == [
        "routing",
        "preparing_context",
    ]


def test_agent_run_restores_phase_after_model_retry():
    events = []
    runtime = AgentRun(
        message_id="message-1",
        claim_token="claim-1",
        transition_callback=lambda **payload: events.append(payload) or True,
        cancellation_check=lambda **_payload: False,
    )

    async def scenario():
        await runtime.transition(
            AgentRunPhase.ROUTING,
            event_type="run.routing",
            label="正在理解请求",
        )
        await runtime.handle_model_runtime_event(
            "retry_scheduled", {"attempt": 1}
        )
        await runtime.handle_model_runtime_event(
            "retry_resumed", {"attempt": 1}
        )

    asyncio.run(scenario())
    assert runtime.phase == AgentRunPhase.ROUTING
    assert [event["phase"] for event in events] == [
        "routing",
        "retrying",
        "routing",
    ]


def test_agent_action_has_its_own_runtime_phase_and_retry_resume():
    events = []
    runtime = AgentRun(
        message_id="message-action",
        claim_token="claim-action",
        transition_callback=lambda **payload: events.append(payload) or True,
        cancellation_check=lambda **_payload: False,
    )

    async def scenario():
        await runtime.transition(
            AgentRunPhase.ROUTING,
            event_type="run.routing",
            label="正在理解请求",
        )
        await runtime.transition(
            AgentRunPhase.PREPARING_CONTEXT,
            event_type="context.preparing",
            label="正在准备上下文",
        )
        await runtime.transition(
            AgentRunPhase.MODEL,
            event_type="model.started",
            label="正在决策",
        )
        await runtime.transition(
            AgentRunPhase.ACTION,
            event_type="action.started",
            label="创作章节正文",
        )
        await runtime.handle_model_runtime_event(
            "retry_scheduled", {"attempt": 1}
        )
        await runtime.handle_model_runtime_event(
            "retry_resumed", {"attempt": 1}
        )

    asyncio.run(scenario())
    assert runtime.phase == AgentRunPhase.ACTION
    assert [event["phase"] for event in events] == [
        "routing",
        "preparing_context",
        "model",
        "action",
        "retrying",
        "action",
    ]


def test_progress_tracker_detects_semantically_repeated_search():
    first = semantic_tool_fingerprint(
        "search_web", {"query": "北京 天气", "max_results": 5}
    )
    reordered = semantic_tool_fingerprint(
        "search_web", {"query": "天气 北京", "max_results": 5}
    )
    assert first == reordered

    tracker = AgentProgressTracker()
    first_result = tracker.assess(
        tool_name="search_web",
        arguments={"query": "北京 天气", "max_results": 5},
        status="completed",
        result={"results": [{"title": "天气"}]},
    )
    assert first_result.made_progress
    assert not first_result.should_stop
    second_result = tracker.assess(
        tool_name="search_web",
        arguments={"query": "天气 北京", "max_results": 5},
        status="completed",
        result={"results": [{"title": "天气"}]},
    )
    assert not second_result.should_stop
    third_result = tracker.assess(
        tool_name="search_web",
        arguments={"query": "北京 天气", "max_results": 5},
        status="completed",
        result={"results": [{"title": "天气"}]},
    )
    assert third_result.should_stop
    assert third_result.reason == "semantic_repeated_tool_call"


def test_progress_tracker_stops_after_distinct_no_evidence_steps():
    tracker = AgentProgressTracker(max_no_progress_steps=3)
    results = [
        tracker.assess(
            tool_name="search_story_memory",
            arguments={"query": f"线索 {index}"},
            status="completed",
            result={},
        )
        for index in range(3)
    ]
    assert not results[0].should_stop
    assert not results[1].should_stop
    assert results[2].should_stop
    assert results[2].reason == "no_new_evidence"
