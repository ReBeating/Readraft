from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.context_compiler import compile_canonical_memory
from app.continuity import ContinuityService, replay_canonical_state
from app.db import Database, utc_now
from app.memory_identity import MemoryIdentityService, expand_identity_terms
from app.main import create_app
from app.memory_schema import StoryDelta
from app.memory_service import MemoryService
from app.security import hash_password


def _build_project(tmp_path: Path, *, username: str = "continuity-author"):
    database = Database(tmp_path / f"{username}.db")
    database.initialize()
    user_id = database.create_user(
        username, hash_password("password-123")
    )
    project_id = f"project-{username}"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚追查一封不可能出现的来信。",
        world_setting="当代海港。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="记者",
        traits="执拗",
        background="来自雾港",
        character_arc="学会信任同伴",
    )
    chapter_ids = []
    for position in range(1, 4):
        chapter_id = f"{username}-chapter-{position}"
        chapter_dir = tmp_path / username / str(position)
        chapter_dir.mkdir(parents=True)
        content_path = chapter_dir / "content.txt"
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=f"第{position}章",
            outline=f"第{position}章大纲",
            key_points="推进调查",
            content_path=content_path,
        )
        chapter_ids.append(chapter_id)
    return database, user_id, project_id, chapter_ids


def _accept_version(
    database: Database,
    *,
    tmp_path: Path,
    user_id: int,
    project_id: str,
    chapter_id: str,
    label: str,
) -> str:
    path = tmp_path / chapter_id / "versions" / f"{label}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"{chapter_id} 的 {label} 正文"
    path.write_text(content, encoding="utf-8")
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=path,
        char_count=len(content),
    )
    assert version_id
    result = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_id=version_id,
        override_reason="测试作者确认此版本进入连续性正史账本",
    )
    assert result
    return version_id


def _delta(
    *,
    summary: str,
    goal_before: str,
    goal_after: str,
    location_before: str,
    location_after: str,
    relationship_before: str,
    relationship_after: str,
    item_action: str,
    item_from: str | None,
    item_to: str | None,
    time_before: str,
    time_after: str,
) -> StoryDelta:
    return StoryDelta.model_validate(
        {
            "chapter_summary": summary,
            "keywords": ["蓝玻璃钥匙", "雾港"],
            "unresolved_questions": [],
            "character_changes": [
                {
                    "character_name": "林岚",
                    "aspect": "goal",
                    "before": goal_before,
                    "after": goal_after,
                    "evidence": "林岚明确说出下一步打算。",
                }
            ],
            "relationship_changes": [
                {
                    "character_a": "林岚",
                    "character_b": "周时",
                    "before": relationship_before,
                    "after": relationship_after,
                    "evidence": "两人重新约定合作方式。",
                }
            ],
            "location_changes": [
                {
                    "subject_name": "林岚",
                    "from_location": location_before,
                    "to_location": location_after,
                    "evidence": "林岚乘车抵达新地点。",
                }
            ],
            "item_changes": [
                {
                    "item_name": "蓝玻璃钥匙",
                    "action": item_action,
                    "from_holder": item_from,
                    "to_holder": item_to,
                    "state": "齿面有盐霜",
                    "evidence": "钥匙在两人眼前完成交接。",
                }
            ],
            "knowledge_changes": [],
            "plot_thread_changes": [],
            "foreshadowing_changes": [],
            "events": [],
            "time_advance": {
                "from_time": time_before,
                "to_time": time_after,
                "elapsed": "约十二小时",
            },
        }
    )


def _lifecycle_delta(
    *,
    summary: str,
    knowledge_fact: str,
    knowledge_state: str,
    plot_action: str,
    hook_action: str,
) -> StoryDelta:
    return StoryDelta.model_validate(
        {
            "chapter_summary": summary,
            "keywords": ["蓝玻璃钥匙", "父亲失踪之谜"],
            "unresolved_questions": [],
            "character_changes": [],
            "relationship_changes": [],
            "location_changes": [],
            "item_changes": [],
            "knowledge_changes": [
                {
                    "character_name": "林岚",
                    "fact": knowledge_fact,
                    "state": knowledge_state,
                    "learned_via": (
                        "查看父亲留下的便笺"
                        if knowledge_state != "forgets"
                        else ""
                    ),
                    "evidence": "林岚对这条信息作出了明确反应。",
                }
            ],
            "plot_thread_changes": [
                {
                    "thread_name": "父亲失踪之谜",
                    "thread_type": "main",
                    "action": plot_action,
                    "update": f"本章对失踪之谜执行 {plot_action}。",
                    "promise": (
                        "解释父亲失踪与蓝玻璃钥匙的关系"
                        if plot_action == "opened"
                        else ""
                    ),
                    "target_payoff": (
                        "找到父亲失踪的直接证据"
                        if plot_action == "opened"
                        else ""
                    ),
                    "evidence": "林岚把钥匙与父亲失踪案并列记录。",
                }
            ],
            "foreshadowing_changes": [
                {
                    "hook_name": "钥匙齿面的盐霜",
                    "action": hook_action,
                    "description": f"本章对盐霜伏笔执行 {hook_action}。",
                    "intended_payoff": (
                        "指向只在退潮时出现的旧港密室"
                        if hook_action == "setup"
                        else ""
                    ),
                    "evidence": "钥匙齿面的盐霜在灯下发白。",
                }
            ],
            "events": [],
            "time_advance": None,
        }
    )


def _knowledge_delta(
    *,
    summary: str,
    character_name: str,
    fact: str,
    state: str,
) -> StoryDelta:
    return StoryDelta.model_validate(
        {
            "chapter_summary": summary,
            "keywords": ["北塔", "寄信人"],
            "unresolved_questions": [],
            "character_changes": [],
            "relationship_changes": [],
            "location_changes": [],
            "item_changes": [],
            "knowledge_changes": [
                {
                    "character_name": character_name,
                    "fact": fact,
                    "canonical_fact": "",
                    "state": state,
                    "learned_via": "查看地址",
                    "evidence": "她指认了纸上的北塔地址。",
                }
            ],
            "plot_thread_changes": [],
            "foreshadowing_changes": [],
            "events": [],
            "time_advance": None,
        }
    )


def _event_delta(
    *,
    summary: str,
    event_key: str,
    cause_event_keys: list[str],
) -> StoryDelta:
    return StoryDelta.model_validate(
        {
            "chapter_summary": summary,
            "keywords": ["父亲来信", "雾港"],
            "unresolved_questions": [],
            "character_changes": [],
            "relationship_changes": [],
            "location_changes": [],
            "item_changes": [],
            "knowledge_changes": [],
            "plot_thread_changes": [],
            "foreshadowing_changes": [],
            "events": [
                {
                    "event_key": event_key,
                    "summary": summary,
                    "participants": ["林岚"],
                    "location": "雾港",
                    "story_time": "",
                    "causes": [],
                    "cause_event_keys": cause_event_keys,
                    "effects": [],
                    "evidence": summary,
                }
            ],
            "time_advance": None,
        }
    )


def _project(
    memory: MemoryService,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
    version_id: str,
    delta: StoryDelta,
) -> str:
    delta_id = memory.create_proposal(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_id=version_id,
        payload=delta,
    )
    assert memory.accept_delta(user_id=user_id, delta_id=delta_id)
    return delta_id


def test_replay_builds_current_state_and_enters_writing_context(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(tmp_path)
    memory = MemoryService(database)
    version_1 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="canon-1",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=version_1,
        delta=_delta(
            summary="林岚抵达雾港并取得钥匙。",
            goal_before="离开旧案",
            goal_after="查清钥匙来历",
            location_before="城外车站",
            location_after="雾港旅店",
            relationship_before="陌生人",
            relationship_after="临时盟友",
            item_action="acquired",
            item_from=None,
            item_to="林岚",
            time_before="第一日清晨",
            time_after="第一日夜晚",
        ),
    )
    version_2 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        label="canon-2",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        version_id=version_2,
        delta=_delta(
            summary="两人进入档案馆，周时接过钥匙。",
            goal_before="查清钥匙来历",
            goal_after="打开旧港密档",
            location_before="雾港旅店",
            location_after="旧港档案馆",
            relationship_before="临时盟友",
            relationship_after="互相试探的搭档",
            item_action="transferred",
            item_from="林岚",
            item_to="周时",
            time_before="第一日夜晚",
            time_after="第二日清晨",
        ),
    )

    dashboard = ContinuityService(database).get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert dashboard
    assert dashboard["latest_run"]["replayed_chapter_count"] == 2
    assert dashboard["counts"]["active"] == 0
    state = dashboard["state"]
    assert state["story_time"]["value"] == "第二日清晨"
    assert state["characters"]["林岚"]["location"]["value"] == "旧港档案馆"
    assert (
        state["characters"]["林岚"]["attributes"]["goal"]["value"]
        == "打开旧港密档"
    )
    assert next(iter(state["relationships"].values()))["value"] == (
        "互相试探的搭档"
    )
    assert state["items"]["蓝玻璃钥匙"]["holder"]["value"] == "周时"

    context = database.get_writing_context(user_id, chapters[2])
    canonical = context["canonical_memory"]
    assert canonical["current_state"]["story_time"]["value"] == "第二日清晨"
    assert (
        canonical["current_state"]["items"]["蓝玻璃钥匙"]["holder"]["value"]
        == "周时"
    )
    compiled = compile_canonical_memory(canonical)
    assert "旧港档案馆" in json.dumps(compiled, ensure_ascii=False)


def test_mismatches_are_deterministic_and_acknowledgement_survives_replay(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(tmp_path)
    memory = MemoryService(database)
    version_1 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="canon-1",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=version_1,
        delta=_delta(
            summary="建立第一章状态基线。",
            goal_before="放弃调查",
            goal_after="追查钥匙",
            location_before="城外",
            location_after="雾港",
            relationship_before="陌生",
            relationship_after="盟友",
            item_action="acquired",
            item_from=None,
            item_to="林岚",
            time_before="第一日",
            time_after="第一日夜晚",
        ),
    )
    version_2 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        label="canon-2",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        version_id=version_2,
        delta=_delta(
            summary="第二章故意提交不一致的变化前状态。",
            goal_before="离开城市",
            goal_after="进入档案馆",
            location_before="机场",
            location_after="档案馆",
            relationship_before="敌人",
            relationship_after="互相怀疑",
            item_action="transferred",
            item_from="周时",
            item_to="林岚",
            time_before="第三日",
            time_after="第三日夜晚",
        ),
    )

    service = ContinuityService(database)
    dashboard = service.get_dashboard(user_id=user_id, project_id=project_id)
    issue_types = {issue["issue_type"] for issue in dashboard["issues"]}
    assert {
        "state_before_mismatch",
        "relationship_before_mismatch",
        "location_before_mismatch",
        "item_holder_mismatch",
        "story_time_mismatch",
    } <= issue_types
    assert dashboard["counts"]["hard"] == 3
    assert database.get_novel_chapter(
        user_id, project_id, chapters[1]
    )["needs_recheck"] == 1

    issue = next(
        item
        for item in dashboard["issues"]
        if item["issue_type"] == "location_before_mismatch"
    )
    assert service.set_issue_status(
        user_id=user_id,
        project_id=project_id,
        issue_id=issue["id"],
        action="acknowledge",
        author_note="人物在这一章故意伪造自己的行程。",
        updated_at=utc_now(),
    )
    with database.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        replay_canonical_state(
            connection,
            project_id=project_id,
            branch_id="main",
            trigger_type="test_replay",
            trigger_chapter_id=chapters[1],
            created_at=utc_now(),
        )
        connection.commit()
    refreshed = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    same_issue = next(
        item
        for item in refreshed["issues"]
        if item["fingerprint"] == issue["fingerprint"]
    )
    assert same_issue["status"] == "acknowledged"
    assert "伪造" in same_issue["author_note"]


def test_replacing_earlier_canon_replays_downstream_without_stale_state(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(tmp_path)
    memory = MemoryService(database)
    version_1 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="old-canon",
    )
    delta_1 = _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=version_1,
        delta=_delta(
            summary="第一章建立基线。",
            goal_before="离开",
            goal_after="调查",
            location_before="城外",
            location_after="雾港",
            relationship_before="陌生",
            relationship_after="盟友",
            item_action="acquired",
            item_from=None,
            item_to="林岚",
            time_before="第一日",
            time_after="第一日夜晚",
        ),
    )
    version_2 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        label="canon-2",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        version_id=version_2,
        delta=_delta(
            summary="第二章延续基线。",
            goal_before="调查",
            goal_after="潜入",
            location_before="雾港",
            location_after="档案馆",
            relationship_before="盟友",
            relationship_after="搭档",
            item_action="transferred",
            item_from="林岚",
            item_to="周时",
            time_before="第一日夜晚",
            time_after="第二日",
        ),
    )

    replacement = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="replacement-without-memory",
    )
    assert replacement != version_1
    assert memory.get_delta(user_id=user_id, delta_id=delta_1)["status"] == (
        "superseded"
    )
    dashboard = ContinuityService(database).get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert dashboard["latest_run"]["replayed_chapter_count"] == 1
    assert dashboard["latest_snapshot"]["chapter_id"] == chapters[1]
    assert dashboard["state"]["items"]["蓝玻璃钥匙"]["holder"]["value"] == (
        "周时"
    )
    assert any(
        issue["issue_type"] == "missing_baseline"
        for issue in dashboard["issues"]
    )
    with database.connection() as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM story_state_snapshots
            WHERE project_id=? AND chapter_id=?
            """,
            (project_id, chapters[0]),
        ).fetchone()[0] == 0
        connection.execute(
            "DELETE FROM novel_projects WHERE id=?", (project_id,)
        )
        connection.commit()
        for table in (
            "continuity_replay_runs",
            "story_state_snapshots",
            "continuity_issues",
        ):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE project_id=?",
                (project_id,),
            ).fetchone()[0] == 0


def test_knowledge_threads_and_foreshadowing_have_current_lifecycle_state(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(
        tmp_path, username="lifecycle-author"
    )
    memory = MemoryService(database)
    version_1 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="lifecycle-1",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=version_1,
        delta=_lifecycle_delta(
            summary="林岚得知钥匙可能打开旧港密室。",
            knowledge_fact="蓝玻璃钥匙可能打开旧港密室",
            knowledge_state="knows",
            plot_action="opened",
            hook_action="setup",
        ),
    )
    version_2 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        label="lifecycle-2",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        version_id=version_2,
        delta=_lifecycle_delta(
            summary="林岚受到药物影响，忘记密室用途但继续调查。",
            knowledge_fact="蓝玻璃钥匙可能打开旧港密室",
            knowledge_state="forgets",
            plot_action="advanced",
            hook_action="advanced",
        ),
    )

    service = ContinuityService(database)
    dashboard = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert dashboard["state"]["schema_version"] == 3
    assert dashboard["state"]["knowledge"]["林岚"][
        "蓝玻璃钥匙可能打开旧港密室"
    ]["state"] == "forgets"
    assert dashboard["state"]["plot_threads"]["父亲失踪之谜"][
        "status"
    ] == "active"
    assert dashboard["state"]["plot_threads"]["父亲失踪之谜"][
        "promise"
    ].startswith("解释父亲失踪")
    assert dashboard["state"]["foreshadowing"]["钥匙齿面的盐霜"][
        "status"
    ] == "advanced"
    assert dashboard["counts"]["knowledge_facts"] == 1
    assert dashboard["counts"]["open_plot_threads"] == 1
    assert dashboard["counts"]["open_foreshadowing"] == 1
    assert dashboard["counts"]["active"] == 0

    context = database.get_writing_context(user_id, chapters[2])
    current_state = context["canonical_memory"]["current_state"]
    assert current_state["knowledge"]["林岚"][
        "蓝玻璃钥匙可能打开旧港密室"
    ]["state"] == "forgets"
    assert current_state["plot_threads"]["父亲失踪之谜"][
        "status"
    ] == "active"
    compiled = compile_canonical_memory(context["canonical_memory"])
    assert compiled["character_knowledge"] == []
    assert compiled["plot_threads"] == []
    assert compiled["foreshadowing"] == []
    serialized = json.dumps(compiled["current_state"], ensure_ascii=False)
    assert "已经遗忘" not in serialized
    assert '"state": "forgets"' in serialized

    # Re-running only v11 upgrades legacy snapshots without calling AI.
    with database.connection() as connection:
        connection.execute(
            "UPDATE story_state_snapshots SET state_json=? "
            "WHERE project_id=?",
            ('{"schema_version":1}', project_id),
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version=11"
        )
        connection.commit()
    database.initialize()
    upgraded = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert upgraded["state"]["schema_version"] == 3
    assert upgraded["state"]["knowledge"]["林岚"][
        "蓝玻璃钥匙可能打开旧港密室"
    ]["state"] == "forgets"


def test_closed_threads_hooks_and_unknown_forgetting_raise_issues(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(
        tmp_path, username="lifecycle-conflict"
    )
    memory = MemoryService(database)
    lifecycle_steps = (
        ("knows", "opened", "setup", "建立承诺和伏笔。"),
        ("knows", "resolved", "payoff", "兑现承诺和伏笔。"),
        (
            "forgets",
            "advanced",
            "advanced",
            "在关闭后继续推进，并遗忘从未记录的信息。",
        ),
    )
    for position, (
        knowledge_state,
        plot_action,
        hook_action,
        summary,
    ) in enumerate(lifecycle_steps):
        version_id = _accept_version(
            database,
            tmp_path=tmp_path,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[position],
            label=f"conflict-{position + 1}",
        )
        _project(
            memory,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[position],
            version_id=version_id,
            delta=_lifecycle_delta(
                summary=summary,
                knowledge_fact=(
                    "寄信人住在北塔"
                    if position == 2
                    else "蓝玻璃钥匙可能打开旧港密室"
                ),
                knowledge_state=knowledge_state,
                plot_action=plot_action,
                hook_action=hook_action,
            ),
        )

    dashboard = ContinuityService(database).get_dashboard(
        user_id=user_id, project_id=project_id
    )
    issue_types = {issue["issue_type"] for issue in dashboard["issues"]}
    assert {
        "knowledge_without_baseline",
        "plot_thread_after_closed",
        "foreshadow_after_closed",
    } <= issue_types
    assert dashboard["counts"]["hard"] == 2
    assert dashboard["counts"]["warning"] == 1
    assert dashboard["state"]["plot_threads"]["父亲失踪之谜"][
        "status"
    ] == "active"
    assert dashboard["state"]["foreshadowing"]["钥匙齿面的盐霜"][
        "status"
    ] == "advanced"
    assert database.get_novel_chapter(
        user_id, project_id, chapters[2]
    )["needs_recheck"] == 1


def test_author_alias_rules_merge_character_and_fact_memory(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(
        tmp_path, username="identity-author"
    )
    memory = MemoryService(database)
    for position, delta in enumerate(
        (
            _knowledge_delta(
                summary="林记者查到寄信人的住址。",
                character_name="林记者",
                fact="寄信人住在北塔",
                state="knows",
            ),
            _knowledge_delta(
                summary="林岚忘记来信者住处。",
                character_name="林岚",
                fact="来信者居于北塔",
                state="forgets",
            ),
        )
    ):
        version_id = _accept_version(
            database,
            tmp_path=tmp_path,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[position],
            label=f"identity-{position + 1}",
        )
        _project(
            memory,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[position],
            version_id=version_id,
            delta=delta,
        )

    service = ContinuityService(database)
    before = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert before["counts"]["knowledge_facts"] == 2
    assert any(
        issue["issue_type"] == "knowledge_without_baseline"
        for issue in before["issues"]
    )

    identity_service = MemoryIdentityService(database)
    identity_service.save_rule(
        user_id=user_id,
        project_id=project_id,
        identity_type="character",
        canonical_text="林岚",
        aliases=["林记者"],
        updated_at=utc_now(),
    )
    identity_service.save_rule(
        user_id=user_id,
        project_id=project_id,
        identity_type="fact",
        canonical_text="寄信人住在北塔",
        aliases=["来信者居于北塔"],
        updated_at=utc_now(),
    )

    after = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert after["state"]["schema_version"] == 3
    assert after["counts"]["knowledge_facts"] == 1
    assert after["counts"]["identity_rules"] == 2
    assert after["state"]["knowledge"]["林岚"]["寄信人住在北塔"][
        "state"
    ] == "forgets"
    assert not any(
        issue["issue_type"] == "knowledge_without_baseline"
        for issue in after["issues"]
    )

    context = database.get_writing_context(user_id, chapters[2])
    assert context["canonical_memory"]["current_state"]["knowledge"][
        "林岚"
    ]["寄信人住在北塔"]["state"] == "forgets"
    assert any(
        "来信者居于北塔" in identity["aliases"]
        for identity in context["memory_identities"]
        if identity["identity_type"] == "fact"
    )
    with database.connection() as connection:
        expanded = expand_identity_terms(
            connection,
            project_id=project_id,
            terms=["寄信人住在北塔"],
        )
    assert "来信者居于北塔" in expanded
    assert identity_service.list_rules(
        user_id=user_id + 999, project_id=project_id
    ) is None


def test_event_causal_chain_replays_and_detects_missing_old_canon(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(
        tmp_path, username="causal-author"
    )
    memory = MemoryService(database)
    version_1 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="causal-1",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=version_1,
        delta=_event_delta(
            summary="林岚收到父亲署名的新信。",
            event_key="父亲来信抵达",
            cause_event_keys=[],
        ),
    )
    version_2 = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        label="causal-2",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
        version_id=version_2,
        delta=_event_delta(
            summary="林岚因来信返回雾港。",
            event_key="林岚返回雾港",
            cause_event_keys=["父亲来信抵达"],
        ),
    )

    service = ContinuityService(database)
    dashboard = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert dashboard["counts"]["events"] == 2
    assert dashboard["counts"]["causal_edges"] == 1
    edge = next(iter(dashboard["state"]["causal_edges"].values()))
    assert edge["cause"] == "父亲来信抵达"
    assert edge["effect"] == "林岚返回雾港"
    context = database.get_writing_context(user_id, chapters[2])
    current_state = context["canonical_memory"]["current_state"]
    assert current_state["events"]["父亲来信抵达"]["summary"].startswith(
        "林岚收到"
    )
    assert len(current_state["causal_edges"]) == 1

    replacement_version = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="causal-1-replacement",
    )
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=replacement_version,
        delta=_event_delta(
            summary="林岚收到一份普通账单。",
            event_key="普通账单抵达",
            cause_event_keys=[],
        ),
    )
    replayed = service.get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert "父亲来信抵达" not in replayed["state"]["events"]
    assert replayed["counts"]["causal_edges"] == 0
    assert any(
        issue["issue_type"] == "causal_reference_missing"
        and issue["entity_name"] == "林岚返回雾港"
        for issue in replayed["issues"]
    )


def test_event_causal_self_reference_and_duplicate_identity_are_blocked(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(
        tmp_path, username="causal-conflict"
    )
    memory = MemoryService(database)
    for position, delta in enumerate(
        (
            _event_delta(
                summary="林岚进入旧港密室。",
                event_key="进入旧港密室",
                cause_event_keys=[],
            ),
            _event_delta(
                summary="林岚在北塔收到来信。",
                event_key="进入旧港密室",
                cause_event_keys=["进入旧港密室"],
            ),
        )
    ):
        version_id = _accept_version(
            database,
            tmp_path=tmp_path,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[position],
            label=f"causal-conflict-{position + 1}",
        )
        _project(
            memory,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[position],
            version_id=version_id,
            delta=delta,
        )
    issues = ContinuityService(database).get_dashboard(
        user_id=user_id, project_id=project_id
    )["issues"]
    issue_types = {issue["issue_type"] for issue in issues}
    assert "duplicate_event_identity" in issue_types
    assert "causal_self_reference" in issue_types
    assert all(
        issue["severity"] == "hard"
        for issue in issues
        if issue["issue_type"] in issue_types
    )


def test_v11_backfills_legacy_identity_columns_and_replays_state(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(
        tmp_path, username="identity-migration"
    )
    memory = MemoryService(database)
    version_id = _accept_version(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        label="legacy-identity",
    )
    legacy_delta = _knowledge_delta(
        summary="林岚收到父亲署名的新信。",
        character_name="林岚",
        fact="父亲的新信在三天前寄出",
        state="knows",
    )
    legacy_payload = legacy_delta.model_dump(mode="json")
    legacy_payload["events"] = [
        {
            "summary": "林岚收到父亲署名的新信。",
            "participants": ["林岚"],
            "location": "林岚住处",
            "story_time": "第一日晚",
            "causes": [],
            "effects": ["林岚决定返回雾港"],
            "evidence": "她拆开信封。",
        }
    ]
    legacy_delta = StoryDelta.model_validate(legacy_payload)
    _project(
        memory,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=version_id,
        delta=legacy_delta,
    )

    with database.connection() as connection:
        connection.execute("DELETE FROM memory_identity_aliases")
        connection.execute("DELETE FROM memory_identities")
        connection.execute(
            """
            UPDATE story_events
            SET event_identity_id=NULL, event_key=''
            """
        )
        connection.execute(
            """
            UPDATE character_knowledge
            SET character_identity_id=NULL, fact_identity_id=NULL,
                fact_key=''
            """
        )
        connection.execute(
            """
            UPDATE story_state_snapshots
            SET state_json='{"schema_version":2}'
            """
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version=11"
        )
        connection.commit()

    database.initialize()
    with database.connection() as connection:
        event = connection.execute(
            """
            SELECT event_key, event_identity_id FROM story_events
            WHERE record_status='canon'
            """
        ).fetchone()
        knowledge = connection.execute(
            """
            SELECT character_identity_id, fact_identity_id, fact_key
            FROM character_knowledge WHERE record_status='canon'
            """
        ).fetchone()
        migration = connection.execute(
            "SELECT name FROM schema_migrations WHERE version=11"
        ).fetchone()
    assert event["event_key"] == "林岚收到父亲署名的新信。"
    assert event["event_identity_id"]
    assert knowledge["character_identity_id"]
    assert knowledge["fact_identity_id"]
    assert knowledge["fact_key"] == "父亲的新信在三天前寄出"
    assert migration["name"] == "memory_identity_and_causality_v11"
    dashboard = ContinuityService(database).get_dashboard(
        user_id=user_id, project_id=project_id
    )
    assert dashboard["state"]["schema_version"] == 3
    assert dashboard["counts"]["events"] == 1


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="测试叙枢",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "web.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_thinking=False,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=5_000,
        deepseek_connect_timeout_seconds=1,
        deepseek_read_timeout_seconds=1,
        deepseek_max_retries=0,
        worker_poll_seconds=0.01,
    )


def test_continuity_page_is_owned_and_explains_empty_state(tmp_path: Path):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        database = application.state.database
        user_id = database.create_user(
            "web-continuity", hash_password("password-123")
        )
        database.create_novel_project(
            user_id=user_id,
            project_id="web-continuity-project",
            title="空白状态账本",
            genre="悬疑",
            premise="测试连续性页面。",
            world_setting="",
            style_guide="",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
        )
        login_page = client.get("/login")
        csrf = login_page.text.split('name="csrf" value="', 1)[1].split(
            '"', 1
        )[0]
        client.post(
            "/login",
            data={
                "username": "web-continuity",
                "password": "password-123",
                "csrf": csrf,
            },
        )
        page = client.get(
            "/novels/web-continuity-project/continuity"
        )
        assert page.status_code == 200
        assert "正史状态账本" in page.text
        assert "还没有可回放的章节状态" in page.text
        assert client.get("/novels/not-owned/continuity").status_code == 404


def test_lifecycle_pages_render_plot_update_text(tmp_path: Path):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        database = application.state.database
        user_id = database.create_user(
            "web-lifecycle", hash_password("password-123")
        )
        project_id = "web-lifecycle-project"
        chapter_id = "web-lifecycle-chapter"
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title="生命周期模板验收",
            genre="悬疑",
            premise="验证剧情线更新能被正常渲染。",
            world_setting="",
            style_guide="",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
        )
        content_path = tmp_path / "web-lifecycle" / "chapter.txt"
        content_path.parent.mkdir(parents=True)
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title="第一章",
            outline="建立剧情线与伏笔。",
            key_points="父亲失踪之谜",
            content_path=content_path,
        )
        version_id = _accept_version(
            database,
            tmp_path=tmp_path,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            label="lifecycle-template",
        )
        delta_id = _project(
            MemoryService(database),
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
            delta=_lifecycle_delta(
                summary="林岚开始追查父亲失踪。",
                knowledge_fact="蓝玻璃钥匙可能打开旧港密室",
                knowledge_state="knows",
                plot_action="opened",
                hook_action="setup",
            ),
        )

        login_page = client.get("/login")
        csrf = login_page.text.split('name="csrf" value="', 1)[1].split(
            '"', 1
        )[0]
        client.post(
            "/login",
            data={
                "username": "web-lifecycle",
                "password": "password-123",
                "csrf": csrf,
            },
        )

        continuity_page = client.get(
            f"/novels/{project_id}/continuity"
        )
        delta_page = client.get(f"/story-deltas/{delta_id}")
        expected_update = "本章对失踪之谜执行 opened。"
        assert expected_update in continuity_page.text
        assert expected_update in delta_page.text
        assert "&lt;built-in method update" not in continuity_page.text
        assert "&lt;built-in method update" not in delta_page.text


def test_memory_identity_rule_web_workflow(tmp_path: Path):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        database = application.state.database
        user_id = database.create_user(
            "web-identity", hash_password("password-123")
        )
        project_id = "web-identity-project"
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title="归一规则验收",
            genre="悬疑",
            premise="验证作者可控制记忆归一。",
            world_setting="",
            style_guide="",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
        )
        login_page = client.get("/login")
        csrf = login_page.text.split('name="csrf" value="', 1)[1].split(
            '"', 1
        )[0]
        client.post(
            "/login",
            data={
                "username": "web-identity",
                "password": "password-123",
                "csrf": csrf,
            },
        )
        page = client.get(f"/novels/{project_id}/continuity")
        page_csrf = page.text.split('name="csrf" value="', 1)[1].split(
            '"', 1
        )[0]
        response = client.post(
            f"/novels/{project_id}/memory-identities",
            data={
                "identity_type": "fact",
                "canonical_text": "寄信人住在北塔",
                "aliases": "来信者居于北塔\n北塔是寄信人住处",
                "csrf": page_csrf,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        saved_page = client.get(response.headers["location"])
        assert "归一规则已保存" in saved_page.text
        assert "寄信人住在北塔" in saved_page.text
        assert "来信者居于北塔" in saved_page.text

        with database.connection() as connection:
            identity_id = str(
                connection.execute(
                    """
                    SELECT id FROM memory_identities
                    WHERE project_id=? AND identity_type='fact'
                      AND source='author'
                    """,
                    (project_id,),
                ).fetchone()["id"]
            )
        remove_csrf = saved_page.text.split(
            'name="csrf" value="', 1
        )[1].split('"', 1)[0]
        response = client.post(
            f"/novels/{project_id}/memory-identities/"
            f"{identity_id}/delete",
            data={"csrf": remove_csrf},
            follow_redirects=False,
        )
        assert response.status_code == 303
        removed_page = client.get(response.headers["location"])
        assert "归一规则已移除" in removed_page.text
        assert "来信者居于北塔" not in removed_page.text
