from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.causal_branch_adoption_schema import CausalBranchTaskPatch
from app.causal_branch_adoption_service import (
    CausalBranchAdoptionService,
)
from app.causal_branch_planner import (
    DEFAULT_CAUSAL_BRANCH_CONTEXT_BUDGET,
    MockCausalBranchPlanner,
    compile_causal_branch_context,
)
from app.causal_branch_schema import CausalBranchSimulationSet
from app.causal_branch_service import CausalBranchSimulationService
from app.causal_suggestion_planner import MockCausalSuggestionPlanner
from app.causal_suggestion_service import CausalSuggestionService
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.planning_service import PlanningService
from app.security import hash_password
from app.story_plan_suggestion_service import StoryPlanSuggestionService
from app.story_planning_schema import PlannedStoryArc, StoryBlueprint
from app.story_planning_service import StoryPlanningService
from app.story_structure_service import StoryStructureSuggestionService
from app.structure_link_service import StructureLinkService


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="长期因果分支测试",
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
        max_jobs_per_day=50,
    )


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _blueprint() -> StoryBlueprint:
    return StoryBlueprint.model_validate(
        {
            "central_question": "父亲为何在失踪后继续寄信？",
            "protagonist_goal": "找到父亲并确认来信投递链。",
            "core_conflict": "林岚追查证据，档案部门持续抹除记录。",
            "stakes": "每次取得证据都会让知情者承担公开风险。",
            "opening_state": "林岚相信父亲已经死亡。",
            "ending_state": "林岚公开证据并理解父亲的选择。",
            "major_turns": [
                "新信迫使林岚回港",
                "档案记录暴露第二名经手人",
                "父亲从受害者变成主动参与者",
            ],
            "must_payoffs": [
                "解释来信的时间与投递方式",
                "让林岚亲自决定是否公开证据",
            ],
            "forbidden_shortcuts": ["不能让新角色口述全部真相"],
            "author_notes": "因果必须由人物行动推动。",
        }
    )


def _arc(title: str, arc_type: str, priority: int) -> PlannedStoryArc:
    return PlannedStoryArc.model_validate(
        {
            "arc_type": arc_type,
            "title": title,
            "dramatic_question": f"{title}最终如何改变林岚的选择？",
            "promise": f"持续给出与{title}有关的可验证变化。",
            "start_state": "只有一项来源不明的线索。",
            "target_payoff": f"用人物行动兑现{title}。",
            "involved_characters": ["林岚", "档案员"],
            "planned_turns": ["确认异常", "旧解释被行动后果推翻"],
            "lifecycle_status": "planned",
            "priority": priority,
        }
    )


def _project(
    tmp_path: Path,
    *,
    database: Database | None = None,
    username: str = "long-causal-author",
) -> tuple[Database, int, str, list[str]]:
    database = database or Database(tmp_path / f"{username}.db")
    database.initialize()
    existing = database.get_user_by_username(username)
    user_id = (
        int(existing["id"])
        if existing
        else database.create_user(
            username,
            hash_password("password-123"),
        )
    )
    project_id = f"{username}-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚追查失踪父亲署名的新信。",
        world_setting="当代海港，档案调阅需要实名授权。",
        style_guide="克制具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        planning_horizon=20,
        story_promise="每个关键揭示都能追溯到此前行动。",
        ending_constraint="必须解释全部来信的投递链。",
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="调查者",
        traits="谨慎、执拗",
        background="因父亲失踪离开雾港。",
        character_arc="从只想证明父亲无辜，到愿意承担公开证据的代价。",
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="档案员",
        role="关系线关键人物",
        traits="克制、害怕失去职位",
        background="知道旧档案系统的维护方式。",
        character_arc="从隐瞒风险到主动留下可核查记录。",
    )
    story = StoryPlanningService(database)
    story.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc("父亲失踪主线", "main", 5),
        confirm=True,
    )
    story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc("档案员关系线", "relationship", 3),
        confirm=True,
    )
    planning = PlanningService(database)
    volume_id = planning.create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第一卷 投递链",
        goal="锁定维护投递链的人。",
        start_state="只有一封新信。",
        end_state="确认投递链仍在运行。",
        major_conflict="档案与证词互相否定。",
        payoff="解释第一层投递方式。",
    )
    chapter_ids: list[str] = []
    for position in range(1, 13):
        chapter_id = f"{project_id}-chapter-{position}"
        chapter_dir = tmp_path / project_id / chapter_id
        chapter_dir.mkdir(parents=True, exist_ok=True)
        content_path = chapter_dir / "content.txt"
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=f"第{position}章 线索推进",
            outline="人物核查记录并让调查状态发生变化。",
            key_points="核查设备日志\n采取不可撤销的下一步行动",
            content_path=content_path,
            volume_id=volume_id,
        )
        role = (
            "reversal"
            if position in {4, 8}
            else "payoff"
            if position in {6, 12}
            else "escalation"
            if position > 1
            else "setup"
        )
        arc_title = (
            "父亲失踪主线"
            if position <= 6 or position in {9, 12}
            else "档案员关系线"
        )
        with database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_chapters
                SET skeleton_role=?, skeleton_arc_titles_json=?,
                    skeleton_ending_hook=?,
                    skeleton_application_id=?
                WHERE id=?
                """,
                (
                    role,
                    json.dumps([arc_title], ensure_ascii=False),
                    "新的行动后果迫使人物继续追查。",
                    "window-a" if position <= 6 else "window-b",
                    chapter_id,
                ),
            )
            connection.commit()
        chapter_ids.append(chapter_id)
    return database, user_id, project_id, chapter_ids


def _complete_source(
    database: Database,
    *,
    user_id: int,
    project_id: str,
) -> tuple[str, int]:
    service = CausalSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_limit=12,
        instruction="比较跨线反转的不同前因。",
        provider="mock",
        model="mock-causality-reviewer",
        credential_source="default",
    )
    claimed = service.claim_next_suggestion()
    assert claimed and claimed["id"] == suggestion_id
    response = asyncio.run(
        MockCausalSuggestionPlanner().propose(
            context=claimed["context_snapshot"],
            instruction=str(claimed["instruction"]),
            provider_user_id="test-user",
        )
    )
    assert response.result.proposals
    assert service.complete_suggestion(
        suggestion_id=suggestion_id,
        claim_token=str(claimed["claim_token"]),
        result=response.result,
        raw_response=response.raw_response,
        provider=response.provider,
        model=response.model,
        input_tokens=0,
        output_tokens=0,
    )
    proposal_index = next(
        index
        for index, proposal in enumerate(response.result.proposals)
        if proposal.target_chapter_id
        != f"{project_id}-chapter-12"
    )
    return suggestion_id, proposal_index


def _complete_simulation(
    service: CausalBranchSimulationService,
    simulation_id: str,
) -> CausalBranchSimulationSet:
    claimed = service.claim_next_simulation()
    assert claimed and claimed["id"] == simulation_id
    response = asyncio.run(
        MockCausalBranchPlanner().simulate(
            context=claimed["context_snapshot"],
            instruction=str(claimed["instruction"]),
            provider_user_id="test-user",
        )
    )
    assert service.complete_simulation(
        simulation_id=simulation_id,
        claim_token=str(claimed["claim_token"]),
        result=response.result,
        raw_response=response.raw_response,
        provider=response.provider,
        model=response.model,
        input_tokens=0,
        output_tokens=0,
    )
    return response.result


def _accept_source(
    database: Database,
    *,
    user_id: int,
    suggestion_id: str,
    proposal_index: int,
) -> dict:
    service = CausalSuggestionService(database)
    suggestion = service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )
    proposal = suggestion["result"]["proposals"][proposal_index]
    return service.accept_proposal(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        cause_text=proposal["cause_text"],
        effect_text=proposal["effect_text"],
        author_note="作者确认会按长期分支补齐人物行动与信息渠道。",
        comparison_confirmed=True,
        semantic_review_confirmed=True,
        semantic_override_reason="作者已核对冲突证据并决定保留这条解释。",
    )


def _completed_branch_fixture(
    tmp_path: Path,
    *,
    username: str,
):
    database, user_id, project_id, chapters = _project(
        tmp_path,
        username=username,
    )
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    branch_service = CausalBranchSimulationService(database)
    simulation_id = branch_service.create_simulation(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        horizon_chapter_count=12,
        instruction="重点检查人物知情和阶段回报。",
        provider="mock",
        model="mock-long-horizon-causal-simulator",
        credential_source="default",
    )
    _complete_simulation(branch_service, simulation_id)
    return (
        database,
        user_id,
        project_id,
        chapters,
        suggestion_id,
        proposal_index,
        simulation_id,
    )


def _structure_snapshot(
    database: Database,
    project_id: str,
) -> dict:
    with database.connection() as connection:
        chapters = [
            tuple(row)
            for row in connection.execute(
                """
                SELECT id, position, title, outline, key_points,
                       skeleton_role, skeleton_arc_titles_json,
                       skeleton_ending_hook, skeleton_application_id,
                       canonical_version_id
                FROM novel_chapters
                WHERE project_id=?
                ORDER BY position
                """,
                (project_id,),
            ).fetchall()
        ]
        return {
            "chapters": chapters,
            "task_cards": connection.execute(
                """
                SELECT COUNT(*) FROM novel_chapter_plans
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()[0],
            "causal_links": connection.execute(
                """
                SELECT COUNT(*) FROM novel_chapter_causal_links
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()[0],
        }


def test_mock_long_horizon_simulation_is_exactly_three_and_read_only(
    tmp_path: Path,
):
    database, user_id, project_id, _chapters = _project(tmp_path)
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    before = _structure_snapshot(database, project_id)
    service = CausalBranchSimulationService(database)
    simulation_id = service.create_simulation(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        horizon_chapter_count=12,
        instruction="重点检查人物何时得知行动后果。",
        provider="mock",
        model="mock-long-horizon-causal-simulator",
        credential_source="default",
    )
    result = _complete_simulation(service, simulation_id)
    assert [branch.branch_key for branch in result.branches] == [
        "minimal_change",
        "distributed_consequences",
        "stress_test",
    ]
    assert all(len(branch.chapter_impacts) >= 3 for branch in result.branches)
    assert all(branch.knowledge_transfers for branch in result.branches)
    assert all(branch.payoff_impacts for branch in result.branches)
    assert _structure_snapshot(database, project_id) == before

    saved = service.get_simulation(
        user_id=user_id,
        simulation_id=simulation_id,
    )
    assert saved
    assert saved["baseline_changed"] is False
    assert len(saved["result"]["branches"]) == 3
    assert (
        saved["result"]["branches"][0]["branch_label"]
        == "最小改动"
    )


def test_simulation_schema_rejects_unknown_chapters_and_identical_branches(
    tmp_path: Path,
):
    database, user_id, project_id, _chapters = _project(
        tmp_path,
        username="branch-schema-author",
    )
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    service = CausalBranchSimulationService(database)
    simulation_id = service.create_simulation(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        horizon_chapter_count=12,
        instruction="",
        provider="mock",
        model="mock-long-horizon-causal-simulator",
        credential_source="default",
    )
    claimed = service.claim_next_simulation()
    assert claimed and claimed["id"] == simulation_id
    response = asyncio.run(
        MockCausalBranchPlanner().simulate(
            context=claimed["context_snapshot"],
            instruction="",
            provider_user_id="test-user",
        )
    )
    payload = response.result.model_dump(mode="json")
    payload["branches"][0]["chapter_impacts"][0][
        "chapter_id"
    ] = "outside-frozen-window"
    broken = CausalBranchSimulationSet.model_validate(payload)
    with pytest.raises(ValueError, match="冻结范围之外"):
        broken.ensure_context_compatible(claimed["context_snapshot"])

    duplicate = response.result.model_dump(mode="json")
    duplicate["branches"][1]["premise"] = duplicate["branches"][0][
        "premise"
    ]
    with pytest.raises(ValueError, match="分支前提"):
        CausalBranchSimulationSet.model_validate(duplicate)


def test_simulation_creation_enforces_pending_and_current_source(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _project(
        tmp_path,
        username="branch-gates-author",
    )
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    service = CausalBranchSimulationService(database)
    other_id = database.create_user(
        "branch-other-author",
        hash_password("password-123"),
    )
    assert service.get_simulation(
        user_id=other_id,
        simulation_id="missing",
    ) is None
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_chapters
            SET skeleton_ending_hook='作者已经改变后续承接'
            WHERE id=?
            """,
            (chapters[1],),
        )
        connection.commit()
    with pytest.raises(ValueError, match="已经变化"):
        service.create_simulation(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=proposal_index,
            horizon_chapter_count=12,
            instruction="",
            provider="mock",
            model="mock-long-horizon-causal-simulator",
            credential_source="default",
        )

    fresh_suggestion_id, fresh_proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    CausalSuggestionService(database).dismiss_proposal(
        user_id=user_id,
        suggestion_id=fresh_suggestion_id,
        proposal_index=fresh_proposal_index,
    )
    with pytest.raises(ValueError, match="待审候选"):
        service.create_simulation(
            user_id=user_id,
            suggestion_id=fresh_suggestion_id,
            proposal_index=fresh_proposal_index,
            horizon_chapter_count=12,
            instruction="",
            provider="mock",
            model="mock-long-horizon-causal-simulator",
            credential_source="default",
        )


def test_active_simulation_blocks_other_long_range_planners(
    tmp_path: Path,
):
    database, user_id, project_id, _chapters = _project(
        tmp_path,
        username="branch-task-lock-author",
    )
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    CausalBranchSimulationService(database).create_simulation(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        horizon_chapter_count=12,
        instruction="",
        provider="mock",
        model="mock-long-horizon-causal-simulator",
        credential_source="default",
    )
    with pytest.raises(ValueError, match="已有一个 AI 任务"):
        StoryPlanSuggestionService(database).create_suggestion(
            user_id=user_id,
            project_id=project_id,
            planning_mode="refine",
            instruction="",
            provider="mock",
            model="mock-story-planner",
            credential_source="default",
        )
    with pytest.raises(ValueError, match="已有一个 AI 任务"):
        StoryStructureSuggestionService(
            database,
            tmp_path / "novels",
        ).create_suggestion(
            user_id=user_id,
            project_id=project_id,
            chapter_count=10,
            instruction="",
            provider="mock",
            model="mock-story-structure-planner",
            credential_source="default",
        )
    with pytest.raises(ValueError, match="已有一个 AI 任务"):
        CausalSuggestionService(database).create_suggestion(
            user_id=user_id,
            project_id=project_id,
            chapter_limit=12,
            instruction="new",
            provider="mock",
            model="mock-causality-reviewer",
            credential_source="default",
        )


def test_branch_context_is_bounded_and_keeps_every_future_chapter(
    tmp_path: Path,
):
    database, user_id, project_id, _chapters = _project(
        tmp_path,
        username="branch-context-author",
    )
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    service = CausalBranchSimulationService(database)
    simulation_id = service.create_simulation(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        horizon_chapter_count=12,
        instruction="",
        provider="mock",
        model="mock-long-horizon-causal-simulator",
        credential_source="default",
    )
    claimed = service.claim_next_simulation()
    assert claimed and claimed["id"] == simulation_id
    context = claimed["context_snapshot"]
    compiled = compile_causal_branch_context(context)
    assert (
        len(json.dumps(compiled, ensure_ascii=False))
        <= DEFAULT_CAUSAL_BRANCH_CONTEXT_BUDGET
    )
    assert {
        item["id"] for item in compiled["future_chapters"]
    } == {
        item["id"] for item in context["future_chapters"]
    }
    assert (
        compiled["selected_proposal"]["target_chapter_id"]
        == context["selected_proposal"]["target_chapter_id"]
    )


def test_long_horizon_web_flow_renders_three_read_only_branches(
    tmp_path: Path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "网页长期因果作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        database, user_id, project_id, _chapters = _project(
            tmp_path,
            database=application.state.database,
            username="网页长期因果作者",
        )
        workbench = client.get(
            f"/novels/{project_id}/workbench"
            "?view=archive&archive_tab=creative&settings_tab=structure"
        )
        response = client.post(
            f"/novels/{project_id}/causal-link-suggestions",
            data={
                "chapter_limit": "12",
                "instruction": "重点检查跨线反转。",
                "csrf": _csrf(workbench.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        suggestion_url = response.headers["location"]
        suggestion_id = suggestion_url.rsplit("/", 1)[-1]
        for _ in range(120):
            status_response = client.get(
                f"/api/causal-link-suggestions/{suggestion_id}"
            )
            if status_response.json()["terminal"]:
                break
            time.sleep(0.02)
        page = client.get(suggestion_url)
        assert "沿这条候选推演未来三种分支" in page.text
        suggestion = CausalSuggestionService(database).get_suggestion(
            user_id=user_id,
            suggestion_id=suggestion_id,
        )
        proposal_index = next(
            proposal["proposal_index"]
            for proposal in suggestion["result"]["proposals"]
            if proposal["target_chapter"]["position"] < 12
        )
        before = _structure_snapshot(database, project_id)
        response = client.post(
            f"/causal-link-suggestions/{suggestion_id}/proposals/"
            f"{proposal_index}/branch-simulations",
            data={
                "horizon_chapter_count": "12",
                "instruction": "重点检查林岚的知情渠道。",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        simulation_url = response.headers["location"]
        simulation_id = simulation_url.rsplit("/", 1)[-1]
        for _ in range(120):
            status_response = client.get(
                f"/api/causal-branch-simulations/{simulation_id}"
            )
            if status_response.json()["terminal"]:
                break
            time.sleep(0.02)
        result_page = client.get(simulation_url)
        assert result_page.status_code == 200
        assert "三种分支如何不同" in result_page.text
        assert "只补必要桥接" in result_page.text
        assert "让后果跨章扩散" in result_page.text
        assert "把候选推到断裂点" in result_page.text
        assert "这是作者决策前的沙盘" in result_page.text
        assert "确认并应用这个分支" not in result_page.text
        assert StructureLinkService(database).list_links(
            user_id=user_id,
            project_id=project_id,
        ) == []
        assert _structure_snapshot(database, project_id) == before


def test_branch_adoption_requires_accepted_link_and_creation_is_read_only(
    tmp_path: Path,
):
    (
        database,
        user_id,
        project_id,
        _chapters,
        suggestion_id,
        proposal_index,
        simulation_id,
    ) = _completed_branch_fixture(
        tmp_path,
        username="branch-adoption-gate-author",
    )
    service = CausalBranchAdoptionService(database)
    with pytest.raises(ValueError, match="先返回原因果候选"):
        service.create_adoption(
            user_id=user_id,
            simulation_id=simulation_id,
            branch_key="minimal_change",
            meaning_confirmed=True,
        )
    _accept_source(
        database,
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
    )
    before = _structure_snapshot(database, project_id)
    adoption_id = service.create_adoption(
        user_id=user_id,
        simulation_id=simulation_id,
        branch_key="minimal_change",
        meaning_confirmed=True,
    )
    assert _structure_snapshot(database, project_id) == before
    adoption = service.get_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
    )
    assert adoption
    assert adoption["status"] == "draft"
    assert len(adoption["items"]) >= 3
    assert adoption["pending_count"] == len(adoption["items"])
    assert adoption["baseline_changed"] is False
    assert all(item["patch"]["must_happen"] for item in adoption["items"])
    assert service.create_adoption(
        user_id=user_id,
        simulation_id=simulation_id,
        branch_key="minimal_change",
        meaning_confirmed=True,
    ) == adoption_id


def test_branch_adoption_applies_only_reviewed_items_and_reverts_safely(
    tmp_path: Path,
):
    (
        database,
        user_id,
        project_id,
        _chapters,
        suggestion_id,
        proposal_index,
        simulation_id,
    ) = _completed_branch_fixture(
        tmp_path,
        username="branch-adoption-apply-author",
    )
    _accept_source(
        database,
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
    )
    service = CausalBranchAdoptionService(database)
    adoption_id = service.create_adoption(
        user_id=user_id,
        simulation_id=simulation_id,
        branch_key="distributed_consequences",
        meaning_confirmed=True,
    )
    adoption = service.get_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
    )
    original = _structure_snapshot(database, project_id)
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    for index, item in enumerate(adoption["items"]):
        decision = "accepted" if index < 2 else "rejected"
        if decision == "accepted":
            accepted_ids.append(str(item["chapter_id"]))
        else:
            rejected_ids.append(str(item["chapter_id"]))
        patch = CausalBranchTaskPatch.model_validate(item["patch"])
        service.review_item(
            user_id=user_id,
            adoption_id=adoption_id,
            item_id=str(item["id"]),
            decision=decision,
            patch=patch,
            author_note=f"作者逐章决定：{decision}",
        )
    result = service.apply_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
        author_confirmed=True,
    )
    assert result["applied_item_count"] == 2
    with database.connection() as connection:
        accepted_rows = connection.execute(
            f"""
            SELECT ch.id, ch.key_points, plan.status, plan.source,
                   plan.must_happen_json
            FROM novel_chapters ch
            JOIN novel_chapter_plans plan ON plan.chapter_id=ch.id
            WHERE ch.id IN ({','.join('?' for _ in accepted_ids)})
            ORDER BY ch.position
            """,
            tuple(accepted_ids),
        ).fetchall()
        assert len(accepted_rows) == 2
        assert all(row["status"] == "draft" for row in accepted_rows)
        assert all(
            row["source"] == "branch_adoption" for row in accepted_rows
        )
        assert all(
            json.loads(row["must_happen_json"]) for row in accepted_rows
        )
        rejected_plan_count = connection.execute(
            f"""
            SELECT COUNT(*) FROM novel_chapter_plans
            WHERE chapter_id IN ({','.join('?' for _ in rejected_ids)})
            """,
            tuple(rejected_ids),
        ).fetchone()[0]
        assert rejected_plan_count == 0
        assert connection.execute(
            """
            SELECT COUNT(*) FROM novel_chapter_versions
            WHERE chapter_id IN (
                SELECT id FROM novel_chapters WHERE project_id=?
            )
            """,
            (project_id,),
        ).fetchone()[0] == 0
    applied = service.get_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
    )
    assert applied["status"] == "applied"
    assert applied["accepted_item_count"] == 2
    assert applied["can_revert"] is True
    service.revert_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
    )
    assert _structure_snapshot(database, project_id) == original
    reverted = service.get_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
    )
    assert reverted["status"] == "reverted"


def test_branch_adoption_rejects_new_plot_threads_and_stale_baseline(
    tmp_path: Path,
):
    (
        database,
        user_id,
        _project_id,
        _chapters,
        suggestion_id,
        proposal_index,
        simulation_id,
    ) = _completed_branch_fixture(
        tmp_path,
        username="branch-adoption-stale-author",
    )
    _accept_source(
        database,
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
    )
    service = CausalBranchAdoptionService(database)
    adoption_id = service.create_adoption(
        user_id=user_id,
        simulation_id=simulation_id,
        branch_key="stress_test",
        meaning_confirmed=True,
    )
    adoption = service.get_adoption(
        user_id=user_id,
        adoption_id=adoption_id,
    )
    first = adoption["items"][0]
    invalid_patch = CausalBranchTaskPatch(
        **{
            **first["patch"],
            "plot_threads": ["不存在的剧情线"],
        }
    )
    with pytest.raises(ValueError, match="推演之外的剧情线"):
        service.review_item(
            user_id=user_id,
            adoption_id=adoption_id,
            item_id=str(first["id"]),
            decision="accepted",
            patch=invalid_patch,
            author_note="",
        )
    for index, item in enumerate(adoption["items"]):
        service.review_item(
            user_id=user_id,
            adoption_id=adoption_id,
            item_id=str(item["id"]),
            decision="accepted" if index == 0 else "rejected",
            patch=CausalBranchTaskPatch.model_validate(item["patch"]),
            author_note="",
        )
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_chapters SET key_points=key_points || ?
            WHERE id=?
            """,
            ("\n作者后来新增的关键事件", first["chapter_id"]),
        )
        connection.commit()
    with pytest.raises(ValueError, match="清单建立后发生了变化"):
        service.apply_adoption(
            user_id=user_id,
            adoption_id=adoption_id,
            author_confirmed=True,
        )


def test_branch_adoption_web_flow_reviews_then_updates_task_card_drafts(
    tmp_path: Path,
):
    application = create_app(_settings(tmp_path))
    database, user_id, project_id, _chapters = _project(
        tmp_path,
        database=application.state.database,
        username="网页分支落地作者",
    )
    suggestion_id, proposal_index = _complete_source(
        database,
        user_id=user_id,
        project_id=project_id,
    )
    branch_service = CausalBranchSimulationService(database)
    simulation_id = branch_service.create_simulation(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        horizon_chapter_count=12,
        instruction="",
        provider="mock",
        model="mock-long-horizon-causal-simulator",
        credential_source="default",
    )
    _complete_simulation(branch_service, simulation_id)
    _accept_source(
        database,
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
    )
    with TestClient(application) as client:
        login = client.get("/login")
        response = client.post(
            "/login",
            data={
                "username": "网页分支落地作者",
                "password": "password-123",
                "csrf": _csrf(login.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        simulation_url = (
            f"/causal-branch-simulations/{simulation_id}"
        )
        simulation_page = client.get(simulation_url)
        assert "建立逐章待审清单" in simulation_page.text
        simulation_form = re.search(
            rf'action="{re.escape(simulation_url)}/branches/'
            r'minimal_change/adoptions".*?'
            r'name="csrf" value="([^"]+)"',
            simulation_page.text,
            re.DOTALL,
        )
        assert simulation_form
        assert simulation_form.group(1) == _csrf(simulation_page.text)
        response = client.post(
            f"{simulation_url}/branches/minimal_change/adoptions",
            data={
                "meaning_confirmed": "true",
                "csrf": _csrf(simulation_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        adoption_url = response.headers["location"]
        adoption_id = adoption_url.split("?", 1)[0].rsplit("/", 1)[-1]
        page = client.get(adoption_url)
        assert page.status_code == 200
        assert "逐项决定，而不是整包接受" in page.text
        assert "目前尚未修改任何章节" in page.text
        service = CausalBranchAdoptionService(database)
        adoption = service.get_adoption(
            user_id=user_id,
            adoption_id=adoption_id,
        )
        for index, item in enumerate(adoption["items"]):
            patch = item["patch"]
            data = {
                "decision": (
                    "accepted" if index == 0 else "rejected"
                ),
                "must_happen": "\n".join(patch["must_happen"]),
                "foreshadow_setup": "\n".join(
                    patch["foreshadow_setup"]
                ),
                "foreshadow_payoff": "\n".join(
                    patch["foreshadow_payoff"]
                ),
                "author_note": "网页逐项审核。",
                "csrf": _csrf(page.text),
            }
            if patch["plot_threads"]:
                data["plot_threads"] = patch["plot_threads"]
            response = client.post(
                f"/causal-branch-adoptions/{adoption_id}"
                f"/items/{item['id']}",
                data=data,
                follow_redirects=False,
            )
            assert response.status_code == 303
            page = client.get(
                f"/causal-branch-adoptions/{adoption_id}"
            )
        assert "只应用上面明确采用的 1 项" in page.text
        apply_form = re.search(
            rf'action="/causal-branch-adoptions/{adoption_id}/apply"'
            r'.*?name="csrf" value="([^"]+)"',
            page.text,
            re.DOTALL,
        )
        assert apply_form
        assert apply_form.group(1) == _csrf(page.text)
        response = client.post(
            f"/causal-branch-adoptions/{adoption_id}/apply",
            data={
                "author_confirmed": "true",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        applied_page = client.get(response.headers["location"])
        assert "已写入 1 个未来章节" in applied_page.text
        assert "打开已更新的任务卡" in applied_page.text
        assert "安全撤销这次任务卡落地" in applied_page.text
