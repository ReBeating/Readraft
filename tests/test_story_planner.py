from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.db import SCHEMA, Database, utc_now
from app.main import create_app
from app.migrations import MIGRATIONS
from app.security import hash_password
from app.story_plan_suggestion_service import StoryPlanSuggestionService
from app.story_planner import DeepSeekStoryPlanner, MockStoryPlanner
from app.story_planner_schema import StoryPlanProposalSet
from app.story_planning_schema import PlannedStoryArc, StoryBlueprint
from app.story_planning_service import StoryPlanningService


def _settings(tmp_path: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        app_name="novelAI 测试",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        deepseek_api_key=api_key,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
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


def _blueprint(question: str) -> StoryBlueprint:
    return StoryBlueprint.model_validate(
        {
            "central_question": question,
            "protagonist_goal": "林岚要找到父亲并确认来信来源。",
            "core_conflict": "林岚追查证据，档案部门持续抹除记录。",
            "stakes": "每次推进都会让一名证人失去公开作证的机会。",
            "opening_state": "林岚相信父亲已经死亡并拒绝回港。",
            "ending_state": "林岚公开证据并接受父亲主动失踪的选择。",
            "major_turns": ["新信迫使她回港", "她发现自己的证词被篡改"],
            "must_payoffs": ["解释全部来信的投递方式"],
            "forbidden_shortcuts": ["不能让新角色口述全部真相"],
            "author_notes": "",
        }
    )


def _arc(title: str, payoff: str) -> PlannedStoryArc:
    return PlannedStoryArc.model_validate(
        {
            "arc_type": "main",
            "title": title,
            "dramatic_question": "林岚能否查明真相？",
            "promise": "持续给出可核查的证据与行动后果。",
            "start_state": "只有一封来源不明的信。",
            "target_payoff": payoff,
            "involved_characters": ["林岚"],
            "planned_turns": ["确认邮戳异常", "旧解释被推翻"],
            "lifecycle_status": "planned",
            "priority": 5,
            "author_notes": "",
        }
    )


def _make_project(
    tmp_path: Path,
    *,
    database: Database,
    user_id: int,
    project_id: str = "story-plan-project",
) -> str:
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚收到失踪父亲署名的新信。",
        world_setting="当代海港。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="逐层揭开一条可核对的现实谜团。",
        target_audience="偏好现实悬疑的成年读者。",
        core_appeal="证据链与人物选择同步推进。",
        ending_constraint="父亲失踪必须得到完整因果解释。",
        planning_horizon=20,
    )
    chapter_id = f"{project_id}-chapter"
    chapter_dir = tmp_path / project_id / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚核对来信并决定回港。",
        key_points="确认邮戳异常",
        content_path=content_path,
    )
    return chapter_id


def _complete_with_mock(
    service: StoryPlanSuggestionService, suggestion_id: str
) -> StoryPlanProposalSet:
    item = service.claim_next_suggestion()
    assert item and item["id"] == suggestion_id

    async def generate():
        return await MockStoryPlanner().propose(
            context=item["context_snapshot"],
            mode=item["planning_mode"],
            instruction=item["instruction"],
            provider_user_id="test-user",
        )

    response = asyncio.run(generate())
    assert service.complete_suggestion(
        suggestion_id=suggestion_id,
        claim_token=item["claim_token"],
        result=response.result,
        raw_response=response.raw_response,
        provider=response.provider,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
    return response.result


def test_story_plan_schema_requires_three_complete_distinct_options():
    async def generate():
        return await MockStoryPlanner().propose(
            context={
                "project": {
                    "title": "雾港来信",
                    "premise": "林岚收到一封不可能出现的信。",
                },
                "characters": [{"name": "林岚"}],
            },
            mode="create",
            instruction="",
            provider_user_id="schema-test",
        )

    result = asyncio.run(generate()).result
    assert len(result.options) == 3
    assert all(
        any(arc.arc_type == "main" for arc in option.plot_arcs)
        for option in result.options
    )

    too_few = result.model_dump(mode="json")
    too_few["options"] = too_few["options"][:2]
    with pytest.raises(ValidationError):
        StoryPlanProposalSet.model_validate(too_few)

    duplicate = result.model_dump(mode="json")
    duplicate["options"][1]["label"] = duplicate["options"][0]["label"]
    with pytest.raises(ValidationError, match="不同名称"):
        StoryPlanProposalSet.model_validate(duplicate)

    no_main = result.model_dump(mode="json")
    for arc in no_main["options"][0]["plot_arcs"]:
        arc["arc_type"] = "subplot"
    with pytest.raises(ValidationError, match="main"):
        StoryPlanProposalSet.model_validate(no_main)

    bad_reference = result.model_dump(mode="json")
    bad_reference["options"][0]["volume_sketches"][0][
        "arc_titles"
    ] = ["不存在的剧情线"]
    with pytest.raises(ValidationError, match="精确剧情线名称"):
        StoryPlanProposalSet.model_validate(bad_reference)

    resolved_arc = result.model_dump(mode="json")
    resolved_arc["options"][0]["plot_arcs"][0][
        "lifecycle_status"
    ] = "resolved"
    with pytest.raises(ValidationError, match="planned 或 active"):
        StoryPlanProposalSet.model_validate(resolved_arc)


def test_deepseek_story_planner_retries_invalid_json_and_sends_safe_context(
    tmp_path: Path,
):
    async def run():
        mock_response = await MockStoryPlanner().propose(
            context={
                "project": {"title": "雾港来信", "premise": "一封来信"},
                "characters": [{"name": "林岚"}],
            },
            mode="refine",
            instruction="避免中段重复调查",
            provider_user_id="mock",
        )
        planner = DeepSeekStoryPlanner(
            _settings(tmp_path, api_key="sk-test-secret")
        )
        payloads = []

        async def fake_post(payload):
            payloads.append(payload)
            content = (
                "{}"
                if len(payloads) == 1
                else mock_response.result.model_dump_json()
            )
            return {
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                },
            }

        planner._analyzer._post = fake_post
        try:
            response = await planner.propose(
                context={
                    "context_policy": {
                        "drafts_excluded": True,
                        "plan_source": "author_confirmed_plan_only",
                    },
                    "project": {
                        "title": "雾港来信",
                        "ending_constraint": "必须解释所有来信",
                    },
                    "confirmed_story_blueprint": {
                        "central_question": "父亲为何继续寄信？"
                    },
                    "canonical_memory": {
                        "story_facts": [
                            {"subject_name": "林岚", "predicate": "已回港"}
                        ]
                    },
                },
                mode="refine",
                instruction="避免中段重复调查",
                provider_user_id="stable-user",
            )
        finally:
            await planner.close()
        return response, payloads

    response, payloads = asyncio.run(run())
    assert len(response.result.options) == 3
    assert len(payloads) == 2
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "父亲为何继续寄信" in first_prompt
    assert "drafts_excluded" in first_prompt
    assert "已回港" in first_prompt
    assert "避免中段重复调查" in first_prompt
    assert "上一次 JSON 未通过" in payloads[1]["messages"][-1]["content"]
    assert response.input_tokens == 20
    assert response.output_tokens == 40


def test_suggestion_snapshot_apply_merge_and_confirmed_plan_isolation(
    tmp_path: Path,
):
    database = Database(tmp_path / "story-plan.db")
    database.initialize()
    owner_id = database.create_user(
        "story-plan-owner", hash_password("password-123")
    )
    other_id = database.create_user(
        "story-plan-other", hash_password("password-123")
    )
    chapter_id = _make_project(
        tmp_path, database=database, user_id=owner_id
    )
    planning = StoryPlanningService(database)
    confirmed_blueprint_id = planning.save_blueprint(
        user_id=owner_id,
        project_id="story-plan-project",
        blueprint=_blueprint("父亲为何仍在寄信？"),
        confirm=True,
    )
    existing_arc_id = planning.create_arc(
        user_id=owner_id,
        project_id="story-plan-project",
        arc=_arc("潮汐主线", "确认父亲失踪的完整因果。"),
        confirm=True,
    )
    planning.save_blueprint(
        user_id=owner_id,
        project_id="story-plan-project",
        blueprint=_blueprint("这个未确认问题不应进入模型快照"),
        confirm=False,
    )
    planning.update_arc(
        user_id=owner_id,
        project_id="story-plan-project",
        arc_id=existing_arc_id,
        arc=_arc("潮汐主线", "这个未确认回报不应进入模型快照。"),
        confirm=False,
    )

    service = StoryPlanSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=owner_id,
        project_id="story-plan-project",
        planning_mode="refine",
        instruction="强化中段因果",
        provider="mock",
        model="mock-story-planner",
        credential_source="default",
        max_jobs_per_day=50,
    )
    claimed = service.claim_next_suggestion()
    assert claimed["context_snapshot"]["confirmed_story_blueprint"][
        "central_question"
    ] == "父亲为何仍在寄信？"
    assert (
        claimed["context_snapshot"]["confirmed_planned_plot_arcs"][0][
            "target_payoff"
        ]
        == "确认父亲失踪的完整因果。"
    )
    assert "未确认" not in json.dumps(
        claimed["context_snapshot"], ensure_ascii=False
    )
    response = asyncio.run(
        MockStoryPlanner().propose(
            context=claimed["context_snapshot"],
            mode="refine",
            instruction="强化中段因果",
            provider_user_id="owner",
        )
    )
    assert service.complete_suggestion(
        suggestion_id=suggestion_id,
        claim_token=claimed["claim_token"],
        result=response.result,
        raw_response=response.raw_response,
        provider=response.provider,
        model=response.model,
        input_tokens=0,
        output_tokens=0,
    )
    assert service.get_suggestion(
        user_id=other_id, suggestion_id=suggestion_id
    ) is None
    with pytest.raises(ValueError):
        service.apply_suggestion(
            user_id=other_id,
            suggestion_id=suggestion_id,
            option_index=0,
            apply_blueprint=True,
            selected_arc_indices=[0],
        )

    before_apply = planning.get_blueprint(
        user_id=owner_id, project_id="story-plan-project"
    )
    assert before_apply["confirmed_version_id"] == confirmed_blueprint_id
    applied = service.apply_suggestion(
        user_id=owner_id,
        suggestion_id=suggestion_id,
        option_index=0,
        apply_blueprint=True,
        selected_arc_indices=[0],
    )
    assert applied["baseline_changed"] is False
    assert applied["arcs"][0]["arc_id"] == existing_arc_id
    assert applied["arcs"][0]["created"] is False

    after_apply = planning.get_blueprint(
        user_id=owner_id, project_id="story-plan-project"
    )
    assert after_apply["confirmed_version_id"] == confirmed_blueprint_id
    assert after_apply["current"]["source"] == "story_planner"
    assert after_apply["current"]["version_status"] == "draft"
    arcs = planning.list_arcs(
        user_id=owner_id, project_id="story-plan-project"
    )
    assert len(arcs) == 1
    assert arcs[0]["id"] == existing_arc_id
    assert arcs[0]["confirmed"]["target_payoff"] == (
        "确认父亲失踪的完整因果。"
    )
    assert arcs[0]["source"] == "story_planner"

    writing_context = database.get_writing_context(owner_id, chapter_id)
    assert writing_context["story_blueprint"]["central_question"] == (
        "父亲为何仍在寄信？"
    )
    assert writing_context["planned_plot_arcs"][0]["target_payoff"] == (
        "确认父亲失踪的完整因果。"
    )

    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_projects
            SET premise='生成方案之后改变的项目梗概', updated_at=?
            WHERE id='story-plan-project'
            """,
            (utc_now(),),
        )
        connection.commit()
    merged = service.apply_suggestion(
        user_id=owner_id,
        suggestion_id=suggestion_id,
        option_index=1,
        apply_blueprint=False,
        selected_arc_indices=[1],
    )
    assert merged["baseline_changed"] is True
    assert len(merged["arcs"]) == 1
    assert merged["arcs"][0]["created"] is True
    suggestion = service.get_suggestion(
        user_id=owner_id, suggestion_id=suggestion_id
    )
    assert len(suggestion["applications"]) == 2
    assert suggestion["applications"][1]["baseline_changed"] is True


def test_story_plan_queue_recovers_restart_and_counts_daily_limit(
    tmp_path: Path,
):
    database = Database(tmp_path / "recovery.db")
    database.initialize()
    user_id = database.create_user(
        "recovery-owner", hash_password("password-123")
    )
    _make_project(tmp_path, database=database, user_id=user_id)
    service = StoryPlanSuggestionService(database)
    with pytest.raises(ValueError, match="API Key"):
        service.create_suggestion(
            user_id=user_id,
            project_id="story-plan-project",
            planning_mode="create",
            instruction="",
            provider="deepseek",
            model="deepseek-chat",
            credential_source="personal",
        )

    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id="story-plan-project",
        planning_mode="create",
        instruction="",
        provider="mock",
        model="mock-story-planner",
        credential_source="default",
        max_jobs_per_day=50,
    )
    first_claim = service.claim_next_suggestion()
    assert first_claim["id"] == suggestion_id
    database.initialize()
    assert service.get_status(
        user_id=user_id, suggestion_id=suggestion_id
    ) == "queued"
    second_claim = service.claim_next_suggestion()
    assert second_claim["claim_token"] != first_claim["claim_token"]
    assert service.fail_suggestion(
        suggestion_id=suggestion_id,
        claim_token=second_claim["claim_token"],
        error="结构校验失败",
        input_tokens=12,
        output_tokens=34,
    )
    failed = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert failed["status"] == "failed"
    assert failed["error"] == "结构校验失败"
    assert failed["input_tokens"] == 12
    with pytest.raises(ValueError, match="上限"):
        service.create_suggestion(
            user_id=user_id,
            project_id="story-plan-project",
            planning_mode="rethink",
            instruction="",
            provider="mock",
            model="mock-story-planner",
            credential_source="default",
            max_jobs_per_day=1,
        )


def test_v15_migrates_v14_database_and_preserves_existing_data(
    tmp_path: Path,
):
    path = tmp_path / "legacy-v14.db"
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    connection.execute(
        """
        CREATE TABLE schema_migrations(
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied_at = utc_now()
    for migration in MIGRATIONS[:14]:
        connection.execute("BEGIN IMMEDIATE")
        migration.apply(connection, applied_at)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (?, ?, ?)
            """,
            (migration.version, migration.name, applied_at),
        )
        connection.commit()
    connection.execute(
        """
        INSERT INTO users(username, password_hash, created_at)
        VALUES ('legacy-story-planner', 'hash', ?)
        """,
        (applied_at,),
    )
    user_id = int(
        connection.execute("SELECT last_insert_rowid()").fetchone()[0]
    )
    connection.execute(
        """
        INSERT INTO novel_projects(
            id, user_id, title, genre, premise, created_at, updated_at
        ) VALUES ('legacy-story-project', ?, '旧项目', '悬疑',
                  '旧项目资料必须保留。', ?, ?)
        """,
        (user_id, applied_at, applied_at),
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()
    assert database.get_novel_project(
        user_id, "legacy-story-project"
    )["premise"] == "旧项目资料必须保留。"
    with database.connection() as migrated:
        assert migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0] == 26
        assert migrated.execute(
            "SELECT COUNT(*) FROM story_plan_suggestions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM story_plan_suggestion_applications"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_link_suggestions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_branch_adoptions"
        ).fetchone()[0] == 0
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []


def test_story_plan_user_delete_cascades_suggestions_and_applications(
    tmp_path: Path,
):
    database = Database(tmp_path / "cascade.db")
    database.initialize()
    user_id = database.create_user(
        "cascade-owner", hash_password("password-123")
    )
    _make_project(tmp_path, database=database, user_id=user_id)
    service = StoryPlanSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id="story-plan-project",
        planning_mode="create",
        instruction="",
        provider="mock",
        model="mock-story-planner",
        credential_source="default",
    )
    _complete_with_mock(service, suggestion_id)
    service.apply_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
        option_index=0,
        apply_blueprint=True,
        selected_arc_indices=[0],
    )
    with database.connection() as connection:
        connection.execute("DELETE FROM users WHERE id=?", (user_id,))
        connection.commit()
        assert connection.execute(
            "SELECT COUNT(*) FROM story_plan_suggestions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM story_plan_suggestion_applications"
        ).fetchone()[0] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_story_planner_web_flow_applies_only_unconfirmed_drafts(
    tmp_path: Path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "全书方案作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        novel_form = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "雾港来信",
                "genre": "悬疑",
                "premise": "林岚收到失踪父亲署名的新信。",
                "story_promise": "逐层揭开可核对的谜团。",
                "ending_constraint": "必须解释全部来信。",
                "csrf": _csrf(novel_form.text),
            },
            follow_redirects=False,
        )
        project_url = response.headers["location"]
        project_id = project_url.rsplit("/", 1)[-1]
        workspace = client.get(project_url)
        assert "生成三套可比较的全书方案" in workspace.text
        chapter_response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 迟到的信",
                "outline": "林岚核对来信。",
                "key_points": "确认邮戳",
                "csrf": _csrf(workspace.text),
            },
            follow_redirects=False,
        )
        chapter_id = chapter_response.headers["location"].rsplit("/", 1)[-1]

        workspace = client.get(project_url)
        response = client.post(
            f"{project_url}/story-plan-suggestions",
            data={
                "planning_mode": "create",
                "instruction": "中段必须有持续发动机",
                "csrf": _csrf(workspace.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        suggestion_url = response.headers["location"]
        suggestion_id = suggestion_url.rsplit("/", 1)[-1]
        deadline = time.monotonic() + 4
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(
                f"/api/story-plan-suggestions/{suggestion_id}"
            ).json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)
        assert payload["status"] == "completed"

        comparison = client.get(suggestion_url)
        assert "单线持续加压" in comparison.text
        assert "双线镜像碰撞" in comparison.text
        assert "代价反转结构" in comparison.text
        assert "提案不会自动进入正文" in comparison.text
        assert "分卷草图（只读）" in comparison.text
        response = client.post(
            f"{suggestion_url}/apply",
            data={
                "option_index": "0",
                "apply_blueprint": "yes",
                "arc_indices": ["0"],
                "csrf": _csrf(comparison.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "story_plan_applied=true" in response.headers["location"]

        database = application.state.database
        user = database.get_user_by_username("全书方案作者")
        planning = StoryPlanningService(database)
        blueprint = planning.get_blueprint(
            user_id=int(user["id"]), project_id=project_id
        )
        assert blueprint["current"]["source"] == "story_planner"
        assert blueprint["current"]["version_status"] == "draft"
        assert blueprint["confirmed"] is None
        arcs = planning.list_arcs(
            user_id=int(user["id"]), project_id=project_id
        )
        assert len(arcs) == 1
        assert arcs[0]["confirmed"] is None
        assert arcs[0]["source"] == "story_planner"
        writing_context = database.get_writing_context(
            int(user["id"]), chapter_id
        )
        assert writing_context["story_blueprint"] is None
        assert writing_context["planned_plot_arcs"] == []

        workspace_after = client.get(response.headers["location"])
        assert "所选全书方案已写入新的未确认草稿" in workspace_after.text
        assert "草稿，不进入上下文" in workspace_after.text
