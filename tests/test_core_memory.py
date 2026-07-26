from __future__ import annotations

from pathlib import Path

from app.context_compiler import (
    build_writing_context_snapshot,
    compile_canonical_memory,
)
from app.db import SCHEMA, Database
from app.memory_schema import StoryDelta
from app.memory_service import MemoryService
from app.security import hash_password
from app.writing import build_writing_messages


def _build_project(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user("memory-writer", hash_password("password-123"))
    project_id = "p" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者返回故乡追查一封不可能出现的信。",
        world_setting="当代海港小城。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="调查记者",
        traits="冷静、执拗",
        background="父亲十年前失踪",
        character_arc="直面家庭秘密",
    )
    chapter_ids = []
    for position in range(1, 4):
        chapter_id = f"{position}" * 32
        chapter_dir = tmp_path / chapter_id
        chapter_dir.mkdir()
        content_path = chapter_dir / "content.txt"
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=f"第{position}章",
            outline=f"第{position}章大纲",
            key_points="关键点",
            content_path=content_path,
        )
        chapter_ids.append(chapter_id)
    return database, user_id, project_id, chapter_ids


def _save_candidate(
    database: Database,
    *,
    tmp_path: Path,
    user_id: int,
    project_id: str,
    chapter_id: str,
    name: str,
    content: str,
) -> str:
    path = tmp_path / chapter_id / "versions" / f"{name}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=path,
        char_count=len(content),
    )
    assert version_id
    return version_id


def _add_chapter(
    database: Database,
    *,
    tmp_path: Path,
    user_id: int,
    project_id: str,
    position: int,
    outline: str,
) -> str:
    chapter_id = f"chapter-{project_id[:8]}-{position}"
    chapter_dir = tmp_path / chapter_id
    chapter_dir.mkdir()
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title=f"第{position}章",
        outline=outline,
        key_points=outline,
        content_path=content_path,
    )
    return chapter_id


def _story_delta() -> StoryDelta:
    return StoryDelta.model_validate(
        {
            "chapter_summary": "林岚收到父亲署名的新信，决定返回雾港。",
            "keywords": ["来信", "雾港"],
            "unresolved_questions": ["寄信人是谁？"],
            "character_changes": [
                {
                    "character_name": "林岚",
                    "aspect": "goal",
                    "before": "留在外地工作",
                    "after": "返回雾港调查父亲失踪",
                    "evidence": "她把返程票放在信封旁。",
                }
            ],
            "relationship_changes": [],
            "location_changes": [
                {
                    "subject_name": "林岚",
                    "from_location": "外地",
                    "to_location": "前往雾港",
                    "evidence": "她买下当晚回雾港的车票。",
                }
            ],
            "item_changes": [
                {
                    "item_name": "父亲来信",
                    "action": "acquired",
                    "from_holder": None,
                    "to_holder": "林岚",
                    "state": "邮戳日期为三天前",
                    "evidence": "信封上盖着三天前的邮戳。",
                }
            ],
            "knowledge_changes": [
                {
                    "character_name": "林岚",
                    "fact": "父亲名下的新信在三天前寄出",
                    "state": "knows",
                    "learned_via": "查看信封邮戳",
                    "evidence": "她反复确认邮戳日期。",
                }
            ],
            "plot_thread_changes": [
                {
                    "thread_name": "父亲失踪之谜",
                    "thread_type": "main",
                    "action": "opened",
                    "update": "十年前失踪者似乎在近日寄出了信。",
                    "promise": "解释信件来源与父亲下落",
                    "target_payoff": "揭开失踪真相",
                    "evidence": "署名与笔迹都属于父亲。",
                }
            ],
            "foreshadowing_changes": [
                {
                    "hook_name": "三天前的邮戳",
                    "action": "setup",
                    "description": "失踪十年的父亲寄出近期信件。",
                    "intended_payoff": "揭示寄信者和寄出方式",
                    "evidence": "邮戳清晰显示三天前。",
                }
            ],
            "events": [
                {
                    "summary": "林岚收到并拆开父亲署名的来信。",
                    "participants": ["林岚"],
                    "location": "林岚住处",
                    "story_time": "当晚",
                    "causes": [],
                    "effects": ["林岚决定返回雾港"],
                    "evidence": "她读完信后购买返程票。",
                }
            ],
            "time_advance": None,
        }
    )


def test_candidate_canon_and_story_delta_projection(tmp_path: Path):
    database, user_id, project_id, chapters = _build_project(tmp_path)
    memory = MemoryService(database)

    first_version = _save_candidate(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        name="first",
        content="林岚拆开了那封信，买下回雾港的车票。",
    )
    chapter = database.get_novel_chapter(user_id, project_id, chapters[0])
    assert chapter["working_version_id"] == first_version
    assert chapter["canonical_version_id"] is None

    accepted = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=first_version,
        override_reason="测试作者明确接受未审计的短篇候选稿",
    )
    assert accepted and accepted["changed"]
    chapter = database.get_novel_chapter(user_id, project_id, chapters[0])
    assert chapter["canonical_version_id"] == first_version

    delta_id = memory.create_proposal(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=first_version,
        payload=_story_delta(),
    )
    assert memory.get_delta(user_id=user_id, delta_id=delta_id)["status"] == (
        "proposed"
    )
    projection = memory.accept_delta(user_id=user_id, delta_id=delta_id)
    assert projection == {
        "delta_id": delta_id,
        "projected": True,
        "already_projected": False,
        "memory_count": 1,
        "event_count": 1,
        "fact_count": 3,
        "knowledge_count": 1,
        "plot_thread_count": 1,
        "foreshadowing_count": 1,
    }

    chapter_memory = memory.get_chapter_memory(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
    )
    assert chapter_memory["summary"].startswith("林岚收到")
    assert chapter_memory["keywords"] == ["来信", "雾港"]
    with database.connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM story_facts WHERE fact_status='canon'"
        ).fetchone()[0] == 3
        knowledge = connection.execute(
            """
            SELECT character_id, knowledge_state
            FROM character_knowledge WHERE record_status='canon'
            """
        ).fetchone()
        assert knowledge["character_id"] is not None
        assert knowledge["knowledge_state"] == "knows"

    unaccepted_candidate = _save_candidate(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        name="unaccepted",
        content="这份候选稿不能进入第二章上下文。",
    )
    assert unaccepted_candidate != first_version
    next_context = database.get_writing_context(user_id, chapters[1])
    assert Path(next_context["previous_chapter"]["content_path"]).name == (
        "first.txt"
    )
    assert next_context["canonical_memory"]["recent_chapters"][0][
        "summary"
    ].startswith("林岚收到")
    assert next_context["canonical_memory"]["character_knowledge"][0][
        "knowledge_state"
    ] == "knows"
    messages = build_writing_messages(
        context=next_context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    prompt = str(messages[1]["content"])
    assert "<canonical_story_memory>" in prompt
    assert "父亲名下的新信在三天前寄出" in prompt
    assert "author_confirmed_canon_only" in prompt


def test_fts_retrieves_only_relevant_older_canon_and_tracks_retraction(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(tmp_path)
    memory = MemoryService(database)
    for position in range(4, 9):
        outline = (
            "林岚重新检查蓝玻璃钥匙与旧港档案。"
            if position == 7
            else f"第{position}章继续普通调查。"
        )
        chapters.append(
            _add_chapter(
                database,
                tmp_path=tmp_path,
                user_id=user_id,
                project_id=project_id,
                position=position,
                outline=outline,
            )
        )

    def project_memory(
        *,
        owner_id: int,
        owner_project_id: str,
        chapter_id: str,
        position: int,
        marker: str,
    ) -> str:
        payload = _story_delta().model_dump(mode="json")
        payload["chapter_summary"] = (
            f"第{position}章确认线索“{marker}”并保存到正史。"
        )
        payload["keywords"] = [marker]
        payload["item_changes"][0]["item_name"] = marker
        payload["item_changes"][0]["evidence"] = f"林岚收好{marker}。"
        payload["events"][0]["summary"] = f"林岚核对{marker}。"
        payload["events"][0]["evidence"] = f"她确认{marker}确实存在。"
        version_id = _save_candidate(
            database,
            tmp_path=tmp_path,
            user_id=owner_id,
            project_id=owner_project_id,
            chapter_id=chapter_id,
            name=f"canon-{position}",
            content=f"林岚确认了{marker}。",
        )
        database.accept_chapter_version(
            user_id=owner_id,
            project_id=owner_project_id,
            chapter_id=chapter_id,
            version_id=version_id,
            override_reason="测试作者明确接受记忆检索所需的正史版本",
        )
        delta_id = memory.create_proposal(
            user_id=owner_id,
            project_id=owner_project_id,
            chapter_id=chapter_id,
            version_id=version_id,
            payload=payload,
        )
        memory.accept_delta(user_id=owner_id, delta_id=delta_id)
        return version_id

    for position, chapter_id in enumerate(chapters[:6], start=1):
        project_memory(
            owner_id=user_id,
            owner_project_id=project_id,
            chapter_id=chapter_id,
            position=position,
            marker=(
                "蓝玻璃钥匙"
                if position == 1
                else f"普通调查线索{position}"
            ),
        )

    # A canonical future chapter must never enter chapter 7's context.
    project_memory(
        owner_id=user_id,
        owner_project_id=project_id,
        chapter_id=chapters[7],
        position=8,
        marker="蓝玻璃钥匙",
    )

    # The same phrase in another user's project must also stay isolated.
    other_user = database.create_user(
        "other-memory-owner", hash_password("password-123")
    )
    other_project = "other-project-for-memory-search"
    database.create_novel_project(
        user_id=other_user,
        project_id=other_project,
        title="另一部作品",
        genre="悬疑",
        premise="不应进入当前用户上下文。",
        world_setting="另一座城市。",
        style_guide="简洁。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    other_chapter = _add_chapter(
        database,
        tmp_path=tmp_path,
        user_id=other_user,
        project_id=other_project,
        position=1,
        outline="蓝玻璃钥匙属于另一部作品。",
    )
    project_memory(
        owner_id=other_user,
        owner_project_id=other_project,
        chapter_id=other_chapter,
        position=1,
        marker="蓝玻璃钥匙",
    )

    context = database.get_writing_context(
        user_id, chapters[6], None, "蓝玻璃钥匙"
    )
    raw_memory = context["canonical_memory"]
    assert [item["position"] for item in raw_memory["recent_chapters"]] == [
        6,
        5,
        4,
        3,
        2,
    ]
    assert raw_memory["retrieval"]["engine"].startswith("sqlite_fts5")
    assert raw_memory["retrieval"]["matched_count"] > 0
    assert "蓝玻璃钥匙" in raw_memory["retrieval"]["query_concepts"]
    assert raw_memory["retrieved_memory"]
    assert {
        item["source_chapter_position"]
        for item in raw_memory["retrieved_memory"]
    } == {1}
    assert all(
        item["source_chapter_id"] == chapters[0]
        for item in raw_memory["retrieved_memory"]
    )
    assert any(
        "蓝玻璃钥匙" in item["excerpt"]
        or "蓝玻璃钥匙" in item["keywords"]
        for item in raw_memory["retrieved_memory"]
    )

    compiled = compile_canonical_memory(raw_memory)
    snapshot = build_writing_context_snapshot(
        context={**context, "canonical_memory": compiled},
        operation="draft",
        instruction="继续追查蓝玻璃钥匙",
        current_content="",
        previous_content="",
    )
    assert snapshot["canonical_memory"]["retrieval"]["matched_count"] > 0
    assert "蓝玻璃钥匙" in snapshot["canonical_memory"]["retrieval"][
        "query_concepts"
    ]
    assert snapshot["canonical_memory"]["retrieved_memory"]
    prompt = str(
        build_writing_messages(
            context={**context, "canonical_memory": compiled},
            operation="draft",
            instruction="继续追查蓝玻璃钥匙",
            current_content="",
            previous_content="",
        )[1]["content"]
    )
    assert "蓝玻璃钥匙" in prompt

    # Re-run only migration 8 to prove it backfills already-confirmed memory.
    with database.connection() as connection:
        connection.execute("DROP TABLE story_memory_fts")
        connection.execute("DROP TABLE story_memory_search_documents")
        connection.execute("DELETE FROM schema_migrations WHERE version=8")
        connection.commit()
    database.initialize()
    with database.connection() as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM story_memory_search_documents
            WHERE project_id=? AND chapter_id=?
            """,
            (project_id, chapters[0]),
        ).fetchone()[0] == 8
        connection.execute(
            "INSERT INTO story_memory_fts(story_memory_fts) "
            "VALUES ('integrity-check')"
        )

    replacement = _save_candidate(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        name="replacement-without-key",
        content="第一章改为与钥匙无关的正史。",
    )
    database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=replacement,
        override_reason="测试作者明确替换旧正史并撤回对应搜索文档",
    )
    with database.connection() as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM story_memory_search_documents
            WHERE chapter_id=?
            """,
            (chapters[0],),
        ).fetchone()[0] == 0
    refreshed = database.get_writing_context(
        user_id, chapters[6], None, "蓝玻璃钥匙"
    )
    assert refreshed["canonical_memory"]["retrieved_memory"] == []


def test_replacing_old_canon_retracts_memory_and_marks_downstream(
    tmp_path: Path,
):
    database, user_id, project_id, chapters = _build_project(tmp_path)
    memory = MemoryService(database)

    first_version = _save_candidate(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        name="first",
        content="第一版",
    )
    database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=first_version,
        override_reason="测试作者明确接受未审计的第一版正文",
    )
    delta_id = memory.create_proposal(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=first_version,
        payload=_story_delta(),
    )
    memory.accept_delta(user_id=user_id, delta_id=delta_id)

    for chapter_id in chapters[1:]:
        version = _save_candidate(
            database,
            tmp_path=tmp_path,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            name="canon",
            content="后续章节",
        )
        database.accept_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version,
            override_reason="测试作者明确接受未审计的后续章节",
        )

    replacement = _save_candidate(
        database,
        tmp_path=tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        name="replacement",
        content="修改后的第一章",
    )
    result = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=replacement,
        override_reason="测试作者明确接受未审计的替换版本",
    )
    assert result["downstream_count"] == 2
    assert database.get_novel_chapter(
        user_id, project_id, chapters[1]
    )["needs_recheck"] == 1
    assert (
        memory.get_chapter_memory(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapters[0],
        )
        is None
    )
    assert memory.get_delta(user_id=user_id, delta_id=delta_id)["status"] == (
        "superseded"
    )

    restored = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        version_id=first_version,
        override_reason="测试作者明确恢复已知的历史版本正文",
    )
    assert restored["changed"]
    assert database.get_novel_chapter(
        user_id, project_id, chapters[0]
    )["canonical_version_id"] == first_version


def test_migration_preserves_existing_account_key_and_version(tmp_path: Path):
    database_path = tmp_path / "legacy.db"
    database = Database(database_path)
    content_path = tmp_path / "content.txt"
    content_path.write_text("旧章节正文", encoding="utf-8")
    with database.connection() as connection:
        connection.executescript(SCHEMA)
        connection.execute(
            """
            INSERT INTO users(id, username, password_hash, created_at)
            VALUES (7, 'legacy', 'password-hash', '2026-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO api_credentials(
                user_id, provider, encrypted_key, key_hint, model,
                created_at, updated_at
            ) VALUES (7, 'deepseek', 'encrypted-secret', 'sk-••••1234',
                      'deepseek-chat',
                      '2026-01-01T00:00:00+00:00',
                      '2026-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO novel_projects(
                id, user_id, title, genre, premise, world_setting,
                style_guide, point_of_view, target_chapter_chars,
                created_at, updated_at
            ) VALUES ('legacy-project', 7, '旧作品', '悬疑', '旧梗概', '',
                      '', '第三人称限知', 3000,
                      '2026-01-01T00:00:00+00:00',
                      '2026-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO novel_chapters(
                id, project_id, position, title, outline, key_points,
                status, content_path, char_count, created_at, updated_at
            ) VALUES ('legacy-chapter', 'legacy-project', 1, '旧章', '', '',
                      'draft', ?, 5,
                      '2026-01-01T00:00:00+00:00',
                      '2026-01-01T00:00:00+00:00')
            """,
            (str(content_path),),
        )
        connection.execute(
            """
            INSERT INTO novel_chapter_versions(
                id, chapter_id, kind, content_path, char_count, created_at
            ) VALUES ('legacy-version', 'legacy-chapter', 'manual', ?, 5,
                      '2026-01-01T00:00:00+00:00')
            """,
            (str(content_path),),
        )
        connection.commit()

    database.initialize()

    assert database.get_api_credential(7)["encrypted_key"] == "encrypted-secret"
    chapter = database.get_novel_chapter(
        7, "legacy-project", "legacy-chapter"
    )
    assert chapter["canonical_version_id"] == "legacy-version"
    assert chapter["working_version_id"] == "legacy-version"
    with database.connection() as connection:
        migrations = connection.execute(
            """
            SELECT version, name FROM schema_migrations
            ORDER BY version
            """
        ).fetchall()
        assert [dict(row) for row in migrations] == [
            {"version": 1, "name": "core_memory_v1"},
            {"version": 2, "name": "planning_v2"},
            {"version": 3, "name": "quality_gate_v3"},
            {"version": 4, "name": "style_editor_v4"},
                {"version": 5, "name": "reader_decisions_v5"},
                {"version": 6, "name": "technique_library_v6"},
                {"version": 7, "name": "scene_workbench_v7"},
                {"version": 8, "name": "memory_search_v8"},
                {"version": 9, "name": "continuity_replay_v9"},
                {"version": 10, "name": "continuity_lifecycle_v10"},
                {
                    "version": 11,
                    "name": "memory_identity_and_causality_v11",
                },
                {
                    "version": 12,
                    "name": "voice_profile_learning_v12",
                },
                {
                    "version": 13,
                    "name": "manual_edit_preference_learning_v13",
                },
                    {
                        "version": 14,
                        "name": "story_blueprint_v14",
                    },
                    {
                        "version": 15,
                        "name": "story_planner_suggestions_v15",
                    },
                    {
                        "version": 16,
                        "name": "story_structure_planner_v16",
                    },
                    {
                        "version": 17,
                        "name": "chapter_causal_links_v17",
                    },
                    {
                        "version": 18,
                        "name": "causal_link_suggestions_v18",
                    },
                    {
                        "version": 19,
                        "name": "causal_branch_simulations_v19",
                    },
                    {
                        "version": 20,
                        "name": "causal_branch_adoptions_v20",
                    },
                        {
                            "version": 21,
                            "name": "scene_requirement_coverage_v21",
                        },
                        {
                            "version": 22,
                            "name": "editing_preference_aggregation_v22",
                        },
                            {
                                "version": 23,
                                "name": "assistant_chat_v23",
                            },
                            {
                                "version": 24,
                                "name": "workbench_prompts_v24",
                            },
                            {
                                "version": 25,
                                "name": "assistant_agent_tools_v25",
                            },
                            {
                                "version": 26,
                                "name": "assistant_agent_steps_v26",
                            },
                            {
                                "version": 27,
                                "name": "model_base_url_v27",
                            },
                            {
                                "version": 28,
                                "name": "multi_provider_credentials_v28",
                            },
                            {
                                "version": 29,
                                "name": "automatic_reasoning_policy_v29",
                            },
                            {
                                "version": 30,
                                "name": "unified_model_adapter_v30",
                            },
                                {
                                    "version": 31,
                                    "name": "unified_work_library_v31",
                                },
                                    {
                                        "version": 32,
                                        "name": "work_archive_semantics_v32",
                                    },
                                    {
                                        "version": 33,
                                        "name": "repository_versions_v33",
                                    },
                                ]
        assert database.get_api_credential(7)["base_url"] == ""
        assert database.get_api_credential(7)["is_default"] == 1
        assert connection.execute(
            """
            SELECT COUNT(*) FROM novel_chapter_plans
            WHERE chapter_id='legacy-chapter'
            """
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT status FROM novel_chapter_versions
            WHERE id='legacy-version'
            """
        ).fetchone()["status"] == "canonical"
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='table' AND name='story_memory_fts'
            """
        ).fetchone()[0] == 1
