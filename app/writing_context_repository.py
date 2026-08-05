from __future__ import annotations

from typing import Any, Dict, Optional

from .continuity import get_continuity_context
from .json_support import load_json as _load_json
from .memory_identity import (
    expand_identity_terms,
    list_identity_context,
)
from .memory_search import (
    SEARCH_ENGINE,
    SEARCH_SCOPE,
    build_query_concepts,
    build_query_terms,
    search_memory_documents,
)


def get_writing_context(
    database: Any,
    user_id: int,
    chapter_id: str,
    retrieval_hint: str = "",
) -> Optional[Dict[str, Any]]:
    with database.connection() as connection:
        chapter = connection.execute(
            """
            SELECT ch.*, p.title AS project_title, p.genre, p.premise,
                   p.world_setting, p.style_guide, p.ai_instructions,
                   p.point_of_view,
                   p.target_chapter_chars, p.canonical_branch_id,
                   p.story_promise, p.target_audience, p.core_appeal,
                   p.ending_constraint, p.planning_horizon,
                   head.content_path AS active_content_path,
                   head.content_hash AS active_content_hash,
                   v.title AS volume_title, v.goal AS volume_goal,
                   v.start_state AS volume_start_state,
                   v.end_state AS volume_end_state,
                   v.major_conflict AS volume_major_conflict,
                   v.payoff AS volume_payoff
            FROM novel_chapters ch
            JOIN novel_projects p ON p.id=ch.project_id
            LEFT JOIN novel_chapter_versions head
              ON head.id=ch.head_version_id
            LEFT JOIN novel_volumes v ON v.id=ch.volume_id
            WHERE ch.id=? AND p.user_id=?
            """,
            (chapter_id, user_id),
        ).fetchone()
        if not chapter:
            return None
        planned_causal_link_rows = connection.execute(
            """
            SELECT link.id, link.project_id,
                   link.source_chapter_id, link.target_chapter_id,
                   link.relation_type, link.cause_text,
                   link.effect_text, link.author_note, link.status,
                   source.position AS source_position,
                   source.title AS source_title,
                   source.skeleton_arc_titles_json
                       AS source_arc_titles_json,
                   source.head_version_id
                       AS source_head_version_id,
                   target.position AS target_position,
                   target.title AS target_title,
                   target.skeleton_arc_titles_json
                       AS target_arc_titles_json,
                   target.head_version_id
                       AS target_head_version_id
            FROM novel_chapter_causal_links link
            JOIN novel_chapters source
              ON source.id=link.source_chapter_id
            JOIN novel_chapters target
              ON target.id=link.target_chapter_id
            WHERE link.project_id=?
              AND link.status='active'
              AND target.head_version_id IS NULL
              AND (
                link.source_chapter_id=?
                OR link.target_chapter_id=?
              )
            ORDER BY target.position, source.position, link.created_at
            LIMIT 40
            """,
            (chapter["project_id"], chapter_id, chapter_id),
        ).fetchall()
        voice_profile = connection.execute(
            """
            SELECT narration_rules, sentence_rhythm, dialogue_voice,
                   sensory_palette, metaphor_policy, allowed_omissions,
                   preferred_patterns_json, banned_expressions_json,
                   author_notes, status, confirmed_at, updated_at
            FROM novel_voice_profiles
            WHERE project_id=? AND status='confirmed'
            """,
            (chapter["project_id"],),
        ).fetchone()
        editing_preferences = connection.execute(
            """
            SELECT id, category, guidance, applicability, updated_at,
                   source_type, support_count
            FROM (
              SELECT aggregate.id, aggregate.category,
                     aggregate.guidance, aggregate.applicability,
                     aggregate.updated_at,
                     'stable_aggregate' AS source_type,
                     (
                       SELECT COUNT(*)
                       FROM
                         author_editing_preference_aggregate_evidence e
                       WHERE e.aggregate_id=aggregate.id
                         AND e.role='support'
                     ) AS support_count,
                     0 AS source_rank
              FROM author_editing_preference_aggregates aggregate
              WHERE aggregate.project_id=?
                AND aggregate.status='active'

              UNION ALL

              SELECT pref.id, pref.category, pref.guidance,
                     pref.applicability, pref.updated_at,
                     'single_observation' AS source_type,
                     1 AS support_count,
                     1 AS source_rank
              FROM author_editing_preferences pref
              WHERE pref.project_id=? AND pref.status='active'
                AND NOT EXISTS(
                  SELECT 1
                  FROM
                    author_editing_preference_aggregate_evidence e
                  JOIN
                    author_editing_preference_aggregates aggregate
                    ON aggregate.id=e.aggregate_id
                  WHERE e.preference_id=pref.id
                    AND aggregate.status='active'
                )
            )
            ORDER BY source_rank, updated_at DESC, id DESC
            LIMIT 20
            """,
            (chapter["project_id"], chapter["project_id"]),
        ).fetchall()
        story_blueprint = connection.execute(
            """
            SELECT v.*
            FROM novel_story_blueprint_heads h
            JOIN novel_story_blueprint_versions v
                ON v.id=h.confirmed_version_id
            WHERE h.project_id=? AND v.project_id=h.project_id
                AND v.version_status='confirmed'
            """,
            (chapter["project_id"],),
        ).fetchone()
        planned_plot_arcs = connection.execute(
            """
            SELECT a.id, a.position, v.arc_type, v.title,
                   v.dramatic_question, v.promise, v.start_state,
                   v.target_payoff, v.involved_characters_json,
                   v.planned_turns_json, v.lifecycle_status,
                   v.priority
            FROM novel_plot_arcs a
            JOIN novel_plot_arc_versions v
                ON v.id=a.confirmed_version_id
            WHERE a.project_id=? AND v.project_id=a.project_id
                AND v.version_status='confirmed'
            ORDER BY v.priority DESC, a.position
            """,
            (chapter["project_id"],),
        ).fetchall()
        characters = connection.execute(
            """
            SELECT id, name, role, traits, background, character_arc,
                   external_goal, internal_need, central_conflict,
                   hidden_fact AS secret, speech_style, initial_state
            FROM novel_characters
            WHERE project_id=?
            ORDER BY position
            """,
            (chapter["project_id"],),
        ).fetchall()
        confirmed_archive_rules = connection.execute(
            """
            SELECT entry.id, entry.category, entry.title,
                   entry.content, entry.evidence, entry.provenance,
                   entry.updated_at
            FROM work_versions version
            JOIN work_archive_entries entry
              ON entry.work_id=version.work_id
            WHERE version.project_id=?
              AND entry.entry_type='creative_rule'
              AND entry.status='confirmed'
            ORDER BY
                CASE entry.category
                    WHEN 'core' THEN 0
                    WHEN 'world' THEN 1
                    WHEN 'character' THEN 2
                    WHEN 'structure' THEN 3
                    WHEN 'style' THEN 4
                    ELSE 5
                END,
                entry.updated_at DESC
            LIMIT 120
            """,
            (chapter["project_id"],),
        ).fetchall()
        plan = connection.execute(
            """
            SELECT * FROM novel_chapter_plans
            WHERE chapter_id=? AND status='confirmed'
            """,
            (chapter_id,),
        ).fetchone()
        scene_beats = []
        if plan:
            scene_beats = connection.execute(
                """
                SELECT id, position, pov_character, goal, obstacle,
                       action, reveal, conceal, subtext, location,
                       key_items_json, end_state, transition,
                       requirement_refs_json
                FROM novel_scene_beats
                WHERE plan_id=? AND beat_status='active'
                ORDER BY position
                """,
                (plan["id"],),
            ).fetchall()
        previous = connection.execute(
            """
            SELECT ch.id, ch.position, ch.title, ch.outline,
                   v.content_path, v.char_count
            FROM novel_chapters ch
            JOIN novel_chapter_versions v
                ON v.id=ch.head_version_id
            WHERE ch.project_id=? AND ch.position<?
            ORDER BY ch.position DESC LIMIT 1
            """,
            (chapter["project_id"], chapter["position"]),
        ).fetchone()
        recent_memory = connection.execute(
            """
            SELECT m.id AS source_id, ch.id AS chapter_id,
                   ch.position, ch.title, m.summary,
                   m.key_events_json, m.unresolved_questions_json,
                   m.keywords_json
            FROM chapter_memory m
            JOIN novel_chapters ch ON ch.id=m.chapter_id
            WHERE m.project_id=? AND m.branch_id=?
                AND m.record_status='canon' AND ch.position<?
            ORDER BY ch.position DESC
            LIMIT 5
            """,
            (
                chapter["project_id"],
                chapter["canonical_branch_id"],
                chapter["position"],
            ),
        ).fetchall()
        facts = connection.execute(
            """
            SELECT f.id AS source_id,
                   ch.position AS source_chapter_position,
                   ch.title AS source_chapter_title,
                   f.fact_type, f.subject_type, f.subject_name,
                   f.predicate, f.object_json, f.evidence
            FROM story_facts f
            JOIN novel_chapters ch ON ch.id=f.chapter_id
            WHERE f.project_id=? AND f.branch_id=?
                AND f.fact_status='canon' AND ch.position<?
            ORDER BY ch.position DESC, f.created_at DESC
            LIMIT 120
            """,
            (
                chapter["project_id"],
                chapter["canonical_branch_id"],
                chapter["position"],
            ),
        ).fetchall()
        knowledge = connection.execute(
            """
            SELECT k.id AS source_id,
                   ch.position AS source_chapter_position,
                   k.character_name, k.fact_text, k.knowledge_state,
                   k.learned_via, k.evidence
            FROM character_knowledge k
            JOIN novel_chapters ch ON ch.id=k.chapter_id
            WHERE k.project_id=? AND k.branch_id=?
                AND k.record_status='canon' AND ch.position<?
            ORDER BY ch.position DESC, k.created_at DESC
            LIMIT 120
            """,
            (
                chapter["project_id"],
                chapter["canonical_branch_id"],
                chapter["position"],
            ),
        ).fetchall()
        plot_threads = connection.execute(
            """
            SELECT t.id AS source_id,
                   ch.position AS source_chapter_position,
                   t.thread_name, t.thread_type, t.action,
                   t.update_text, t.promise, t.target_payoff, t.evidence
            FROM plot_threads t
            JOIN novel_chapters ch ON ch.id=t.chapter_id
            WHERE t.project_id=? AND t.branch_id=?
                AND t.record_status='canon' AND ch.position<?
            ORDER BY ch.position DESC, t.created_at DESC
            LIMIT 80
            """,
            (
                chapter["project_id"],
                chapter["canonical_branch_id"],
                chapter["position"],
            ),
        ).fetchall()
        hooks = connection.execute(
            """
            SELECT f.id AS source_id,
                   ch.position AS source_chapter_position,
                   f.hook_name, f.action, f.description,
                   f.intended_payoff, f.evidence
            FROM foreshadowing f
            JOIN novel_chapters ch ON ch.id=f.chapter_id
            WHERE f.project_id=? AND f.branch_id=?
                AND f.record_status='canon' AND ch.position<?
            ORDER BY ch.position DESC, f.created_at DESC
            LIMIT 80
            """,
            (
                chapter["project_id"],
                chapter["canonical_branch_id"],
                chapter["position"],
            ),
        ).fetchall()
        retrieval_scenes = []
        for row in scene_beats:
            scene = dict(row)
            scene["key_items"] = _load_json(scene.pop("key_items_json"), [])
            scene["requirement_refs"] = _load_json(
                scene.pop("requirement_refs_json"), []
            )
            retrieval_scenes.append(scene)
        retrieval_task_card = None
        if plan:
            retrieval_task_card = dict(plan)
            for stored, public in (
                ("plot_threads_json", "plot_threads"),
                ("must_happen_json", "must_happen"),
                ("must_preserve_json", "must_preserve"),
                ("forbidden_json", "forbidden"),
                ("foreshadow_setup_json", "foreshadow_setup"),
                ("foreshadow_payoff_json", "foreshadow_payoff"),
            ):
                retrieval_task_card[public] = _load_json(
                    retrieval_task_card.pop(stored), []
                )
            retrieval_task_card["scenes"] = retrieval_scenes
        retrieval_query_terms = build_query_terms(
            chapter=dict(chapter),
            characters=[dict(row) for row in characters],
            task_card=retrieval_task_card,
            scenes=retrieval_scenes,
            focused_scene_id=None,
            retrieval_hint=retrieval_hint,
        )
        retrieval_query_concepts = build_query_concepts(
            chapter=dict(chapter),
            characters=[dict(row) for row in characters],
            task_card=retrieval_task_card,
            scenes=retrieval_scenes,
            focused_scene_id=None,
            retrieval_hint=retrieval_hint,
        )
        retrieval_query_terms = expand_identity_terms(
            connection,
            project_id=str(chapter["project_id"]),
            terms=retrieval_query_terms,
            max_terms=96,
        )
        retrieval_query_concepts = expand_identity_terms(
            connection,
            project_id=str(chapter["project_id"]),
            terms=retrieval_query_concepts,
            max_terms=48,
        )
        memory_identities = list_identity_context(
            connection,
            project_id=str(chapter["project_id"]),
            limit=160,
        )
        retrieved_memory = search_memory_documents(
            connection,
            project_id=str(chapter["project_id"]),
            branch_id=str(chapter["canonical_branch_id"]),
            before_chapter_position=int(chapter["position"]),
            query_terms=retrieval_query_terms,
            query_concepts=retrieval_query_concepts,
            excluded_chapter_ids=[str(row["chapter_id"]) for row in recent_memory],
        )
        continuity_context = get_continuity_context(
            connection,
            project_id=str(chapter["project_id"]),
            branch_id=str(chapter["canonical_branch_id"]),
            before_chapter_position=int(chapter["position"]),
            query_concepts=retrieval_query_concepts,
        )
        technique_rows = connection.execute(
            """
            SELECT tc.id, tc.name, tc.dimension, tc.effect,
                   tc.execution_rule, tc.originality_boundary,
                   b.id AS binding_id, b.scope_type,
                   b.usage_modes_json, b.author_adaptation, b.priority,
                   v.position AS volume_position,
                   v.title AS volume_title,
                   scoped_ch.position AS chapter_position,
                   scoped_ch.title AS chapter_title,
                   sb.position AS scene_position,
                   sb.goal AS scene_goal,
                   scene_ch.position AS scene_chapter_position,
                   scene_ch.title AS scene_chapter_title
            FROM novel_technique_bindings b
            JOIN reference_technique_cards tc ON tc.id=b.technique_id
            LEFT JOIN novel_volumes v ON v.id=b.volume_id
            LEFT JOIN novel_chapters scoped_ch ON scoped_ch.id=b.chapter_id
            LEFT JOIN novel_scene_beats sb ON sb.id=b.scene_beat_id
            LEFT JOIN novel_chapter_plans scene_plan
                ON scene_plan.id=sb.plan_id
            LEFT JOIN novel_chapters scene_ch
                ON scene_ch.id=scene_plan.chapter_id
            WHERE b.project_id=? AND b.status='enabled'
                AND tc.status='active'
                AND (
                    b.scope_type='project'
                    OR (b.scope_type='volume' AND b.volume_id=?)
                    OR (b.scope_type='chapter' AND b.chapter_id=?)
                    OR (
                        b.scope_type='scene'
                        AND scene_plan.chapter_id=?
                        AND sb.beat_status='active'
                    )
                )
            ORDER BY b.priority DESC, b.created_at
            """,
            (
                chapter["project_id"],
                chapter["volume_id"],
                chapter_id,
                chapter_id,
            ),
        ).fetchall()
    recent_items = []
    for row in recent_memory:
        item = dict(row)
        item["key_events"] = _load_json(item.pop("key_events_json"), [])
        item["unresolved_questions"] = _load_json(
            item.pop("unresolved_questions_json"), []
        )
        item["keywords"] = _load_json(item.pop("keywords_json"), [])
        recent_items.append(item)
    fact_items = []
    for row in facts:
        item = dict(row)
        item["object"] = _load_json(item.pop("object_json"), {})
        fact_items.append(item)
    technique_items = []
    for row in technique_rows:
        item = dict(row)
        item["usage_modes"] = _load_json(item.pop("usage_modes_json"), [])
        if item["scope_type"] == "project":
            item["scope_label"] = "全书"
        elif item["scope_type"] == "volume":
            item["scope_label"] = (
                f"第 {item['volume_position']} 卷《{item['volume_title']}》"
            )
        elif item["scope_type"] == "chapter":
            item["scope_label"] = (
                f"第 {item['chapter_position']} 章《{item['chapter_title']}》"
            )
        else:
            item["scope_label"] = (
                f"第 {item['scene_chapter_position']} 章 / "
                f"场景 {item['scene_position']}：{item['scene_goal']}"
            )
        technique_items.append(item)
    task_card = None
    if plan:
        task_card = dict(plan)
        for stored, public in (
            ("plot_threads_json", "plot_threads"),
            ("must_happen_json", "must_happen"),
            ("must_preserve_json", "must_preserve"),
            ("forbidden_json", "forbidden"),
            ("foreshadow_setup_json", "foreshadow_setup"),
            ("foreshadow_payoff_json", "foreshadow_payoff"),
        ):
            task_card[public] = _load_json(task_card.pop(stored), [])
        task_card["scenes"] = []
        for row in scene_beats:
            scene = dict(row)
            scene["key_items"] = _load_json(scene.pop("key_items_json"), [])
            scene["requirement_refs"] = _load_json(
                scene.pop("requirement_refs_json"), []
            )
            task_card["scenes"].append(scene)
    confirmed_voice_profile = None
    if voice_profile:
        confirmed_voice_profile = dict(voice_profile)
        confirmed_voice_profile["preferred_patterns"] = _load_json(
            confirmed_voice_profile.pop("preferred_patterns_json"), []
        )
        confirmed_voice_profile["banned_expressions"] = _load_json(
            confirmed_voice_profile.pop("banned_expressions_json"), []
        )
    confirmed_story_blueprint = None
    if story_blueprint:
        confirmed_story_blueprint = dict(story_blueprint)
        confirmed_story_blueprint["major_turns"] = _load_json(
            confirmed_story_blueprint.pop("major_turns_json"), []
        )
        confirmed_story_blueprint["must_payoffs"] = _load_json(
            confirmed_story_blueprint.pop("must_payoffs_json"), []
        )
        confirmed_story_blueprint["forbidden_shortcuts"] = _load_json(
            confirmed_story_blueprint.pop("forbidden_shortcuts_json"),
            [],
        )
    confirmed_plot_arcs = []
    for row in planned_plot_arcs:
        item = dict(row)
        item["involved_characters"] = _load_json(
            item.pop("involved_characters_json"), []
        )
        item["planned_turns"] = _load_json(item.pop("planned_turns_json"), [])
        confirmed_plot_arcs.append(item)
    planned_causal_links = []
    for row in planned_causal_link_rows:
        item = dict(row)
        source_arcs = _load_json(item.pop("source_arc_titles_json"), [])
        target_arcs = _load_json(item.pop("target_arc_titles_json"), [])
        if not isinstance(source_arcs, list):
            source_arcs = []
        if not isinstance(target_arcs, list):
            target_arcs = []
        source_arcs = [str(value) for value in source_arcs if str(value).strip()]
        target_arcs = [str(value) for value in target_arcs if str(value).strip()]
        item["source_arc_titles"] = source_arcs
        item["target_arc_titles"] = target_arcs
        item["shared_arc_titles"] = sorted(set(source_arcs) & set(target_arcs))
        item["cross_line"] = bool(
            source_arcs and target_arcs and not item["shared_arc_titles"]
        )
        item["source_is_canonical"] = bool(
            item.pop("source_head_version_id", None)
        )
        item["target_is_canonical"] = bool(
            item.pop("target_head_version_id", None)
        )
        planned_causal_links.append(item)
    chapter_item = dict(chapter)
    chapter_item["cache_content_path"] = str(
        chapter_item.get("content_path") or ""
    )
    chapter_item["content_path"] = str(
        chapter_item.pop("active_content_path", "") or ""
    )
    chapter_item["content_hash"] = str(
        chapter_item.pop("active_content_hash", "") or ""
    )
    chapter_item["skeleton_arc_titles"] = _load_json(
        chapter_item.pop("skeleton_arc_titles_json", "[]"), []
    )
    return {
        "chapter": chapter_item,
        "characters": [dict(row) for row in characters],
        "previous_chapter": dict(previous) if previous else None,
        "task_card": task_card,
        "voice_profile": confirmed_voice_profile,
        "story_blueprint": confirmed_story_blueprint,
        "planned_plot_arcs": confirmed_plot_arcs,
        "planned_causal_links": planned_causal_links,
        "confirmed_archive_rules": [dict(row) for row in confirmed_archive_rules],
        "confirmed_editing_preferences": [dict(row) for row in editing_preferences],
        "canonical_memory": {
            "recent_chapters": recent_items,
            "story_facts": fact_items,
            "character_knowledge": [dict(row) for row in knowledge],
            "plot_threads": [dict(row) for row in plot_threads],
            "foreshadowing": [dict(row) for row in hooks],
            "retrieved_memory": retrieved_memory,
            "current_state": continuity_context["current_state"],
            "continuity_issues": continuity_context["continuity_issues"],
            "continuity_replay": continuity_context["continuity_replay"],
            "retrieval": {
                "engine": SEARCH_ENGINE,
                "scope": SEARCH_SCOPE,
                "query_terms": retrieval_query_terms,
                "query_concepts": retrieval_query_concepts,
                "matched_count": len(retrieved_memory),
                "excluded_recent_chapter_count": len(recent_memory),
            },
        },
        "memory_identities": memory_identities,
        "technique_cards": technique_items,
    }
