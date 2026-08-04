from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Optional

from .db import Database, utc_after, utc_now
from .json_support import (
    load_json_dict as _json_dict,
    load_json_list as _json_list,
)
from .preference_extraction import build_edit_sample


EDIT_PREFERENCE_CATEGORIES = {
    "diction",
    "sentence_rhythm",
    "narration_distance",
    "dialogue",
    "emotional_expression",
    "sensory_detail",
    "metaphor",
    "omission",
    "paragraph_structure",
    "other",
}

EDIT_PREFERENCE_STYLE_ISSUES = {
    "diction": {"cliche", "repetition"},
    "sentence_rhythm": {"uniform_rhythm"},
    "narration_distance": {
        "abstract_emotion",
        "over_explanation",
        "unnecessary_summary",
    },
    "dialogue": {"dialogue_convergence"},
    "emotional_expression": {"abstract_emotion"},
    "sensory_detail": {"generic_atmosphere", "non_specific_detail"},
    "metaphor": {"cliche"},
    "omission": {
        "over_explanation",
        "over_complete_paragraph",
        "unnecessary_summary",
    },
    "paragraph_structure": {
        "uniform_rhythm",
        "over_complete_paragraph",
        "repetition",
    },
    "other": set(),
}

EFFECT_WINDOW_DAYS = 90
EFFECT_MAX_VERSIONS_PER_SIDE = 20
EFFECT_MIN_VERSIONS = 3
EFFECT_MIN_CHAPTERS = 3
EFFECT_MIN_CHARS = 6000


def _normalized_text(value: Any) -> str:
    return "".join(
        char.lower()
        for char in str(value or "")
        if char.isalnum()
    )


def _text_similarity(left: Any, right: Any) -> float:
    left_text = _normalized_text(left)
    right_text = _normalized_text(right)
    if not left_text or not right_text:
        return 0.0
    sequence_score = SequenceMatcher(
        None, left_text, right_text
    ).ratio()
    left_pairs = {
        left_text[index : index + 2]
        for index in range(max(1, len(left_text) - 1))
    }
    right_pairs = {
        right_text[index : index + 2]
        for index in range(max(1, len(right_text) - 1))
    }
    union = left_pairs | right_pairs
    pair_score = (
        len(left_pairs & right_pairs) / len(union) if union else 0.0
    )
    return max(sequence_score, pair_score)


def _parse_datetime(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class PreferenceService:
    def __init__(self, database: Database):
        self.database = database

    def create_suggestion(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        source_type: str,
        after_version_id: str,
        expected_scene_beat_id: Optional[str] = None,
        provider: str,
        model: str,
        credential_source: str,
    ) -> str:
        if source_type not in {"chapter", "scene"}:
            raise ValueError("不支持的改稿来源")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        source = self._resolve_source(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            source_type=source_type,
            after_version_id=after_version_id,
        )
        if not source:
            raise ValueError("手工改稿版本不存在")
        if (
            source_type == "scene"
            and expected_scene_beat_id
            and str(source.get("scene_beat_id") or "")
            != expected_scene_beat_id
        ):
            raise ValueError("场景改稿版本与当前场景不一致")
        if (
            str(source["after_created_by"]) != "author"
            or str(source["after_source"]) != "manual"
        ):
            raise ValueError("只能从作者手工保存的改稿版本中学习")
        before_text = self._read_text(str(source["before_content_path"]))
        after_text = self._read_text(str(source["after_content_path"]))
        if not before_text.strip() or not after_text.strip():
            raise ValueError("修改前后正文不能为空")
        before_hash = self._content_hash(before_text)
        after_hash = self._content_hash(after_text)
        if before_hash == after_hash:
            raise ValueError("这次保存没有产生正文变化")
        change_sample = build_edit_sample(before_text, after_text)
        if int(change_sample["changed_char_count"]) < 20:
            raise ValueError("这次改动过小，暂不适合归纳长期偏好")
        if not change_sample["blocks"]:
            raise ValueError("没有形成可供分析的改稿差异")

        suggestion_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._resolve_source_row(
                connection,
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                source_type=source_type,
                after_version_id=after_version_id,
            )
            if (
                not current
                or str(current["before_version_id"])
                != str(source["before_version_id"])
                or str(current["after_created_by"]) != "author"
                or str(current["after_source"]) != "manual"
                or (
                    source_type == "scene"
                    and expected_scene_beat_id
                    and str(current["scene_beat_id"] or "")
                    != expected_scene_beat_id
                )
            ):
                connection.rollback()
                raise ValueError("手工改稿版本已经变化，请刷新后重试")
            if credential_source == "personal":
                credential = connection.execute(
                    """
                    SELECT 1 FROM api_credentials
                    WHERE user_id=? AND provider=?
                    """,
                    (user_id, provider),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "所选模型服务 API Key 或凭据不存在，请重新配置"
                    )
            duplicate = connection.execute(
                """
                SELECT id FROM editing_preference_suggestions
                WHERE project_id=? AND source_type=? AND after_version_id=?
                  AND status IN ('queued', 'running', 'ready', 'applied')
                ORDER BY created_at DESC LIMIT 1
                """,
                (project_id, source_type, after_version_id),
            ).fetchone()
            if duplicate:
                connection.rollback()
                return str(duplicate["id"])
            active = connection.execute(
                """
                SELECT
                  (SELECT id FROM generation_jobs
                   WHERE user_id=? AND status IN ('queued', 'running')
                   LIMIT 1) AS generation_id,
                  (SELECT id FROM voice_profile_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   LIMIT 1) AS voice_id,
                  (SELECT id FROM editing_preference_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   LIMIT 1) AS preference_id
                """,
                (user_id, user_id, user_id),
            ).fetchone()
            if any(active[key] for key in active.keys()):
                connection.rollback()
                raise ValueError(
                    "你已有一个 AI 任务正在排队或运行，请等待其完成"
                )
            connection.execute(
                """
                INSERT INTO editing_preference_suggestions(
                    id, project_id, user_id, chapter_id, source_type,
                    scene_beat_id, before_version_id, after_version_id,
                    before_content_hash, after_content_hash,
                    change_sample_json, changed_char_count,
                    author_change_summary, provider, model,
                    credential_source, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'queued', ?)
                """,
                (
                    suggestion_id,
                    project_id,
                    user_id,
                    chapter_id,
                    source_type,
                    source.get("scene_beat_id"),
                    source["before_version_id"],
                    after_version_id,
                    before_hash,
                    after_hash,
                    json.dumps(change_sample, ensure_ascii=False),
                    int(change_sample["changed_char_count"]),
                    str(source.get("author_change_summary") or ""),
                    provider,
                    model,
                    credential_source,
                    now,
                ),
            )
            connection.commit()
        return suggestion_id

    def list_suggestions(
        self, *, user_id: int, project_id: str, limit: int = 8
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT suggestion.*, ch.title AS chapter_title,
                       ch.position,
                       sb.goal AS scene_goal
                FROM editing_preference_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                JOIN novel_chapters ch ON ch.id=suggestion.chapter_id
                LEFT JOIN novel_scene_beats sb
                    ON sb.id=suggestion.scene_beat_id
                WHERE suggestion.project_id=? AND p.user_id=?
                ORDER BY suggestion.created_at DESC, suggestion.rowid DESC
                LIMIT ?
                """,
                (project_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_suggestion(
        self, *, user_id: int, suggestion_id: str
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT suggestion.*, p.title AS project_title,
                       p.genre, p.point_of_view, p.style_guide,
                       ch.title AS chapter_title, ch.position,
                       sb.goal AS scene_goal
                FROM editing_preference_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                JOIN novel_chapters ch ON ch.id=suggestion.chapter_id
                LEFT JOIN novel_scene_beats sb
                    ON sb.id=suggestion.scene_beat_id
                WHERE suggestion.id=? AND p.user_id=?
                """,
                (suggestion_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["suggestion"] = _json_dict(
            result.pop("suggestion_json")
        )
        result["change_sample"] = _json_dict(
            result.pop("change_sample_json")
        )
        result["applied_preference_ids"] = _json_list(
            result.pop("applied_preference_ids_json")
        )
        return result

    def claim_next_suggestion(self) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次偏好提取租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT suggestion.*, p.title, p.genre, p.point_of_view,
                       p.style_guide, ch.title AS chapter_title,
                       sb.goal AS scene_goal
                FROM editing_preference_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                JOIN novel_chapters ch ON ch.id=suggestion.chapter_id
                LEFT JOIN novel_scene_beats sb
                    ON sb.id=suggestion.scene_beat_id
                WHERE suggestion.status='queued'
                ORDER BY suggestion.created_at
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='running', started_at=?, error=NULL,
                    claim_token=?, lease_expires_at=?
                WHERE id=? AND status='queued'
                """,
                (
                    now,
                    claim_token,
                    utc_after(2 * 60 * 60),
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        result = dict(row)
        result["change_sample"] = _json_dict(
            result.pop("change_sample_json")
        )
        result["claim_token"] = claim_token
        return result

    def load_running_source_pair(
        self, *, suggestion_id: str, claim_token: str
    ) -> tuple[str, str]:
        with self.database.connection() as connection:
            job = connection.execute(
                """
                SELECT * FROM editing_preference_suggestions
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (suggestion_id, claim_token),
            ).fetchone()
            if not job:
                raise ValueError("编辑偏好任务已经失效")
            source = self._resolve_source_row(
                connection,
                user_id=int(job["user_id"]),
                project_id=str(job["project_id"]),
                chapter_id=str(job["chapter_id"]),
                source_type=str(job["source_type"]),
                after_version_id=str(job["after_version_id"]),
            )
        if (
            not source
            or str(source["before_version_id"])
            != str(job["before_version_id"])
        ):
            raise ValueError("改稿来源版本已经不存在")
        before_text = self._read_text(str(source["before_content_path"]))
        after_text = self._read_text(str(source["after_content_path"]))
        if (
            self._content_hash(before_text)
            != str(job["before_content_hash"])
            or self._content_hash(after_text)
            != str(job["after_content_hash"])
        ):
            raise ValueError("改稿来源文件与任务创建时不一致")
        return before_text, after_text

    def complete_suggestion(
        self,
        *,
        suggestion_id: str,
        claim_token: str,
        suggestion: Mapping[str, Any],
        raw_response: str,
        provider: str,
        model: str,
        valid_evidence_count: int,
        dropped_evidence_count: int,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='ready', suggestion_json=?, raw_response=?,
                    provider=?, model=?,
                    valid_evidence_count=?, dropped_evidence_count=?,
                    input_tokens=?, output_tokens=?, finished_at=?,
                    claim_token=NULL, lease_expires_at=NULL, error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    json.dumps(dict(suggestion), ensure_ascii=False),
                    raw_response,
                    provider,
                    model,
                    valid_evidence_count,
                    dropped_evidence_count,
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    suggestion_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def fail_suggestion(
        self,
        *,
        suggestion_id: str,
        claim_token: str,
        error: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='failed', error=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    error[:2000],
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    suggestion_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self, suggestion_id: str, claim_token: str, error: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], suggestion_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def apply_suggestion(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        selections: list[Mapping[str, Any]],
    ) -> Optional[str]:
        if not selections:
            raise ValueError("至少选择一条编辑偏好")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT suggestion.*
                FROM editing_preference_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.id=? AND p.user_id=?
                  AND suggestion.status='ready'
                """,
                (suggestion_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            proposed = _json_dict(row["suggestion_json"]).get(
                "preferences", []
            )
            applied_ids: list[str] = []
            used_indices: set[int] = set()
            for selection in selections:
                index = int(selection["index"])
                if (
                    index in used_indices
                    or index < 0
                    or index >= len(proposed)
                ):
                    connection.rollback()
                    raise ValueError("编辑偏好选择已经失效")
                used_indices.add(index)
                category = str(selection["category"])
                if category not in EDIT_PREFERENCE_CATEGORIES:
                    connection.rollback()
                    raise ValueError("不支持的编辑偏好类别")
                guidance = str(selection["guidance"]).strip()
                applicability = str(selection["applicability"]).strip()
                if not 8 <= len(guidance) <= 600:
                    connection.rollback()
                    raise ValueError("偏好规则需为 8–600 个字符")
                if not 4 <= len(applicability) <= 500:
                    connection.rollback()
                    raise ValueError("适用范围需为 4–500 个字符")
                candidate = proposed[index]
                preference_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO author_editing_preferences(
                        id, project_id, suggestion_id, category, guidance,
                        applicability, before_quote, after_quote, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        preference_id,
                        row["project_id"],
                        suggestion_id,
                        category,
                        guidance,
                        applicability,
                        str(candidate.get("before_quote") or ""),
                        str(candidate.get("after_quote") or ""),
                        now,
                        now,
                    ),
                )
                applied_ids.append(preference_id)
            cursor = connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='applied', applied_preference_ids_json=?,
                    reviewed_at=?
                WHERE id=? AND status='ready'
                """,
                (
                    json.dumps(applied_ids, ensure_ascii=False),
                    now,
                    suggestion_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, row["project_id"]),
            )
            connection.commit()
        return str(row["project_id"])

    def reject_suggestion(
        self, *, user_id: int, suggestion_id: str
    ) -> Optional[str]:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE editing_preference_suggestions
                SET status='rejected', reviewed_at=?
                WHERE id=? AND status='ready' AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=editing_preference_suggestions.project_id
                      AND p.user_id=?
                )
                RETURNING project_id
                """,
                (utc_now(), suggestion_id, user_id),
            )
            row = cursor.fetchone()
            connection.commit()
        return str(row["project_id"]) if row else None

    def list_active_preferences(
        self,
        *,
        user_id: int,
        project_id: str,
        limit: int = 30,
        exclude_aggregated: bool = False,
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT pref.*, suggestion.source_type,
                       suggestion.after_version_id,
                       ch.position AS chapter_position,
                       ch.title AS chapter_title,
                       sb.goal AS scene_goal
                FROM author_editing_preferences pref
                JOIN novel_projects p ON p.id=pref.project_id
                LEFT JOIN editing_preference_suggestions suggestion
                  ON suggestion.id=pref.suggestion_id
                LEFT JOIN novel_chapters ch
                  ON ch.id=suggestion.chapter_id
                LEFT JOIN novel_scene_beats sb
                  ON sb.id=suggestion.scene_beat_id
                WHERE pref.project_id=? AND p.user_id=?
                  AND pref.status='active'
                  AND (
                    ?=0 OR NOT EXISTS(
                      SELECT 1
                      FROM author_editing_preference_aggregate_evidence e
                      JOIN author_editing_preference_aggregates aggregate
                        ON aggregate.id=e.aggregate_id
                      WHERE e.preference_id=pref.id
                        AND aggregate.status='active'
                    )
                  )
                ORDER BY pref.updated_at DESC, pref.rowid DESC
                LIMIT ?
                """,
                (
                    project_id,
                    user_id,
                    1 if exclude_aggregated else 0,
                    limit,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_aggregation_candidates(
        self, *, user_id: int, project_id: str
    ) -> list[dict[str, Any]]:
        preferences = self.list_active_preferences(
            user_id=user_id,
            project_id=project_id,
            limit=120,
            exclude_aggregated=True,
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for preference in preferences:
            grouped.setdefault(
                str(preference["category"]), []
            ).append(preference)

        candidates: list[dict[str, Any]] = []
        for category, items in grouped.items():
            source_keys = {
                (
                    str(item.get("source_type") or ""),
                    str(item.get("after_version_id") or ""),
                )
                for item in items
                if item.get("source_type")
                and item.get("after_version_id")
            }
            if len(source_keys) < 2:
                continue
            visible_items = items[:12]
            pair_scores = []
            for index, left in enumerate(visible_items):
                for right in visible_items[index + 1 :]:
                    pair_scores.append(
                        _text_similarity(
                            (
                                f"{left['guidance']} "
                                f"{left['applicability']}"
                            ),
                            (
                                f"{right['guidance']} "
                                f"{right['applicability']}"
                            ),
                        )
                    )
            similarity = max(pair_scores, default=0.0)
            if similarity >= 0.7:
                relation_hint = "措辞高度相近"
            elif similarity >= 0.45:
                relation_hint = "表达有明显重合"
            else:
                relation_hint = "属于同一类别，需由作者判断关系"
            candidates.append(
                {
                    "category": category,
                    "preferences": visible_items,
                    "source_count": len(source_keys),
                    "similarity_percent": int(
                        round(similarity * 100)
                    ),
                    "relation_hint": relation_hint,
                    "truncated_count": max(
                        0, len(items) - len(visible_items)
                    ),
                    "suggested_guidance": str(
                        visible_items[0]["guidance"]
                    ),
                    "suggested_applicability": str(
                        visible_items[0]["applicability"]
                    ),
                }
            )
        candidates.sort(
            key=lambda item: (
                -int(item["source_count"]),
                -int(item["similarity_percent"]),
                str(item["category"]),
            )
        )
        return candidates

    def get_memory_summary(
        self, *, user_id: int, project_id: str
    ) -> Optional[dict[str, int]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT
                  (
                    SELECT COUNT(*)
                    FROM author_editing_preference_aggregates aggregate
                    WHERE aggregate.project_id=p.id
                      AND aggregate.status='active'
                  ) AS stable_count,
                  (
                    SELECT COUNT(*)
                    FROM author_editing_preferences pref
                    WHERE pref.project_id=p.id
                      AND pref.status='active'
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
                  ) AS single_count,
                  (
                    SELECT COUNT(*)
                    FROM
                      author_editing_preference_aggregate_evidence evidence
                    JOIN author_editing_preference_aggregates aggregate
                      ON aggregate.id=evidence.aggregate_id
                    WHERE aggregate.project_id=p.id
                      AND aggregate.status='active'
                      AND evidence.role='conflict'
                  ) AS conflict_count
                FROM novel_projects p
                WHERE p.id=? AND p.user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
        if not row:
            return None
        return {
            "stable_count": int(row["stable_count"] or 0),
            "single_count": int(row["single_count"] or 0),
            "conflict_count": int(row["conflict_count"] or 0),
        }

    def create_aggregate(
        self,
        *,
        user_id: int,
        project_id: str,
        category: str,
        guidance: str,
        applicability: str,
        author_note: str,
        support_preference_ids: list[str],
        conflict_preference_ids: list[str],
    ) -> str:
        category = str(category or "").strip()
        guidance = str(guidance or "").strip()
        applicability = str(applicability or "").strip()
        author_note = str(author_note or "").strip()
        support_ids = list(
            dict.fromkeys(
                str(value).strip()
                for value in support_preference_ids
                if str(value).strip()
            )
        )
        conflict_ids = list(
            dict.fromkeys(
                str(value).strip()
                for value in conflict_preference_ids
                if str(value).strip()
            )
        )
        if category not in EDIT_PREFERENCE_CATEGORIES:
            raise ValueError("不支持的编辑偏好类别")
        if not 8 <= len(guidance) <= 600:
            raise ValueError("稳定偏好规则需为 8–600 个字符")
        if not 4 <= len(applicability) <= 500:
            raise ValueError("适用范围需为 4–500 个字符")
        if len(author_note) > 1000:
            raise ValueError("作者说明不能超过 1000 个字符")
        if len(support_ids) < 2:
            raise ValueError("至少选择两次独立改稿作为支持证据")
        if len(support_ids) + len(conflict_ids) > 12:
            raise ValueError("一次最多归并 12 条改稿观察")
        if set(support_ids) & set(conflict_ids):
            raise ValueError("同一条改稿观察不能同时支持又冲突")

        aggregate_id = uuid.uuid4().hex
        now = utc_now()
        all_ids = support_ids + conflict_ids
        placeholders = ",".join("?" for _ in all_ids)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            project = connection.execute(
                """
                SELECT id FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project:
                connection.rollback()
                raise ValueError("作品不存在")
            rows = connection.execute(
                f"""
                SELECT pref.id, pref.category, pref.status,
                       suggestion.source_type,
                       suggestion.after_version_id
                FROM author_editing_preferences pref
                LEFT JOIN editing_preference_suggestions suggestion
                  ON suggestion.id=pref.suggestion_id
                WHERE pref.project_id=? AND pref.id IN ({placeholders})
                """,
                (project_id, *all_ids),
            ).fetchall()
            by_id = {str(row["id"]): row for row in rows}
            if len(by_id) != len(all_ids):
                connection.rollback()
                raise ValueError("改稿观察不存在或不属于当前作品")
            if any(
                str(by_id[item_id]["status"]) != "active"
                for item_id in all_ids
            ):
                connection.rollback()
                raise ValueError("只能归并仍启用的改稿观察")
            if any(
                str(by_id[item_id]["category"]) != category
                for item_id in all_ids
            ):
                connection.rollback()
                raise ValueError("一次只能归并同一类别的改稿观察")
            support_sources = {
                (
                    str(by_id[item_id]["source_type"] or ""),
                    str(by_id[item_id]["after_version_id"] or ""),
                )
                for item_id in support_ids
                if by_id[item_id]["source_type"]
                and by_id[item_id]["after_version_id"]
            }
            if len(support_sources) < 2:
                connection.rollback()
                raise ValueError(
                    "支持证据必须来自至少两次不同的手工改稿"
                )
            occupied = connection.execute(
                f"""
                SELECT evidence.preference_id
                FROM author_editing_preference_aggregate_evidence evidence
                JOIN author_editing_preference_aggregates aggregate
                  ON aggregate.id=evidence.aggregate_id
                WHERE evidence.preference_id IN ({placeholders})
                  AND aggregate.status='active'
                LIMIT 1
                """,
                tuple(all_ids),
            ).fetchone()
            if occupied:
                connection.rollback()
                raise ValueError(
                    "所选改稿观察已归入另一条启用中的稳定偏好"
                )
            duplicate = connection.execute(
                """
                SELECT id
                FROM author_editing_preference_aggregates
                WHERE project_id=? AND status='active'
                  AND lower(guidance)=lower(?)
                  AND lower(applicability)=lower(?)
                LIMIT 1
                """,
                (project_id, guidance, applicability),
            ).fetchone()
            if duplicate:
                connection.rollback()
                raise ValueError("相同的稳定偏好已经启用")
            connection.execute(
                """
                INSERT INTO author_editing_preference_aggregates(
                    id, project_id, category, guidance, applicability,
                    author_note, status, created_at, updated_at,
                    confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    aggregate_id,
                    project_id,
                    category,
                    guidance,
                    applicability,
                    author_note,
                    now,
                    now,
                    now,
                ),
            )
            for preference_id in support_ids:
                connection.execute(
                    """
                    INSERT INTO
                      author_editing_preference_aggregate_evidence(
                        aggregate_id, preference_id, role, created_at
                      ) VALUES (?, ?, 'support', ?)
                    """,
                    (aggregate_id, preference_id, now),
                )
            for preference_id in conflict_ids:
                connection.execute(
                    """
                    INSERT INTO
                      author_editing_preference_aggregate_evidence(
                        aggregate_id, preference_id, role, created_at
                      ) VALUES (?, ?, 'conflict', ?)
                    """,
                    (aggregate_id, preference_id, now),
                )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return aggregate_id

    def list_aggregates(
        self,
        *,
        user_id: int,
        project_id: str,
        include_archived: bool = False,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT aggregate.*
                FROM author_editing_preference_aggregates aggregate
                JOIN novel_projects p ON p.id=aggregate.project_id
                WHERE aggregate.project_id=? AND p.user_id=?
                  AND (?=1 OR aggregate.status='active')
                ORDER BY
                  CASE aggregate.status
                    WHEN 'active' THEN 0 ELSE 1
                  END,
                  aggregate.updated_at DESC,
                  aggregate.rowid DESC
                LIMIT ?
                """,
                (
                    project_id,
                    user_id,
                    1 if include_archived else 0,
                    limit,
                ),
            ).fetchall()
            aggregate_ids = [str(row["id"]) for row in rows]
            evidence_rows = []
            if aggregate_ids:
                placeholders = ",".join(
                    "?" for _ in aggregate_ids
                )
                evidence_rows = connection.execute(
                    f"""
                    SELECT evidence.aggregate_id, evidence.role,
                           pref.id, pref.category, pref.guidance,
                           pref.applicability, pref.before_quote,
                           pref.after_quote, pref.status,
                           suggestion.source_type,
                           suggestion.after_version_id,
                           ch.position AS chapter_position,
                           ch.title AS chapter_title,
                           sb.goal AS scene_goal
                    FROM
                      author_editing_preference_aggregate_evidence evidence
                    JOIN author_editing_preferences pref
                      ON pref.id=evidence.preference_id
                    LEFT JOIN editing_preference_suggestions suggestion
                      ON suggestion.id=pref.suggestion_id
                    LEFT JOIN novel_chapters ch
                      ON ch.id=suggestion.chapter_id
                    LEFT JOIN novel_scene_beats sb
                      ON sb.id=suggestion.scene_beat_id
                    WHERE evidence.aggregate_id IN ({placeholders})
                    ORDER BY evidence.created_at, pref.created_at
                    """,
                    tuple(aggregate_ids),
                ).fetchall()
        members: dict[str, list[dict[str, Any]]] = {
            aggregate_id: [] for aggregate_id in aggregate_ids
        }
        for row in evidence_rows:
            item = dict(row)
            members[str(item.pop("aggregate_id"))].append(item)
        results = []
        for row in rows:
            item = dict(row)
            item["members"] = members[str(item["id"])]
            item["supports"] = [
                member
                for member in item["members"]
                if member["role"] == "support"
            ]
            item["conflicts"] = [
                member
                for member in item["members"]
                if member["role"] == "conflict"
            ]
            support_count = len(
                {
                    (
                        str(member.get("source_type") or ""),
                        str(member.get("after_version_id") or ""),
                    )
                    for member in item["supports"]
                }
            )
            item["support_count"] = support_count
            item["conflict_count"] = len(item["conflicts"])
            if support_count >= 5:
                confidence_label = "高频偏好"
            elif support_count >= 3:
                confidence_label = "稳定偏好"
            else:
                confidence_label = "重复出现"
            if item["conflict_count"]:
                confidence_label += " · 有冲突证据"
            item["confidence_label"] = confidence_label
            item["effect_observation"] = (
                self.get_effect_observation(
                    user_id=user_id,
                    aggregate=item,
                )
                if item["status"] == "active"
                else None
            )
            results.append(item)
        return results

    def get_effect_observation(
        self,
        *,
        user_id: int,
        aggregate: Mapping[str, Any],
    ) -> dict[str, Any]:
        category = str(aggregate.get("category") or "")
        issue_types = sorted(
            EDIT_PREFERENCE_STYLE_ISSUES.get(category, set())
        )
        result: dict[str, Any] = {
            "status": "unmapped",
            "direction": None,
            "issue_types": issue_types,
            "before": self._empty_effect_period(),
            "after": self._empty_effect_period(),
            "provider": "",
            "model": "",
            "comparability": "not_available",
            "window_days": EFFECT_WINDOW_DAYS,
            "disclaimer": (
                "这是同期审校中的相关问题观察，不能证明变化由该规则导致。"
            ),
        }
        if not issue_types:
            return result
        cutoff = _parse_datetime(aggregate.get("confirmed_at"))
        if not cutoff:
            result["status"] = "insufficient_both"
            return result
        project_id = str(aggregate.get("project_id") or "")
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                WITH eligible AS (
                  SELECT audit.*, audit.rowid AS audit_rowid,
                         version.chapter_id AS version_chapter_id,
                         version.created_at AS version_created_at,
                         version.content_hash,
                         version.effective_char_count,
                         version.char_count,
                         ROW_NUMBER() OVER(
                           PARTITION BY audit.version_id
                           ORDER BY audit.created_at DESC, audit.rowid DESC
                         ) AS latest_rank
                  FROM chapter_style_audits audit
                  JOIN novel_chapter_versions version
                    ON version.id=audit.version_id
                  JOIN novel_projects project
                    ON project.id=audit.project_id
                  WHERE audit.project_id=? AND project.user_id=?
                    AND (
                      (
                        julianday(version.created_at)<julianday(?)
                        AND julianday(audit.created_at)<julianday(?)
                      )
                      OR (
                        julianday(version.created_at)>julianday(?)
                        AND julianday(audit.created_at)>julianday(?)
                      )
                    )
                )
                SELECT eligible.id AS audit_id,
                       eligible.version_id,
                       eligible.version_chapter_id AS chapter_id,
                       eligible.version_created_at,
                       eligible.created_at AS audit_created_at,
                       eligible.content_hash,
                       eligible.effective_char_count,
                       eligible.char_count,
                       eligible.provider,
                       eligible.model,
                       eligible.issue_count AS located_issue_count,
                       eligible.dropped_issue_count,
                       issue.id AS issue_id,
                       issue.issue_type
                FROM eligible
                LEFT JOIN chapter_style_issues issue
                  ON issue.audit_id=eligible.id
                WHERE eligible.latest_rank=1
                ORDER BY eligible.version_created_at DESC,
                         eligible.created_at DESC
                """,
                (
                    project_id,
                    user_id,
                    cutoff.isoformat(timespec="seconds"),
                    cutoff.isoformat(timespec="seconds"),
                    cutoff.isoformat(timespec="seconds"),
                    cutoff.isoformat(timespec="seconds"),
                ),
            ).fetchall()

        observations: dict[str, dict[str, Any]] = {}
        for row in rows:
            audit_id = str(row["audit_id"])
            item = observations.setdefault(
                audit_id,
                {
                    "audit_id": audit_id,
                    "version_id": str(row["version_id"]),
                    "chapter_id": str(row["chapter_id"]),
                    "version_at": _parse_datetime(
                        row["version_created_at"]
                    ),
                    "audit_at": _parse_datetime(
                        row["audit_created_at"]
                    ),
                    "content_key": (
                        f"hash:{row['content_hash']}"
                        if str(row["content_hash"] or "")
                        else f"version:{row['version_id']}"
                    ),
                    "chars": max(
                        0,
                        int(
                            row["effective_char_count"]
                            or row["char_count"]
                            or 0
                        ),
                    ),
                    "provider": str(row["provider"] or ""),
                    "model": str(row["model"] or ""),
                    "located_issue_count": int(
                        row["located_issue_count"] or 0
                    ),
                    "dropped": int(
                        row["dropped_issue_count"] or 0
                    ),
                    "relevant_issue_ids": set(),
                },
            )
            if (
                row["issue_id"]
                and str(row["issue_type"]) in issue_types
            ):
                item["relevant_issue_ids"].add(
                    str(row["issue_id"])
                )

        lower_bound = cutoff - timedelta(days=EFFECT_WINDOW_DAYS)
        upper_bound = cutoff + timedelta(days=EFFECT_WINDOW_DAYS)
        periods: dict[str, list[dict[str, Any]]] = {
            "before": [],
            "after": [],
        }
        for item in observations.values():
            version_at = item["version_at"]
            audit_at = item["audit_at"]
            if not version_at or not audit_at:
                continue
            if (
                lower_bound <= version_at < cutoff
                and lower_bound <= audit_at < cutoff
            ):
                periods["before"].append(item)
            elif (
                cutoff < version_at <= upper_bound
                and cutoff < audit_at <= upper_bound
            ):
                periods["after"].append(item)

        deduplicated: dict[str, dict[str, dict[str, Any]]] = {
            "before": {},
            "after": {},
        }
        for period_name, items in periods.items():
            for item in items:
                key = str(item["content_key"])
                current = deduplicated[period_name].get(key)
                current_sort = (
                    current["version_at"],
                    current["audit_at"],
                ) if current else None
                item_sort = (item["version_at"], item["audit_at"])
                if current is None or item_sort > current_sort:
                    deduplicated[period_name][key] = item
        shared_content = set(deduplicated["before"]) & set(
            deduplicated["after"]
        )
        for content_key in shared_content:
            deduplicated["before"].pop(content_key, None)
            deduplicated["after"].pop(content_key, None)

        before_items = list(deduplicated["before"].values())
        after_items = list(deduplicated["after"].values())
        before_pairs = {
            (item["provider"], item["model"]) for item in before_items
        }
        after_pairs = {
            (item["provider"], item["model"]) for item in after_items
        }
        common_pairs = before_pairs & after_pairs
        if before_items and after_items and not common_pairs:
            result["comparability"] = "low"
        elif common_pairs:
            pair_counts = {
                pair: (
                    sum(
                        1
                        for item in before_items
                        if (item["provider"], item["model"]) == pair
                    ),
                    sum(
                        1
                        for item in after_items
                        if (item["provider"], item["model"]) == pair
                    ),
                )
                for pair in common_pairs
            }
            selected_pair = max(
                common_pairs,
                key=lambda pair: (
                    min(pair_counts[pair]),
                    sum(pair_counts[pair]),
                    pair,
                ),
            )
            before_items = [
                item
                for item in before_items
                if (item["provider"], item["model"]) == selected_pair
            ]
            after_items = [
                item
                for item in after_items
                if (item["provider"], item["model"]) == selected_pair
            ]
            result["provider"], result["model"] = selected_pair
            result["comparability"] = "matched"

        before_items = sorted(
            before_items,
            key=lambda item: (
                item["version_at"],
                item["audit_at"],
            ),
            reverse=True,
        )[:EFFECT_MAX_VERSIONS_PER_SIDE]
        after_items = sorted(
            after_items,
            key=lambda item: (
                item["version_at"],
                item["audit_at"],
            ),
            reverse=True,
        )[:EFFECT_MAX_VERSIONS_PER_SIDE]
        result["before"] = self._summarize_effect_period(before_items)
        result["after"] = self._summarize_effect_period(after_items)
        before_ready = self._effect_period_ready(result["before"])
        after_ready = self._effect_period_ready(result["after"])
        if not before_ready and not after_ready:
            result["status"] = "insufficient_both"
            return result
        if not before_ready:
            result["status"] = "insufficient_before"
            return result
        if not after_ready:
            result["status"] = "insufficient_after"
            return result
        if result["comparability"] == "low":
            result["status"] = "low_comparability"
            return result

        result["status"] = "observable"
        before_rate = float(result["before"]["rate_per_10k"])
        after_rate = float(result["after"]["rate_per_10k"])
        result["delta_per_10k"] = round(
            after_rate - before_rate, 2
        )
        if before_rate == 0:
            result["direction"] = (
                "stable" if after_rate == 0 else "increased"
            )
        elif after_rate <= before_rate * 0.75:
            result["direction"] = "decreased"
        elif after_rate >= before_rate * 1.25:
            result["direction"] = "increased"
        else:
            result["direction"] = "stable"
        if (
            result["before"]["capped_observations"]
            or result["after"]["capped_observations"]
        ):
            result["capped_observations"] = True
        return result

    @staticmethod
    def _empty_effect_period() -> dict[str, Any]:
        return {
            "versions": 0,
            "chapters": 0,
            "chars": 0,
            "issues": 0,
            "rate_per_10k": 0.0,
            "dropped": 0,
            "capped_observations": 0,
        }

    @classmethod
    def _summarize_effect_period(
        cls, observations: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if not observations:
            return cls._empty_effect_period()
        char_count = sum(
            max(0, int(item.get("chars") or 0))
            for item in observations
        )
        issue_count = sum(
            len(item.get("relevant_issue_ids") or [])
            for item in observations
        )
        return {
            "versions": len(observations),
            "chapters": len(
                {
                    str(item.get("chapter_id") or "")
                    for item in observations
                    if item.get("chapter_id")
                }
            ),
            "chars": char_count,
            "issues": issue_count,
            "rate_per_10k": round(
                issue_count * 10_000 / char_count, 2
            )
            if char_count
            else 0.0,
            "dropped": sum(
                max(0, int(item.get("dropped") or 0))
                for item in observations
            ),
            "capped_observations": sum(
                1
                for item in observations
                if int(item.get("located_issue_count") or 0)
                + int(item.get("dropped") or 0)
                >= 12
            ),
        }

    @staticmethod
    def _effect_period_ready(period: Mapping[str, Any]) -> bool:
        return (
            int(period.get("versions") or 0) >= EFFECT_MIN_VERSIONS
            and int(period.get("chapters") or 0) >= EFFECT_MIN_CHAPTERS
            and int(period.get("chars") or 0) >= EFFECT_MIN_CHARS
        )

    def archive_aggregate(
        self, *, user_id: int, aggregate_id: str
    ) -> Optional[str]:
        now = utc_now()
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE author_editing_preference_aggregates
                SET status='archived', updated_at=?, archived_at=?
                WHERE id=? AND status='active' AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE
                      p.id=author_editing_preference_aggregates.project_id
                      AND p.user_id=?
                )
                RETURNING project_id
                """,
                (now, now, aggregate_id, user_id),
            )
            row = cursor.fetchone()
            connection.commit()
        return str(row["project_id"]) if row else None

    def archive_preference(
        self, *, user_id: int, preference_id: str
    ) -> Optional[str]:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE author_editing_preferences
                SET status='archived', updated_at=?
                WHERE id=? AND status='active' AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=author_editing_preferences.project_id
                      AND p.user_id=?
                )
                AND NOT EXISTS(
                    SELECT 1
                    FROM author_editing_preference_aggregate_evidence e
                    JOIN author_editing_preference_aggregates aggregate
                      ON aggregate.id=e.aggregate_id
                    WHERE e.preference_id=author_editing_preferences.id
                      AND aggregate.status='active'
                )
                RETURNING project_id
                """,
                (utc_now(), preference_id, user_id),
            )
            row = cursor.fetchone()
            connection.commit()
        return str(row["project_id"]) if row else None

    def _resolve_source(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        source_type: str,
        after_version_id: str,
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = self._resolve_source_row(
                connection,
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                source_type=source_type,
                after_version_id=after_version_id,
            )
        return dict(row) if row else None

    @staticmethod
    def _resolve_source_row(
        connection,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        source_type: str,
        after_version_id: str,
    ):
        if source_type == "chapter":
            return connection.execute(
                """
                SELECT p.id AS project_id, p.title AS project_title,
                       p.genre, p.point_of_view, p.style_guide,
                       ch.id AS chapter_id, ch.title AS chapter_title,
                       NULL AS scene_beat_id, NULL AS scene_goal,
                       before_v.id AS before_version_id,
                       before_v.content_path AS before_content_path,
                       after_v.id AS after_version_id,
                       after_v.content_path AS after_content_path,
                       after_v.created_by AS after_created_by,
                       after_v.source AS after_source,
                       after_v.change_summary AS author_change_summary
                FROM novel_chapter_versions after_v
                JOIN novel_chapter_versions before_v
                  ON before_v.id=after_v.parent_version_id
                 AND before_v.chapter_id=after_v.chapter_id
                JOIN novel_chapters ch ON ch.id=after_v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE after_v.id=? AND ch.id=? AND p.id=? AND p.user_id=?
                """,
                (after_version_id, chapter_id, project_id, user_id),
            ).fetchone()
        return connection.execute(
            """
            SELECT p.id AS project_id, p.title AS project_title,
                   p.genre, p.point_of_view, p.style_guide,
                   ch.id AS chapter_id, ch.title AS chapter_title,
                   sb.id AS scene_beat_id, sb.goal AS scene_goal,
                   before_v.id AS before_version_id,
                   before_v.content_path AS before_content_path,
                   after_v.id AS after_version_id,
                   after_v.content_path AS after_content_path,
                   after_v.created_by AS after_created_by,
                   after_v.source AS after_source,
                   after_v.change_summary AS author_change_summary
            FROM novel_scene_versions after_v
            JOIN novel_scene_versions before_v
              ON before_v.id=after_v.parent_version_id
             AND before_v.scene_beat_id=after_v.scene_beat_id
            JOIN novel_scene_beats sb ON sb.id=after_v.scene_beat_id
            JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
            JOIN novel_chapters ch ON ch.id=cp.chapter_id
            JOIN novel_projects p ON p.id=ch.project_id
            WHERE after_v.id=? AND ch.id=? AND p.id=? AND p.user_id=?
            """,
            (after_version_id, chapter_id, project_id, user_id),
        ).fetchone()

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("改稿来源文件无法读取") from exc

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
