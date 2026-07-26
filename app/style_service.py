from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .db import Database, utc_after, utc_now
from .style_schema import StyleAuditResult, TargetedRewriteResult


def _json_list(value: Any) -> list[Any]:
    try:
        result = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return result if isinstance(result, list) else []


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


class StyleService:
    def __init__(self, database: Database):
        self.database = database

    def get_voice_profile(
        self, *, user_id: int, project_id: str
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT vp.*, p.title AS project_title, p.style_guide
                FROM novel_voice_profiles vp
                JOIN novel_projects p ON p.id=vp.project_id
                WHERE vp.project_id=? AND p.user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["preferred_patterns"] = _json_list(
            result.pop("preferred_patterns_json")
        )
        result["banned_expressions"] = _json_list(
            result.pop("banned_expressions_json")
        )
        return result

    def update_voice_profile(
        self,
        *,
        user_id: int,
        project_id: str,
        narration_rules: str,
        sentence_rhythm: str,
        dialogue_voice: str,
        sensory_palette: str,
        metaphor_policy: str,
        allowed_omissions: str,
        preferred_patterns: list[str],
        banned_expressions: list[str],
        author_notes: str,
        confirm: bool,
    ) -> bool:
        now = utc_now()
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_voice_profiles
                SET narration_rules=?, sentence_rhythm=?, dialogue_voice=?,
                    sensory_palette=?, metaphor_policy=?,
                    allowed_omissions=?, preferred_patterns_json=?,
                    banned_expressions_json=?, author_notes=?, status=?,
                    confirmed_at=?, updated_at=?
                WHERE project_id=? AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=novel_voice_profiles.project_id
                        AND p.user_id=?
                )
                """,
                (
                    narration_rules,
                    sentence_rhythm,
                    dialogue_voice,
                    sensory_palette,
                    metaphor_policy,
                    allowed_omissions,
                    json.dumps(preferred_patterns, ensure_ascii=False),
                    json.dumps(banned_expressions, ensure_ascii=False),
                    author_notes,
                    "confirmed" if confirm else "draft",
                    now if confirm else None,
                    now,
                    project_id,
                    user_id,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (now, project_id),
                )
            connection.commit()
        return cursor.rowcount == 1

    def create_voice_suggestion(
        self,
        *,
        user_id: int,
        project_id: str,
        sample_title: str,
        sample_text: str,
        author_intent: str,
        provider: str,
        model: str,
        credential_source: str,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        suggestion_id = uuid.uuid4().hex
        now = utc_now()
        sample_hash = hashlib.sha256(
            sample_text.encode("utf-8")
        ).hexdigest()
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
                raise ValueError("小说项目不存在")
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
                SELECT id FROM voice_profile_suggestions
                WHERE project_id=? AND sample_hash=?
                  AND status IN ('queued', 'running', 'ready')
                ORDER BY created_at DESC LIMIT 1
                """,
                (project_id, sample_hash),
            ).fetchone()
            if duplicate:
                connection.rollback()
                return str(duplicate["id"])
            active_voice = connection.execute(
                """
                SELECT id FROM voice_profile_suggestions
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            active_generation = connection.execute(
                """
                SELECT id FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active_voice or active_generation:
                connection.rollback()
                raise ValueError(
                    "你已有一个 AI 任务正在排队或运行，请等待其完成"
                )
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                counts = connection.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM generation_jobs
                       WHERE user_id=? AND created_at>=?) AS generations,
                      (SELECT COUNT(*) FROM analysis_jobs
                       WHERE user_id=? AND created_at>=?) AS analyses,
                      (SELECT COUNT(*) FROM voice_profile_suggestions
                       WHERE user_id=? AND created_at>=?) AS voice_jobs
                    """,
                    (
                        user_id,
                        day_start,
                        user_id,
                        day_start,
                        user_id,
                        day_start,
                    ),
                ).fetchone()
                total = sum(int(counts[key] or 0) for key in counts.keys())
                if total >= max_jobs_per_day:
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            connection.execute(
                """
                INSERT INTO voice_profile_suggestions(
                    id, project_id, user_id, sample_title, sample_text,
                    sample_hash, sample_char_count, author_intent,
                    provider, model, credential_source, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    suggestion_id,
                    project_id,
                    user_id,
                    sample_title,
                    sample_text,
                    sample_hash,
                    len(sample_text),
                    author_intent,
                    provider,
                    model,
                    credential_source,
                    now,
                ),
            )
            connection.commit()
        return suggestion_id

    def list_voice_suggestions(
        self, *, user_id: int, project_id: str, limit: int = 8
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT suggestion.*
                FROM voice_profile_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.project_id=? AND p.user_id=?
                ORDER BY suggestion.created_at DESC, suggestion.rowid DESC
                LIMIT ?
                """,
                (project_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_voice_suggestion(
        self, *, user_id: int, suggestion_id: str
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT suggestion.*, p.title AS project_title,
                       p.genre, p.point_of_view, p.style_guide,
                       vp.narration_rules AS current_narration_rules,
                       vp.sentence_rhythm AS current_sentence_rhythm,
                       vp.dialogue_voice AS current_dialogue_voice,
                       vp.sensory_palette AS current_sensory_palette,
                       vp.metaphor_policy AS current_metaphor_policy,
                       vp.allowed_omissions AS current_allowed_omissions,
                       vp.preferred_patterns_json
                           AS current_preferred_patterns_json,
                       vp.banned_expressions_json
                           AS current_banned_expressions_json,
                       vp.author_notes AS current_author_notes,
                       vp.status AS current_voice_status
                FROM voice_profile_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                JOIN novel_voice_profiles vp ON vp.project_id=p.id
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
        result["applied_profile"] = _json_dict(
            result.pop("applied_profile_json")
        )
        result["current_preferred_patterns"] = _json_list(
            result.pop("current_preferred_patterns_json")
        )
        result["current_banned_expressions"] = _json_list(
            result.pop("current_banned_expressions_json")
        )
        return result

    def claim_next_voice_suggestion(self) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE voice_profile_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次声纹提取租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT suggestion.*, p.title, p.genre, p.point_of_view,
                       p.style_guide, p.target_audience
                FROM voice_profile_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
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
                UPDATE voice_profile_suggestions
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
        result["claim_token"] = claim_token
        return result

    def complete_voice_suggestion(
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
                UPDATE voice_profile_suggestions
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

    def fail_voice_suggestion(
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
                UPDATE voice_profile_suggestions
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

    def release_voice_suggestion_claim(
        self, suggestion_id: str, claim_token: str, error: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE voice_profile_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], suggestion_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def apply_voice_suggestion(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        narration_rules: str,
        sentence_rhythm: str,
        dialogue_voice: str,
        sensory_palette: str,
        metaphor_policy: str,
        allowed_omissions: str,
        preferred_patterns: list[str],
        banned_expressions: list[str],
        author_notes: str,
        confirm: bool,
    ) -> Optional[str]:
        now = utc_now()
        applied_profile = {
            "narration_rules": narration_rules,
            "sentence_rhythm": sentence_rhythm,
            "dialogue_voice": dialogue_voice,
            "sensory_palette": sensory_palette,
            "metaphor_policy": metaphor_policy,
            "allowed_omissions": allowed_omissions,
            "preferred_patterns": preferred_patterns,
            "banned_expressions": banned_expressions,
            "author_notes": author_notes,
            "status": "confirmed" if confirm else "draft",
        }
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                """
                SELECT suggestion.project_id
                FROM voice_profile_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.id=? AND p.user_id=?
                  AND suggestion.status='ready'
                """,
                (suggestion_id, user_id),
            ).fetchone()
            if not target:
                connection.rollback()
                return None
            project_id = str(target["project_id"])
            profile_cursor = connection.execute(
                """
                UPDATE novel_voice_profiles
                SET narration_rules=?, sentence_rhythm=?, dialogue_voice=?,
                    sensory_palette=?, metaphor_policy=?,
                    allowed_omissions=?, preferred_patterns_json=?,
                    banned_expressions_json=?, author_notes=?, status=?,
                    confirmed_at=?, updated_at=?
                WHERE project_id=?
                """,
                (
                    narration_rules,
                    sentence_rhythm,
                    dialogue_voice,
                    sensory_palette,
                    metaphor_policy,
                    allowed_omissions,
                    json.dumps(preferred_patterns, ensure_ascii=False),
                    json.dumps(banned_expressions, ensure_ascii=False),
                    author_notes,
                    "confirmed" if confirm else "draft",
                    now if confirm else None,
                    now,
                    project_id,
                ),
            )
            suggestion_cursor = connection.execute(
                """
                UPDATE voice_profile_suggestions
                SET status='applied', applied_profile_json=?, reviewed_at=?
                WHERE id=? AND status='ready'
                """,
                (
                    json.dumps(applied_profile, ensure_ascii=False),
                    now,
                    suggestion_id,
                ),
            )
            if (
                profile_cursor.rowcount != 1
                or suggestion_cursor.rowcount != 1
            ):
                connection.rollback()
                return None
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return project_id

    def reject_voice_suggestion(
        self, *, user_id: int, suggestion_id: str
    ) -> Optional[str]:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE voice_profile_suggestions
                SET status='rejected', reviewed_at=?
                WHERE id=? AND status='ready' AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=voice_profile_suggestions.project_id
                      AND p.user_id=?
                )
                RETURNING project_id
                """,
                (utc_now(), suggestion_id, user_id),
            )
            row = cursor.fetchone()
            connection.commit()
        return str(row["project_id"]) if row else None

    def list_preferences(
        self, *, user_id: int, project_id: str, limit: int = 30
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            decision_rows = connection.execute(
                """
                SELECT pref.*
                FROM author_style_preferences pref
                JOIN novel_projects p ON p.id=pref.project_id
                WHERE pref.project_id=? AND p.user_id=?
                ORDER BY pref.created_at DESC, pref.rowid DESC
                LIMIT ?
                """,
                (project_id, user_id, limit),
            ).fetchall()
            learned_rows = connection.execute(
                """
                SELECT id, project_id, category, guidance, applicability,
                       created_at, source_type
                FROM (
                  SELECT aggregate.id, aggregate.project_id,
                         aggregate.category, aggregate.guidance,
                         aggregate.applicability,
                         aggregate.confirmed_at AS created_at,
                         aggregate.updated_at,
                         'stable_editing_preference' AS source_type,
                         0 AS source_rank
                  FROM author_editing_preference_aggregates aggregate
                  JOIN novel_projects p
                    ON p.id=aggregate.project_id
                  WHERE aggregate.project_id=? AND p.user_id=?
                    AND aggregate.status='active'

                  UNION ALL

                  SELECT pref.id, pref.project_id, pref.category,
                         pref.guidance, pref.applicability,
                         pref.created_at, pref.updated_at,
                         'manual_edit_learning' AS source_type,
                         1 AS source_rank
                  FROM author_editing_preferences pref
                  JOIN novel_projects p ON p.id=pref.project_id
                  WHERE pref.project_id=? AND p.user_id=?
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
                )
                ORDER BY source_rank, updated_at DESC, id DESC
                LIMIT ?
                """,
                (
                    project_id,
                    user_id,
                    project_id,
                    user_id,
                    limit,
                ),
            ).fetchall()
        preferences = []
        for row in decision_rows:
            item = dict(row)
            item["source_type"] = "style_decision"
            item["applicability"] = ""
            preferences.append(item)
        for row in learned_rows:
            item = dict(row)
            preferences.append(
                {
                    "id": item["id"],
                    "project_id": item["project_id"],
                    "issue_id": None,
                    "issue_type": item["category"],
                    "decision": "confirmed_manual_edit",
                    "original_text": "",
                    "replacement_text": "",
                    "guidance": item["guidance"],
                    "applicability": item["applicability"],
                    "created_at": item["created_at"],
                    "source_type": item["source_type"],
                }
            )
        preferences.sort(
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )
        return preferences[:limit]

    def create_audit(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
        result: StyleAuditResult,
        located_issues: list[Mapping[str, Any]],
        dropped_issue_count: int,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> str:
        now = utc_now()
        audit_id = uuid.uuid4().hex
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                """
                SELECT vp.id AS voice_profile_id, vp.status AS voice_status
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                JOIN novel_voice_profiles vp ON vp.project_id=p.id
                WHERE v.id=? AND ch.id=? AND p.id=? AND p.user_id=?
                """,
                (version_id, chapter_id, project_id, user_id),
            ).fetchone()
            if not target:
                connection.rollback()
                raise ValueError("正文版本或作品声纹不存在")
            if str(target["voice_status"]) != "confirmed":
                connection.rollback()
                raise ValueError("作品声纹尚未确认")
            connection.execute(
                """
                UPDATE chapter_style_issues
                SET status='superseded', updated_at=?
                WHERE version_id=? AND status='open'
                """,
                (now, version_id),
            )
            connection.execute(
                """
                UPDATE style_rewrite_candidates
                SET status='rejected', decided_at=?
                WHERE source_version_id=? AND status='candidate'
                """,
                (now, version_id),
            )
            connection.execute(
                """
                INSERT INTO chapter_style_audits(
                    id, project_id, chapter_id, version_id, voice_profile_id,
                    summary, issue_count, dropped_issue_count, provider, model,
                    result_json, input_tokens, output_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    project_id,
                    chapter_id,
                    version_id,
                    target["voice_profile_id"],
                    result.summary,
                    len(located_issues),
                    dropped_issue_count,
                    provider,
                    model,
                    result.model_dump_json(),
                    input_tokens,
                    output_tokens,
                    now,
                ),
            )
            for position, issue in enumerate(located_issues, start=1):
                connection.execute(
                    """
                    INSERT INTO chapter_style_issues(
                        id, audit_id, project_id, chapter_id, version_id,
                        position, paragraph_index, start_offset, end_offset,
                        quote, issue_type, severity, evidence, reader_impact,
                        rewrite_direction, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              'open', ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        audit_id,
                        project_id,
                        chapter_id,
                        version_id,
                        position,
                        int(issue["paragraph_index"]),
                        int(issue["start_offset"]),
                        int(issue["end_offset"]),
                        str(issue["quote"]),
                        str(issue["issue_type"]),
                        str(issue["severity"]),
                        str(issue["evidence"]),
                        str(issue["reader_impact"]),
                        str(issue["rewrite_direction"]),
                        now,
                        now,
                    ),
                )
            connection.execute(
                """
                UPDATE novel_chapter_versions
                SET style_status=?, style_issue_count=?
                WHERE id=?
                """,
                (
                    "issues" if located_issues else "pass",
                    len(located_issues),
                    version_id,
                ),
            )
            connection.commit()
        return audit_id

    def get_latest_audit(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            audit = connection.execute(
                """
                SELECT a.*
                FROM chapter_style_audits a
                JOIN novel_projects p ON p.id=a.project_id
                WHERE a.project_id=? AND a.chapter_id=? AND a.version_id=?
                    AND p.user_id=?
                ORDER BY a.created_at DESC, a.rowid DESC LIMIT 1
                """,
                (project_id, chapter_id, version_id, user_id),
            ).fetchone()
            if not audit:
                return None
            issues = connection.execute(
                """
                SELECT * FROM chapter_style_issues
                WHERE audit_id=? ORDER BY position
                """,
                (audit["id"],),
            ).fetchall()
        result = dict(audit)
        result["issues"] = [dict(row) for row in issues]
        return result

    def get_issue(
        self, *, user_id: int, issue_id: str
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT i.*, ch.title AS chapter_title, ch.content_path,
                       p.title AS project_title, p.user_id,
                       v.content_path AS version_content_path,
                       v.content_hash AS version_content_hash,
                       v.quality_status,
                       vp.status AS voice_status
                FROM chapter_style_issues i
                JOIN novel_chapters ch ON ch.id=i.chapter_id
                JOIN novel_projects p ON p.id=i.project_id
                JOIN novel_chapter_versions v ON v.id=i.version_id
                JOIN novel_voice_profiles vp ON vp.project_id=p.id
                WHERE i.id=? AND p.user_id=?
                """,
                (issue_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def get_issue_review(
        self, *, user_id: int, issue_id: str
    ) -> Optional[dict[str, Any]]:
        issue = self.get_issue(user_id=user_id, issue_id=issue_id)
        if not issue:
            return None
        with self.database.connection() as connection:
            candidates = connection.execute(
                """
                SELECT c.*
                FROM style_rewrite_candidates c
                JOIN novel_projects p ON p.id=c.project_id
                WHERE c.issue_id=? AND p.user_id=?
                ORDER BY c.created_at DESC, c.position
                """,
                (issue_id, user_id),
            ).fetchall()
        issue["candidates"] = [dict(row) for row in candidates]
        return issue

    def save_rewrite_candidates(
        self,
        *,
        user_id: int,
        issue_id: str,
        result: TargetedRewriteResult,
        provider: str,
        model: str,
    ) -> list[str]:
        now = utc_now()
        candidate_ids = []
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            issue = connection.execute(
                """
                SELECT i.*
                FROM chapter_style_issues i
                JOIN novel_projects p ON p.id=i.project_id
                WHERE i.id=? AND p.user_id=? AND i.status='open'
                """,
                (issue_id, user_id),
            ).fetchone()
            if not issue:
                connection.rollback()
                raise ValueError("问题不存在、已处理或无权访问")
            connection.execute(
                """
                UPDATE style_rewrite_candidates
                SET status='rejected', decided_at=?
                WHERE issue_id=? AND status='candidate'
                """,
                (now, issue_id),
            )
            for position, alternative in enumerate(
                result.alternatives, start=1
            ):
                candidate_id = uuid.uuid4().hex
                candidate_ids.append(candidate_id)
                connection.execute(
                    """
                    INSERT INTO style_rewrite_candidates(
                        id, issue_id, project_id, chapter_id,
                        source_version_id, position, replacement_text,
                        rationale, provider, model, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?)
                    """,
                    (
                        candidate_id,
                        issue_id,
                        issue["project_id"],
                        issue["chapter_id"],
                        issue["version_id"],
                        position,
                        alternative.replacement_text,
                        alternative.rationale,
                        provider,
                        model,
                        now,
                    ),
                )
            connection.commit()
        return candidate_ids

    def ignore_issue(
        self, *, user_id: int, issue_id: str
    ) -> Optional[dict[str, str]]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            issue = connection.execute(
                """
                SELECT i.*
                FROM chapter_style_issues i
                JOIN novel_projects p ON p.id=i.project_id
                WHERE i.id=? AND p.user_id=? AND i.status='open'
                """,
                (issue_id, user_id),
            ).fetchone()
            if not issue:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE chapter_style_issues
                SET status='ignored', updated_at=? WHERE id=?
                """,
                (now, issue_id),
            )
            connection.execute(
                """
                INSERT INTO author_style_preferences(
                    id, project_id, issue_id, issue_type, decision,
                    original_text, replacement_text, guidance, created_at
                ) VALUES (?, ?, ?, ?, 'ignored', ?, '', ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    issue["project_id"],
                    issue_id,
                    issue["issue_type"],
                    issue["quote"],
                    issue["rewrite_direction"],
                    now,
                ),
            )
            self._refresh_version_style_status(
                connection, str(issue["version_id"])
            )
            connection.commit()
        return {
            "project_id": str(issue["project_id"]),
            "chapter_id": str(issue["chapter_id"]),
            "version_id": str(issue["version_id"]),
        }

    def get_candidate(
        self, *, user_id: int, candidate_id: str
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT c.*, i.quote, i.start_offset, i.end_offset,
                       i.issue_type, i.rewrite_direction,
                       i.status AS issue_status,
                       v.content_path AS source_content_path,
                       v.content_hash AS source_content_hash,
                       ch.content_path AS working_content_path,
                       ch.title AS chapter_title, p.title AS project_title
                FROM style_rewrite_candidates c
                JOIN chapter_style_issues i ON i.id=c.issue_id
                JOIN novel_chapter_versions v ON v.id=c.source_version_id
                JOIN novel_chapters ch ON ch.id=c.chapter_id
                JOIN novel_projects p ON p.id=c.project_id
                WHERE c.id=? AND p.user_id=?
                """,
                (candidate_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def accept_rewrite_candidate(
        self,
        *,
        user_id: int,
        candidate_id: str,
        version_path: Path,
        char_count: int,
        effective_char_count: int,
        content_hash: str,
    ) -> Optional[dict[str, str]]:
        now = utc_now()
        version_id = uuid.uuid4().hex
        quality_status = (
            "block" if effective_char_count < 2000 else "pending"
        )
        hard_issue_count = 1 if effective_char_count < 2000 else 0
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute(
                """
                SELECT c.*, i.quote, i.issue_type, i.rewrite_direction,
                       i.status AS issue_status
                FROM style_rewrite_candidates c
                JOIN chapter_style_issues i ON i.id=c.issue_id
                JOIN novel_projects p ON p.id=c.project_id
                WHERE c.id=? AND p.user_id=? AND c.status='candidate'
                """,
                (candidate_id, user_id),
            ).fetchone()
            if not candidate or str(candidate["issue_status"]) != "open":
                connection.rollback()
                return None
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (candidate["chapter_id"],),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在处理本章，请等待任务完成后再接受改写")
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count, created_at,
                    parent_version_id, status, source, content_hash,
                    change_summary, created_by, quality_status,
                    effective_char_count, hard_issue_count, style_status,
                    style_issue_count
                ) VALUES (?, ?, 'targeted_rewrite', ?, ?, ?, ?, 'candidate',
                          'targeted_rewrite', ?, ?, 'ai', ?, ?, ?, 'pending', 0)
                """,
                (
                    version_id,
                    candidate["chapter_id"],
                    str(version_path),
                    char_count,
                    now,
                    candidate["source_version_id"],
                    content_hash,
                    (
                        f"定点修改 {candidate['issue_type']}："
                        f"{candidate['rewrite_direction']}"
                    )[:1000],
                    quality_status,
                    effective_char_count,
                    hard_issue_count,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET char_count=?, status='draft', working_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    char_count,
                    version_id,
                    now,
                    candidate["chapter_id"],
                ),
            )
            connection.execute(
                """
                UPDATE style_rewrite_candidates
                SET status=CASE WHEN id=? THEN 'accepted' ELSE 'rejected' END,
                    result_version_id=CASE WHEN id=? THEN ? ELSE NULL END,
                    decided_at=?
                WHERE issue_id=? AND status='candidate'
                """,
                (
                    candidate_id,
                    candidate_id,
                    version_id,
                    now,
                    candidate["issue_id"],
                ),
            )
            connection.execute(
                """
                UPDATE chapter_style_issues
                SET status='rewritten', updated_at=? WHERE id=?
                """,
                (now, candidate["issue_id"]),
            )
            connection.execute(
                """
                INSERT INTO author_style_preferences(
                    id, project_id, issue_id, issue_type, decision,
                    original_text, replacement_text, guidance, created_at
                ) VALUES (?, ?, ?, ?, 'accepted_rewrite', ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    candidate["project_id"],
                    candidate["issue_id"],
                    candidate["issue_type"],
                    candidate["quote"],
                    candidate["replacement_text"],
                    candidate["rewrite_direction"],
                    now,
                ),
            )
            self._refresh_version_style_status(
                connection, str(candidate["source_version_id"])
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, candidate["project_id"]),
            )
            connection.commit()
        return {
            "project_id": str(candidate["project_id"]),
            "chapter_id": str(candidate["chapter_id"]),
            "version_id": version_id,
        }

    @staticmethod
    def _refresh_version_style_status(
        connection: sqlite3.Connection, version_id: str
    ) -> None:
        open_count = connection.execute(
            """
            SELECT COUNT(*) AS count FROM chapter_style_issues
            WHERE version_id=? AND status='open'
            """,
            (version_id,),
        ).fetchone()
        connection.execute(
            """
            UPDATE novel_chapter_versions
            SET style_status=CASE WHEN ?=0 THEN 'reviewed' ELSE 'issues' END
            WHERE id=?
            """,
            (int(open_count["count"] or 0), version_id),
        )
