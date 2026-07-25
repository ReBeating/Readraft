from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .db import Database, utc_now
from .quality_audit import effective_char_count


SCENE_PLAN_FIELDS = (
    "pov_character",
    "goal",
    "obstacle",
    "action",
    "reveal",
    "conceal",
    "subtext",
    "location",
    "key_items",
    "end_state",
    "transition",
    "requirement_refs",
)


def _load_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def scene_plan_payload(scene: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(scene)
    key_items = data.get("key_items")
    if key_items is None:
        key_items = _load_json(data.get("key_items_json"), [])
    requirement_refs = data.get("requirement_refs")
    if requirement_refs is None:
        requirement_refs = _load_json(
            data.get("requirement_refs_json"), []
        )
    return {
        "pov_character": str(data.get("pov_character") or ""),
        "goal": str(data.get("goal") or ""),
        "obstacle": str(data.get("obstacle") or ""),
        "action": str(data.get("action") or ""),
        "reveal": str(data.get("reveal") or ""),
        "conceal": str(data.get("conceal") or ""),
        "subtext": str(data.get("subtext") or ""),
        "location": str(data.get("location") or ""),
        "key_items": [str(item) for item in key_items or []],
        "end_state": str(data.get("end_state") or ""),
        "transition": str(data.get("transition") or ""),
        "requirement_refs": [
            {
                "kind": str(item.get("kind") or ""),
                "text": str(item.get("text") or ""),
            }
            for item in requirement_refs or []
            if isinstance(item, Mapping)
        ],
    }


def scene_plan_fingerprint(scene: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        scene_plan_payload(scene),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def scene_target_chars(target_chars: int, scene_count: int) -> int:
    return max(500, min(2_400, int(target_chars / max(scene_count, 1))))


def scene_minimum_chars(target_chars: int, scene_count: int) -> int:
    per_scene = int(target_chars / max(scene_count, 1))
    return max(300, min(800, int(per_scene * 0.4)))


class SceneService:
    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _read_text(path: Any) -> str:
        if not path:
            return ""
        try:
            return Path(str(path)).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return ""

    def get_workbench(
        self, *, user_id: int, project_id: str, chapter_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            chapter = connection.execute(
                """
                SELECT ch.id, ch.project_id, ch.position, ch.title,
                       ch.content_path, ch.working_version_id,
                       ch.canonical_version_id, ch.char_count,
                       p.title AS project_title, p.point_of_view,
                       cp.id AS plan_id, cp.status AS plan_status,
                       cp.target_chars AS plan_target_chars
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                return None
            scene_rows = connection.execute(
                """
                SELECT sb.*, sv.kind AS version_kind,
                       sv.source AS version_source,
                       sv.parent_version_id AS version_parent_version_id,
                       sv.created_by AS version_created_by,
                       sv.content_path AS version_content_path,
                       sv.char_count AS version_char_count,
                       sv.effective_char_count AS version_effective_chars,
                       sv.plan_fingerprint AS version_plan_fingerprint,
                       sv.quality_status AS version_quality_status,
                       sv.hard_issue_count AS version_hard_issue_count,
                       sv.quality_override_reason,
                       sv.created_at AS version_created_at
                FROM novel_scene_beats sb
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                LEFT JOIN novel_scene_versions sv
                    ON sv.id=sb.current_version_id
                WHERE cp.chapter_id=? AND sb.beat_status='active'
                ORDER BY sb.position
                """,
                (chapter_id,),
            ).fetchall()
            version_rows = connection.execute(
                """
                SELECT sv.*
                FROM novel_scene_versions sv
                JOIN novel_scene_beats sb ON sb.id=sv.scene_beat_id
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                WHERE cp.chapter_id=?
                ORDER BY sv.created_at DESC, sv.rowid DESC
                """,
                (chapter_id,),
            ).fetchall()
            audit_rows = connection.execute(
                """
                SELECT qa.*
                FROM scene_quality_audits qa
                JOIN novel_scene_versions sv ON sv.id=qa.scene_version_id
                JOIN novel_scene_beats sb ON sb.id=sv.scene_beat_id
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                WHERE cp.chapter_id=?
                ORDER BY qa.created_at DESC, qa.rowid DESC
                """,
                (chapter_id,),
            ).fetchall()
            active_job = connection.execute(
                """
                SELECT id, operation, subject_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

        histories: dict[str, list[Dict[str, Any]]] = {}
        for row in version_rows:
            item = dict(row)
            histories.setdefault(str(item["scene_beat_id"]), []).append(item)
        latest_audits: dict[str, Dict[str, Any]] = {}
        for row in audit_rows:
            item = dict(row)
            version_id = str(item["scene_version_id"])
            if version_id in latest_audits:
                continue
            item["report"] = _load_json(item.get("result_json"), {})
            latest_audits[version_id] = item

        scenes: list[Dict[str, Any]] = []
        for row in scene_rows:
            item = dict(row)
            item["key_items"] = _load_json(
                item.pop("key_items_json"), []
            )
            item["requirement_refs"] = _load_json(
                item.pop("requirement_refs_json"), []
            )
            fingerprint = scene_plan_fingerprint(item)
            item["plan_fingerprint"] = fingerprint
            item["content"] = self._read_text(
                item.get("version_content_path")
            )
            has_version = bool(item.get("current_version_id"))
            item["is_stale"] = bool(
                has_version
                and (
                    str(item.get("draft_status") or "") == "stale"
                    or str(item.get("draft_plan_fingerprint") or "")
                    != fingerprint
                    or str(item.get("version_plan_fingerprint") or "")
                    != fingerprint
                )
            )
            item["quality_status"] = str(
                item.get("version_quality_status") or "pending"
            )
            item["audit"] = latest_audits.get(
                str(item.get("current_version_id") or "")
            )
            item["versions"] = histories.get(str(item["id"]), [])[:8]
            item["ready"] = bool(
                has_version
                and item["content"].strip()
                and not item["is_stale"]
                and item["quality_status"] in {"pass", "overridden"}
            )
            scenes.append(item)

        target_chars = int(chapter["plan_target_chars"] or 3000)
        minimum_chars = scene_minimum_chars(target_chars, len(scenes))
        target_per_scene = scene_target_chars(target_chars, len(scenes))
        for item in scenes:
            item["minimum_chars"] = minimum_chars
            item["target_chars"] = target_per_scene
        return {
            "chapter": dict(chapter),
            "plan_status": str(chapter["plan_status"] or "draft"),
            "scenes": scenes,
            "scene_count": len(scenes),
            "ready_count": sum(bool(item["ready"]) for item in scenes),
            "all_ready": bool(
                len(scenes) >= 2 and all(item["ready"] for item in scenes)
            ),
            "target_chars": target_chars,
            "active_job": dict(active_job) if active_job else None,
        }

    def get_generation_state(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
    ) -> Dict[str, Any]:
        workbench = self.get_workbench(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not workbench:
            raise ValueError("章节不存在")
        if workbench["plan_status"] != "confirmed":
            raise ValueError("章节任务卡尚未确认")
        scenes = list(workbench["scenes"])
        focused_index = next(
            (
                index
                for index, item in enumerate(scenes)
                if str(item["id"]) == scene_beat_id
            ),
            None,
        )
        if focused_index is None:
            raise ValueError("场景节拍不存在")
        focused = scenes[focused_index]
        sequence = []
        for item in scenes:
            sequence.append(
                {
                    "id": str(item["id"]),
                    "position": int(item["position"]),
                    **scene_plan_payload(item),
                    "draft_status": str(item.get("draft_status") or "empty"),
                    "char_count": int(item.get("draft_char_count") or 0),
                }
            )
        previous_scene = scenes[focused_index - 1] if focused_index else None
        next_scene = (
            scenes[focused_index + 1]
            if focused_index + 1 < len(scenes)
            else None
        )
        return {
            "workbench": workbench,
            "focused_scene": {
                "id": str(focused["id"]),
                "position": int(focused["position"]),
                **scene_plan_payload(focused),
                "plan_fingerprint": str(focused["plan_fingerprint"]),
            },
            "scene_sequence": sequence,
            "current_content": str(focused.get("content") or ""),
            "previous_scene_content": (
                str(previous_scene.get("content") or "")
                if previous_scene and not previous_scene["is_stale"]
                else ""
            ),
            "previous_scene": (
                {
                    "position": int(previous_scene["position"]),
                    **scene_plan_payload(previous_scene),
                }
                if previous_scene
                else None
            ),
            "next_scene": (
                {
                    "position": int(next_scene["position"]),
                    **scene_plan_payload(next_scene),
                }
                if next_scene
                else None
            ),
            "target_chars": int(focused["target_chars"]),
            "minimum_chars": int(focused["minimum_chars"]),
            "chapter_content_path": str(
                workbench["chapter"]["content_path"]
            ),
        }

    def get_audit_state(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        scene_version_id: str,
    ) -> Dict[str, Any]:
        workbench = self.get_workbench(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not workbench:
            raise ValueError("章节不存在")
        scene = next(
            (
                item
                for item in workbench["scenes"]
                if str(item.get("current_version_id") or "")
                == scene_version_id
            ),
            None,
        )
        if not scene:
            raise ValueError("只能检查当前场景版本")
        state = self.get_generation_state(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            scene_beat_id=str(scene["id"]),
        )
        state["scene_version_id"] = scene_version_id
        state["scene_version_path"] = str(
            scene.get("version_content_path") or ""
        )
        return state

    def record_manual_version(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        version_path: Path,
        content: str,
        source: str = "manual",
        kind: str = "manual",
        change_summary: str = "",
    ) -> str:
        if source not in {"manual", "restored"}:
            raise ValueError("不支持的场景版本来源")
        version_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            scene = self._owned_scene_row(
                connection,
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                scene_beat_id=scene_beat_id,
            )
            if not scene:
                connection.rollback()
                raise ValueError("场景节拍不存在")
            if str(scene["plan_status"]) != "confirmed":
                connection.rollback()
                raise ValueError("章节任务卡尚未确认")
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 任务正在运行，请完成后再保存场景")
            fingerprint = scene_plan_fingerprint(scene)
            self._insert_scene_version(
                connection,
                version_id=version_id,
                scene_beat_id=scene_beat_id,
                job_id=None,
                parent_version_id=scene["current_version_id"],
                kind=kind,
                source=source,
                content_path=version_path,
                content=content,
                plan_fingerprint=fingerprint,
                created_by="author",
                quality_status="pending",
                hard_issue_count=0,
                created_at=now,
            )
            connection.execute(
                """
                UPDATE novel_scene_versions
                SET change_summary=?
                WHERE id=?
                """,
                (change_summary[:1000], version_id),
            )
            connection.execute(
                """
                UPDATE novel_scene_beats
                SET current_version_id=?, draft_status='draft',
                    draft_plan_fingerprint=?, draft_char_count=?,
                    draft_updated_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    version_id,
                    fingerprint,
                    len(content),
                    now,
                    now,
                    scene_beat_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return version_id

    def complete_generation(
        self,
        *,
        job_id: str,
        claim_token: str,
        version_path: Path,
        content: str,
        report: Mapping[str, Any],
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        audit_input_tokens: int = 0,
        audit_output_tokens: int = 0,
        warning: str = "",
    ) -> Optional[str]:
        now = utc_now()
        version_id = uuid.uuid4().hex
        audit_id = uuid.uuid4().hex
        verdict = str(report.get("verdict") or "pending")
        if verdict not in {"pass", "block", "pending"}:
            raise ValueError("不支持的场景检查状态")
        findings = report.get("findings") or []
        hard_issue_count = sum(
            isinstance(item, Mapping) and item.get("severity") == "hard"
            for item in findings
        )
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT j.*, sb.current_version_id, cp.status AS plan_status,
                       sb.pov_character, sb.goal, sb.obstacle, sb.action,
                       sb.reveal, sb.conceal, sb.subtext, sb.location,
                       sb.key_items_json, sb.end_state, sb.transition,
                       sb.requirement_refs_json
                FROM generation_jobs j
                JOIN novel_scene_beats sb ON sb.id=j.subject_id
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                WHERE j.id=? AND j.status='running' AND j.claim_token=?
                    AND j.operation IN ('generate_scene', 'rewrite_scene')
                    AND cp.chapter_id=j.chapter_id
                    AND sb.beat_status='active'
                """,
                (job_id, claim_token),
            ).fetchone()
            if not job or str(job["plan_status"]) != "confirmed":
                connection.rollback()
                return None
            fingerprint = scene_plan_fingerprint(job)
            snapshot = _load_json(job["context_snapshot_json"], {})
            recorded_fingerprint = str(
                (snapshot.get("focused_scene") or {}).get(
                    "plan_fingerprint"
                )
                or ""
            )
            if recorded_fingerprint != fingerprint:
                connection.rollback()
                raise ValueError("场景节拍在生成期间发生变化，结果未接入当前草稿")
            self._insert_scene_version(
                connection,
                version_id=version_id,
                scene_beat_id=str(job["subject_id"]),
                job_id=job_id,
                parent_version_id=job["current_version_id"],
                kind=str(job["operation"]),
                source="generated",
                content_path=version_path,
                content=content,
                plan_fingerprint=fingerprint,
                created_by="ai",
                quality_status=verdict,
                hard_issue_count=hard_issue_count,
                created_at=now,
            )
            self._insert_scene_audit(
                connection,
                audit_id=audit_id,
                project_id=str(job["project_id"]),
                chapter_id=str(job["chapter_id"]),
                scene_beat_id=str(job["subject_id"]),
                scene_version_id=version_id,
                report=report,
                provider=provider,
                model=model,
                input_tokens=audit_input_tokens,
                output_tokens=audit_output_tokens,
                created_at=now,
            )
            connection.execute(
                """
                UPDATE novel_scene_beats
                SET current_version_id=?, draft_status='draft',
                    draft_plan_fingerprint=?, draft_char_count=?,
                    draft_updated_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    version_id,
                    fingerprint,
                    len(content),
                    now,
                    now,
                    job["subject_id"],
                ),
            )
            connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', input_tokens=?, output_tokens=?,
                    result_char_count=?, error=?, finished_at=?,
                    claim_token=NULL, lease_expires_at=NULL,
                    result_json=?
                WHERE id=?
                """,
                (
                    input_tokens,
                    output_tokens,
                    len(content),
                    warning[:2000] or None,
                    now,
                    json.dumps(
                        {
                            "scene_beat_id": str(job["subject_id"]),
                            "scene_version_id": version_id,
                            "scene_quality_audit_id": audit_id,
                            "quality": dict(report),
                        },
                        ensure_ascii=False,
                    ),
                    job_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, job["project_id"]),
            )
            connection.commit()
        return version_id

    def complete_audit(
        self,
        *,
        job_id: str,
        claim_token: str,
        report: Mapping[str, Any],
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        now = utc_now()
        audit_id = uuid.uuid4().hex
        verdict = str(report.get("verdict") or "pending")
        if verdict not in {"pass", "block", "pending"}:
            raise ValueError("不支持的场景检查状态")
        findings = report.get("findings") or []
        hard_issue_count = sum(
            isinstance(item, Mapping) and item.get("severity") == "hard"
            for item in findings
        )
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT j.*, sv.scene_beat_id, sv.content_hash,
                       sb.current_version_id
                FROM generation_jobs j
                JOIN novel_scene_versions sv ON sv.id=j.subject_id
                JOIN novel_scene_beats sb ON sb.id=sv.scene_beat_id
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                WHERE j.id=? AND j.status='running' AND j.claim_token=?
                    AND j.operation='audit_scene'
                    AND cp.chapter_id=j.chapter_id
                    AND sb.beat_status='active'
                """,
                (job_id, claim_token),
            ).fetchone()
            if (
                not job
                or str(job["current_version_id"] or "")
                != str(job["subject_id"])
            ):
                connection.rollback()
                return False
            self._insert_scene_audit(
                connection,
                audit_id=audit_id,
                project_id=str(job["project_id"]),
                chapter_id=str(job["chapter_id"]),
                scene_beat_id=str(job["scene_beat_id"]),
                scene_version_id=str(job["subject_id"]),
                report=report,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                created_at=now,
            )
            connection.execute(
                """
                UPDATE novel_scene_versions
                SET quality_status=?, hard_issue_count=?,
                    quality_override_reason='', quality_overridden_at=NULL
                WHERE id=?
                """,
                (verdict, hard_issue_count, job["subject_id"]),
            )
            connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', result_json=?, input_tokens=?,
                    output_tokens=?, result_char_count=?,
                    finished_at=?, error=NULL, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE id=?
                """,
                (
                    json.dumps(
                        {
                            "scene_beat_id": str(job["scene_beat_id"]),
                            "scene_version_id": str(job["subject_id"]),
                            "scene_quality_audit_id": audit_id,
                            "quality": dict(report),
                        },
                        ensure_ascii=False,
                    ),
                    input_tokens,
                    output_tokens,
                    int(report.get("effective_char_count") or 0),
                    now,
                    job_id,
                ),
            )
            connection.commit()
        return True

    def override_audit(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        scene_version_id: str,
        reason: str,
    ) -> bool:
        clean_reason = reason.strip()
        if len(clean_reason) < 10:
            raise ValueError("覆盖场景检查至少需要说明 10 个字符")
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_scene_versions
                SET quality_status='overridden',
                    quality_override_reason=?,
                    quality_overridden_at=?
                WHERE id=? AND scene_beat_id=?
                    AND quality_status IN ('block', 'pending')
                    AND EXISTS(
                        SELECT 1
                        FROM novel_scene_beats sb
                        JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                        JOIN novel_chapters ch ON ch.id=cp.chapter_id
                        JOIN novel_projects p ON p.id=ch.project_id
                        WHERE sb.id=? AND sb.current_version_id=?
                            AND ch.id=? AND p.id=? AND p.user_id=?
                            AND sb.beat_status='active'
                            AND sb.draft_status!='stale'
                    )
                """,
                (
                    clean_reason[:2000],
                    utc_now(),
                    scene_version_id,
                    scene_beat_id,
                    scene_beat_id,
                    scene_version_id,
                    chapter_id,
                    project_id,
                    user_id,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def build_assembly(
        self, *, user_id: int, project_id: str, chapter_id: str
    ) -> Dict[str, Any]:
        workbench = self.get_workbench(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not workbench:
            raise ValueError("章节不存在")
        if workbench["plan_status"] != "confirmed":
            raise ValueError("章节任务卡尚未确认")
        if workbench["active_job"]:
            raise ValueError("AI 任务正在运行，请完成后再组装章节")
        if len(workbench["scenes"]) < 2:
            raise ValueError("至少需要两个场景节拍")
        not_ready = [
            int(item["position"])
            for item in workbench["scenes"]
            if not item["ready"]
        ]
        if not_ready:
            joined = "、".join(str(position) for position in not_ready)
            raise ValueError(
                f"场景 {joined} 尚未通过检查或已因任务卡变化失效"
            )
        content = "\n\n".join(
            str(item["content"]).strip() for item in workbench["scenes"]
        ).strip()
        return {
            "content": content,
            "scene_versions": [
                {
                    "scene_beat_id": str(item["id"]),
                    "scene_version_id": str(item["current_version_id"]),
                    "position": int(item["position"]),
                }
                for item in workbench["scenes"]
            ],
        }

    def record_assembly(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_path: Path,
        content: str,
        scene_versions: Iterable[Mapping[str, Any]],
    ) -> str:
        expected = [
            {
                "scene_beat_id": str(item["scene_beat_id"]),
                "scene_version_id": str(item["scene_version_id"]),
                "position": int(item["position"]),
            }
            for item in scene_versions
        ]
        version_id = uuid.uuid4().hex
        now = utc_now()
        effective_count = effective_char_count(content)
        quality_status = "block" if effective_count < 2000 else "pending"
        hard_issue_count = 1 if effective_count < 2000 else 0
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chapter = connection.execute(
                """
                SELECT ch.working_version_id
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                connection.rollback()
                raise ValueError("章节不存在")
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 任务正在运行，请完成后再组装章节")
            current_rows = connection.execute(
                """
                SELECT sb.*, sv.quality_status,
                       sv.plan_fingerprint AS version_plan_fingerprint
                FROM novel_scene_beats sb
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                JOIN novel_scene_versions sv
                    ON sv.id=sb.current_version_id
                WHERE cp.chapter_id=? AND cp.status='confirmed'
                    AND sb.beat_status='active'
                ORDER BY sb.position
                """,
                (chapter_id,),
            ).fetchall()
            actual = [
                {
                    "scene_beat_id": str(row["id"]),
                    "scene_version_id": str(row["current_version_id"]),
                    "position": int(row["position"]),
                }
                for row in current_rows
            ]
            if actual != expected or len(actual) < 2:
                connection.rollback()
                raise ValueError("场景版本已经变化，请刷新后重新组装")
            for row in current_rows:
                fingerprint = scene_plan_fingerprint(row)
                if (
                    str(row["draft_status"]) == "stale"
                    or str(row["draft_plan_fingerprint"]) != fingerprint
                    or str(row["version_plan_fingerprint"]) != fingerprint
                    or str(row["quality_status"])
                    not in {"pass", "overridden"}
                ):
                    connection.rollback()
                    raise ValueError("存在未通过检查或已失效的场景")
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count,
                    created_at, parent_version_id, status, source,
                    content_hash, change_summary, created_by,
                    quality_status, effective_char_count, hard_issue_count
                ) VALUES (?, ?, 'scene_assembly', ?, ?, ?, ?,
                          'candidate', 'scene_assembly', ?, ?, 'author',
                          ?, ?, ?)
                """,
                (
                    version_id,
                    chapter_id,
                    str(version_path),
                    len(content),
                    now,
                    chapter["working_version_id"],
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    (
                        f"由 {len(expected)} 个已检查场景版本按任务卡顺序组装"
                    ),
                    quality_status,
                    effective_count,
                    hard_issue_count,
                ),
            )
            for item in expected:
                connection.execute(
                    """
                    INSERT INTO novel_scene_assembly_items(
                        chapter_version_id, scene_beat_id,
                        scene_version_id, position
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        item["scene_beat_id"],
                        item["scene_version_id"],
                        item["position"],
                    ),
                )
            connection.execute(
                """
                UPDATE novel_scene_beats
                SET draft_status='assembled', draft_updated_at=?
                WHERE plan_id=(
                    SELECT id FROM novel_chapter_plans
                    WHERE chapter_id=?
                )
                    AND beat_status='active'
                """,
                (now, chapter_id),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET working_version_id=?, char_count=?, status='draft',
                    updated_at=?
                WHERE id=?
                """,
                (version_id, len(content), now, chapter_id),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return version_id

    @staticmethod
    def _owned_scene_row(
        connection,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
    ):
        return connection.execute(
            """
            SELECT sb.*, cp.status AS plan_status
            FROM novel_scene_beats sb
            JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
            JOIN novel_chapters ch ON ch.id=cp.chapter_id
            JOIN novel_projects p ON p.id=ch.project_id
            WHERE sb.id=? AND ch.id=? AND p.id=? AND p.user_id=?
                AND sb.beat_status='active'
            """,
            (scene_beat_id, chapter_id, project_id, user_id),
        ).fetchone()

    @staticmethod
    def _insert_scene_version(
        connection,
        *,
        version_id: str,
        scene_beat_id: str,
        job_id: Optional[str],
        parent_version_id: Any,
        kind: str,
        source: str,
        content_path: Path,
        content: str,
        plan_fingerprint: str,
        created_by: str,
        quality_status: str,
        hard_issue_count: int,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO novel_scene_versions(
                id, scene_beat_id, job_id, parent_version_id, kind,
                source, status, content_path, char_count,
                effective_char_count, content_hash, plan_fingerprint,
                created_by, quality_status, hard_issue_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?,
                      ?, ?, ?)
            """,
            (
                version_id,
                scene_beat_id,
                job_id,
                parent_version_id,
                kind,
                source,
                str(content_path),
                len(content),
                effective_char_count(content),
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                plan_fingerprint,
                created_by,
                quality_status,
                hard_issue_count,
                created_at,
            ),
        )

    @staticmethod
    def _insert_scene_audit(
        connection,
        *,
        audit_id: str,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        scene_version_id: str,
        report: Mapping[str, Any],
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO scene_quality_audits(
                id, project_id, chapter_id, scene_beat_id,
                scene_version_id, verdict, effective_char_count,
                minimum_effective_chars, provider, model, result_json,
                input_tokens, output_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                project_id,
                chapter_id,
                scene_beat_id,
                scene_version_id,
                str(report.get("verdict") or "pending"),
                int(report.get("effective_char_count") or 0),
                int(report.get("minimum_effective_chars") or 1),
                provider,
                model,
                json.dumps(dict(report), ensure_ascii=False),
                input_tokens,
                output_tokens,
                created_at,
            ),
        )
