"""Persistence boundary for versioned, layered reference analysis."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

from .reference_analysis_aggregation import aggregate_document
from .reference_analysis_schema import (
    ALL_LAYERS,
    ANALYSIS_SCHEMA_VERSION,
)
from .db import Database, has_active_user_ai_task, utc_after, utc_now


class AnalysisRepository:
    def __init__(self, database: Database):
        self.database = database

    def create_job(
        self,
        *,
        user_id: int,
        document_id: str,
        provider: str,
        model: str,
        credential_source: str = "default",
    ) -> str:
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=? AND provider=?",
                    (user_id, provider),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError("所选模型服务 API Key 或凭据不存在，请重新配置")
            active = connection.execute(
                """
                SELECT id FROM analysis_jobs
                WHERE document_id=? AND user_id=?
                  AND status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (document_id, user_id),
            ).fetchone()
            if active:
                connection.rollback()
                return str(active["id"])
            if has_active_user_ai_task(connection, user_id):
                connection.rollback()
                raise ValueError("你已有一个 AI 任务正在排队或运行，请等待其完成")
            chapters = connection.execute(
                """
                SELECT c.id, c.content_path, c.content_hash
                FROM chapters c
                JOIN documents d ON d.id=c.document_id
                WHERE c.document_id=? AND d.user_id=?
                ORDER BY c.position
                """,
                (document_id, user_id),
            ).fetchall()
            if not chapters:
                connection.rollback()
                raise ValueError("文档没有可分析章节")
            connection.execute(
                """
                INSERT INTO analysis_jobs(
                    id, document_id, user_id, provider, model,
                    credential_source, status, total_chapters,
                    schema_version, aggregate_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, '{}', ?)
                """,
                (
                    job_id,
                    document_id,
                    user_id,
                    provider,
                    model,
                    credential_source,
                    len(chapters),
                    ANALYSIS_SCHEMA_VERSION,
                    utc_now(),
                ),
            )
            for chapter in chapters:
                digest = str(chapter["content_hash"] or "")
                if not digest:
                    try:
                        digest = hashlib.sha256(
                            Path(str(chapter["content_path"])).read_bytes()
                        ).hexdigest()
                    except OSError as exc:
                        connection.rollback()
                        raise ValueError("章节正文文件无法读取") from exc
                    connection.execute(
                        "UPDATE chapters SET content_hash=? WHERE id=?",
                        (digest, str(chapter["id"])),
                    )
                analysis_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO chapter_analyses(
                        id, job_id, chapter_id, status,
                        content_hash, schema_version
                    ) VALUES (?, ?, ?, 'queued', ?, ?)
                    """,
                    (
                        analysis_id,
                        job_id,
                        str(chapter["id"]),
                        digest,
                        ANALYSIS_SCHEMA_VERSION,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO chapter_analysis_layers(
                        analysis_id, layer, status
                    ) VALUES (?, ?, 'queued')
                    """,
                    [(analysis_id, layer) for layer in ALL_LAYERS],
                )
            connection.commit()
        return job_id

    def get_job(self, user_id: int, job_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT j.*, d.title AS document_title
                FROM analysis_jobs j
                JOIN documents d ON d.id=j.document_id
                WHERE j.id=? AND j.user_id=?
                """,
                (job_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def claim_next(self) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE chapter_analyses
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次处理租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT a.id AS analysis_id, a.job_id, a.chapter_id,
                       a.attempts, a.content_hash, a.schema_version,
                       c.title AS chapter_title, c.content_path, c.position,
                       j.user_id, j.document_id, j.provider, j.model,
                       j.credential_source
                FROM chapter_analyses a
                JOIN analysis_jobs j ON j.id=a.job_id
                JOIN chapters c ON c.id=a.chapter_id
                WHERE a.status='queued'
                  AND j.status IN ('queued', 'running')
                ORDER BY j.created_at, c.position
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='running', attempts=attempts+1, started_at=?,
                    error=NULL, claim_token=?, lease_expires_at=?
                WHERE id=? AND status='queued'
                """,
                (
                    now,
                    claim_token,
                    utc_after(2 * 60 * 60),
                    str(row["analysis_id"]),
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE chapter_analysis_layers
                SET status='queued', started_at=NULL,
                    error=COALESCE(error, '上一次分析层处理被中断')
                WHERE analysis_id=? AND status='running'
                """,
                (str(row["analysis_id"]),),
            )
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status='running', started_at=COALESCE(started_at, ?),
                    error=NULL
                WHERE id=?
                """,
                (now, str(row["job_id"])),
            )
            connection.commit()
        claimed = dict(row)
        claimed["claim_token"] = claim_token
        return claimed

    def release_claim(
        self, analysis_id: str, job_id: str, claim_token: str, error: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND job_id=? AND status='running'
                  AND claim_token=?
                """,
                (error[:2000], analysis_id, job_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def layers(self, analysis_id: str) -> dict[str, dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chapter_analysis_layers
                WHERE analysis_id=? ORDER BY rowid
                """,
                (analysis_id,),
            ).fetchall()
        return {str(row["layer"]): dict(row) for row in rows}

    def cached_layer(
        self,
        *,
        content_hash: str,
        layer: str,
        provider: str,
        model: str,
    ) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM chapter_analysis_cache
                WHERE content_hash=? AND schema_version=? AND layer=?
                  AND provider=? AND model=?
                """,
                (
                    content_hash,
                    ANALYSIS_SCHEMA_VERSION,
                    layer,
                    provider,
                    model,
                ),
            ).fetchone()
            if row:
                connection.execute(
                    """
                    UPDATE chapter_analysis_cache SET last_used_at=?
                    WHERE content_hash=? AND schema_version=? AND layer=?
                      AND provider=? AND model=?
                    """,
                    (
                        utc_now(),
                        content_hash,
                        ANALYSIS_SCHEMA_VERSION,
                        layer,
                        provider,
                        model,
                    ),
                )
                connection.commit()
        if not row:
            return None
        try:
            value = json.loads(str(row["result_json"]))
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def start_layer(
        self, *, analysis_id: str, layer: str, claim_token: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE chapter_analysis_layers
                SET status='running', attempts=attempts+1, error=NULL,
                    started_at=?, finished_at=NULL
                WHERE analysis_id=? AND layer=?
                  AND status IN ('queued', 'failed')
                  AND EXISTS(
                    SELECT 1 FROM chapter_analyses a
                    WHERE a.id=analysis_id AND a.status='running'
                      AND a.claim_token=?
                  )
                """,
                (utc_now(), analysis_id, layer, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_layer(
        self,
        *,
        analysis_id: str,
        content_hash: str,
        layer: str,
        provider: str,
        model: str,
        result: Mapping[str, Any],
        raw_response: str,
        input_tokens: int,
        output_tokens: int,
        claim_token: str,
        cache_hit: bool = False,
    ) -> bool:
        encoded = json.dumps(dict(result), ensure_ascii=False)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE chapter_analysis_layers
                SET status='completed', result_json=?, raw_response=?,
                    input_tokens=?, output_tokens=?, cache_hit=?, error=NULL,
                    finished_at=?
                WHERE analysis_id=? AND layer=?
                  AND EXISTS(
                    SELECT 1 FROM chapter_analyses a
                    WHERE a.id=analysis_id AND a.status='running'
                      AND a.claim_token=?
                  )
                """,
                (
                    encoded,
                    raw_response,
                    input_tokens,
                    output_tokens,
                    int(cache_hit),
                    utc_now(),
                    analysis_id,
                    layer,
                    claim_token,
                ),
            )
            if cursor.rowcount and not cache_hit:
                now = utc_now()
                connection.execute(
                    """
                    INSERT INTO chapter_analysis_cache(
                        content_hash, schema_version, layer, provider, model,
                        result_json, created_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(content_hash, schema_version, layer, provider, model)
                    DO UPDATE SET result_json=excluded.result_json,
                                  last_used_at=excluded.last_used_at
                    """,
                    (
                        content_hash,
                        ANALYSIS_SCHEMA_VERSION,
                        layer,
                        provider,
                        model,
                        encoded,
                        now,
                        now,
                    ),
                )
            connection.commit()
        return cursor.rowcount == 1

    def fail_layer(
        self,
        *,
        analysis_id: str,
        layer: str,
        error: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE chapter_analysis_layers
                SET status='failed', error=?, input_tokens=?, output_tokens=?,
                    finished_at=?
                WHERE analysis_id=? AND layer=?
                """,
                (
                    error[:2000],
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    analysis_id,
                    layer,
                ),
            )
            connection.commit()

    def complete(
        self,
        *,
        analysis_id: str,
        job_id: str,
        result: Mapping[str, Any],
        claim_token: str,
    ) -> bool:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tokens = connection.execute(
                """
                SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens
                FROM chapter_analysis_layers WHERE analysis_id=?
                """,
                (analysis_id,),
            ).fetchone()
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='completed', result_json=?, raw_response='',
                    input_tokens=?, output_tokens=?, error=NULL, finished_at=?,
                    claim_token=NULL, lease_expires_at=NULL
                WHERE id=? AND job_id=? AND status='running'
                  AND claim_token=?
                """,
                (
                    json.dumps(dict(result), ensure_ascii=False),
                    int(tokens["input_tokens"] or 0),
                    int(tokens["output_tokens"] or 0),
                    utc_now(),
                    analysis_id,
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount:
                self._refresh_job(connection, job_id)
            connection.commit()
        return cursor.rowcount == 1

    def fail(
        self,
        *,
        analysis_id: str,
        job_id: str,
        error: str,
        claim_token: str,
    ) -> bool:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tokens = connection.execute(
                """
                SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens
                FROM chapter_analysis_layers WHERE analysis_id=?
                """,
                (analysis_id,),
            ).fetchone()
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='failed', error=?, input_tokens=?, output_tokens=?,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL
                WHERE id=? AND job_id=? AND status='running'
                  AND claim_token=?
                """,
                (
                    error[:2000],
                    int(tokens["input_tokens"] or 0),
                    int(tokens["output_tokens"] or 0),
                    utc_now(),
                    analysis_id,
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount:
                self._refresh_job(connection, job_id)
            connection.commit()
        return cursor.rowcount == 1

    def retry_failed(self, user_id: int, job_id: str) -> bool:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT status FROM analysis_jobs WHERE id=? AND user_id=?",
                (job_id, user_id),
            ).fetchone()
            if not owner or owner["status"] not in {"partial", "failed"}:
                connection.rollback()
                return False
            other = connection.execute(
                """
                SELECT 1 FROM analysis_jobs
                WHERE user_id=? AND id<>? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (user_id, job_id),
            ).fetchone()
            if other:
                connection.rollback()
                return False
            failed_ids = [
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM chapter_analyses WHERE job_id=? AND status='failed'",
                    (job_id,),
                ).fetchall()
            ]
            if not failed_ids:
                connection.rollback()
                return False
            placeholders = ",".join("?" for _ in failed_ids)
            connection.execute(
                f"""
                UPDATE chapter_analysis_layers
                SET status='queued', error=NULL, started_at=NULL,
                    finished_at=NULL
                WHERE analysis_id IN ({placeholders}) AND status='failed'
                """,
                failed_ids,
            )
            connection.execute(
                f"""
                UPDATE chapter_analyses
                SET status='queued', error=NULL, started_at=NULL,
                    finished_at=NULL, claim_token=NULL, lease_expires_at=NULL
                WHERE id IN ({placeholders})
                """,
                failed_ids,
            )
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status='queued', failed_chapters=0, error=NULL,
                    finished_at=NULL
                WHERE id=?
                """,
                (job_id,),
            )
            connection.commit()
        return True

    def get_analysis(
        self, user_id: int, analysis_id: str
    ) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT a.*, c.title AS chapter_title, c.position,
                       c.char_count, c.content_path,
                       d.title AS document_title, d.id AS document_id,
                       j.model, j.provider
                FROM chapter_analyses a
                JOIN chapters c ON c.id=a.chapter_id
                JOIN analysis_jobs j ON j.id=a.job_id
                JOIN documents d ON d.id=j.document_id
                WHERE a.id=? AND j.user_id=?
                """,
                (analysis_id, user_id),
            ).fetchone()
            layers = connection.execute(
                """
                SELECT layer, status, result_json, input_tokens,
                       output_tokens, attempts, cache_hit, error
                FROM chapter_analysis_layers
                WHERE analysis_id=? ORDER BY rowid
                """,
                (analysis_id,),
            ).fetchall()
        if not row:
            return None
        result = dict(row)
        result["layers"] = [dict(item) for item in layers]
        return result

    def export_job(self, user_id: int, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(user_id, job_id)
        if not job:
            return None
        chapters = self.database.list_chapters(user_id, job["document_id"], job_id)
        return {
            "schema_version": str(job.get("schema_version") or "1.0"),
            "document": {
                "id": job["document_id"],
                "title": job["document_title"],
            },
            "job": {
                key: job.get(key)
                for key in (
                    "id",
                    "provider",
                    "model",
                    "status",
                    "created_at",
                    "finished_at",
                )
            },
            "aggregate": self.decode_object(job.get("aggregate_json")),
            "chapters": [
                {
                    "position": chapter["position"],
                    "title": chapter["title"],
                    "status": chapter["analysis_status"],
                    "analysis": self.decode_object(chapter.get("result_json")),
                }
                for chapter in chapters
            ],
        }

    def _refresh_job(self, connection: Any, job_id: str) -> None:
        counts = connection.execute(
            """
            SELECT
                SUM(status='completed') AS completed,
                SUM(status='failed') AS failed,
                SUM(status IN ('queued', 'running')) AS active,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens
            FROM chapter_analyses WHERE job_id=?
            """,
            (job_id,),
        ).fetchone()
        completed = int(counts["completed"] or 0)
        failed = int(counts["failed"] or 0)
        active = int(counts["active"] or 0)
        if active:
            state, finished_at = "running", None
        elif failed and completed:
            state, finished_at = "partial", utc_now()
        elif failed:
            state, finished_at = "failed", utc_now()
        else:
            state, finished_at = "completed", utc_now()
        aggregate: dict[str, Any] = {}
        if not active:
            results = []
            rows = connection.execute(
                """
                SELECT analysis.result_json, chapter.position
                FROM chapter_analyses analysis
                JOIN chapters chapter ON chapter.id=analysis.chapter_id
                WHERE analysis.job_id=? AND analysis.status='completed'
                ORDER BY chapter.position
                """,
                (job_id,),
            ).fetchall()
            for row in rows:
                parsed = self.decode_object(row["result_json"])
                if parsed:
                    parsed.setdefault("chapter_position", int(row["position"]))
                    results.append(parsed)
            aggregate = aggregate_document(results)
        connection.execute(
            """
            UPDATE analysis_jobs
            SET status=?, completed_chapters=?, failed_chapters=?,
                input_tokens=?, output_tokens=?, finished_at=?,
                aggregate_json=?
            WHERE id=?
            """,
            (
                state,
                completed,
                failed,
                int(counts["input_tokens"] or 0),
                int(counts["output_tokens"] or 0),
                finished_at,
                json.dumps(aggregate, ensure_ascii=False),
                job_id,
            ),
        )

    @staticmethod
    def decode_object(value: Any) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
