from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .db import Database, utc_now


IMPACT_DECISIONS = {"recheck", "keep", "rewrite_future", "retire"}


def _text(value: Any) -> str:
    return str(value or "").strip()


class CanonImpactService:
    def __init__(self, database: Database):
        self.database = database

    def list_reports(
        self, *, user_id: int, project_id: str, limit: int = 8
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT r.*, ch.position AS chapter_position,
                       ch.title AS chapter_title
                FROM canon_impact_reports r
                JOIN novel_chapters ch ON ch.id=r.chapter_id
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.project_id=? AND p.user_id=?
                ORDER BY r.created_at DESC, r.rowid DESC
                LIMIT ?
                """,
                (project_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_report(
        self, *, user_id: int, report_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT r.*, p.title AS project_title,
                       ch.position AS chapter_position,
                       ch.title AS chapter_title,
                       ch.canonical_version_id AS current_canonical_version_id,
                       old.char_count AS old_char_count,
                       old.created_at AS old_version_created_at,
                       proposed.char_count AS proposed_char_count,
                       proposed.created_at AS proposed_version_created_at,
                       proposed.content_hash AS proposed_content_hash,
                       proposed.content_path AS proposed_content_path,
                       ch.content_path AS working_content_path
                FROM canon_impact_reports r
                JOIN novel_projects p ON p.id=r.project_id
                JOIN novel_chapters ch ON ch.id=r.chapter_id
                JOIN novel_chapter_versions old ON old.id=r.old_version_id
                JOIN novel_chapter_versions proposed
                    ON proposed.id=r.proposed_version_id
                WHERE r.id=? AND p.user_id=?
                """,
                (report_id, user_id),
            ).fetchone()
            if not row:
                return None
            items = connection.execute(
                """
                SELECT i.*, ch.position AS downstream_chapter_position,
                       ch.title AS downstream_chapter_title
                FROM canon_impact_items i
                LEFT JOIN novel_chapters ch
                    ON ch.id=i.downstream_chapter_id
                WHERE i.report_id=?
                ORDER BY i.position
                """,
                (report_id,),
            ).fetchall()
        result = dict(row)
        result["items"] = [dict(item) for item in items]
        return result

    def prepare_report(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        proposed_version_id: str,
        override_reason: str = "",
    ) -> str:
        now = utc_now()
        clean_override = override_reason.strip()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                """
                SELECT ch.position, ch.title, ch.canonical_version_id,
                       v.char_count,
                       old.char_count AS old_char_count
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                JOIN novel_chapter_versions v
                    ON v.id=? AND v.chapter_id=ch.id
                LEFT JOIN novel_chapter_versions old
                    ON old.id=ch.canonical_version_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (
                    proposed_version_id,
                    chapter_id,
                    project_id,
                    user_id,
                ),
            ).fetchone()
            if not target:
                connection.rollback()
                raise ValueError("章节或候选版本不存在")
            old_version_id = _text(target["canonical_version_id"])
            if not old_version_id:
                connection.rollback()
                raise ValueError("本章尚无旧正史，不需要旧章影响报告")
            if old_version_id == proposed_version_id:
                connection.rollback()
                raise ValueError("这个版本已经是当前正史")
            existing = connection.execute(
                """
                SELECT id FROM canon_impact_reports
                WHERE chapter_id=? AND old_version_id=?
                  AND proposed_version_id=? AND status='pending'
                """,
                (chapter_id, old_version_id, proposed_version_id),
            ).fetchone()
            if existing:
                connection.commit()
                return str(existing["id"])

            downstream = connection.execute(
                """
                SELECT ch.id, ch.position, ch.title, ch.canonical_version_id,
                       ch.needs_recheck, m.summary
                FROM novel_chapters ch
                LEFT JOIN chapter_memory m
                    ON m.chapter_id=ch.id
                   AND m.version_id=ch.canonical_version_id
                   AND m.record_status='canon'
                WHERE ch.project_id=? AND ch.position>?
                  AND ch.canonical_version_id IS NOT NULL
                ORDER BY ch.position
                """,
                (project_id, target["position"]),
            ).fetchall()
            downstream_ids = [str(row["id"]) for row in downstream]
            old_rows = connection.execute(
                """
                SELECT subject_name AS name, predicate AS detail
                FROM story_facts
                WHERE chapter_id=? AND version_id=? AND fact_status='canon'
                UNION ALL
                SELECT character_name AS name, fact_text AS detail
                FROM character_knowledge
                WHERE chapter_id=? AND version_id=? AND record_status='canon'
                UNION ALL
                SELECT thread_name AS name, update_text AS detail
                FROM plot_threads
                WHERE chapter_id=? AND version_id=? AND record_status='canon'
                UNION ALL
                SELECT hook_name AS name, description AS detail
                FROM foreshadowing
                WHERE chapter_id=? AND version_id=? AND record_status='canon'
                """,
                (
                    chapter_id,
                    old_version_id,
                    chapter_id,
                    old_version_id,
                    chapter_id,
                    old_version_id,
                    chapter_id,
                    old_version_id,
                ),
            ).fetchall()
            old_terms = {
                term.casefold()
                for row in old_rows
                for term in (_text(row["name"]),)
                if len(term) >= 2
            }
            items: List[Dict[str, Any]] = []
            for row in downstream:
                evidence = (
                    f"第 {row['position']} 章已有正史版本"
                    + (
                        f"，确认后的故事记忆摘要为：{_text(row['summary'])[:500]}"
                        if row["summary"]
                        else "，尚未形成确认后的故事记忆"
                    )
                )
                items.append(
                    {
                        "item_type": "chapter",
                        "downstream_chapter_id": str(row["id"]),
                        "source_record_id": str(
                            row["canonical_version_id"]
                        ),
                        "title": (
                            f"第 {row['position']} 章《{row['title']}》"
                        ),
                        "evidence": evidence,
                        "risk_level": "high",
                        "recommended_action": (
                            "检查这一章是否依赖被替换版本中的事实；"
                            "确认不受影响后再清除待复查标记。"
                        ),
                        "decision": "recheck",
                    }
                )
            if downstream_ids:
                placeholders = ",".join("?" for _ in downstream_ids)
                dependencies = connection.execute(
                    f"""
                    SELECT 'fact' AS item_type, f.id AS source_record_id,
                           f.chapter_id, f.subject_name AS title,
                           f.predicate || ' ' || f.object_json AS evidence
                    FROM story_facts f
                    WHERE f.chapter_id IN ({placeholders})
                      AND f.fact_status='canon'
                    UNION ALL
                    SELECT 'knowledge', k.id, k.chapter_id,
                           k.character_name, k.fact_text
                    FROM character_knowledge k
                    WHERE k.chapter_id IN ({placeholders})
                      AND k.record_status='canon'
                    """,
                    (*downstream_ids, *downstream_ids),
                ).fetchall()
                for row in dependencies:
                    haystack = (
                        _text(row["title"]) + " " + _text(row["evidence"])
                    ).casefold()
                    matched = sorted(
                        term for term in old_terms if term in haystack
                    )
                    if not matched:
                        continue
                    items.append(
                        {
                            "item_type": str(row["item_type"]),
                            "downstream_chapter_id": str(row["chapter_id"]),
                            "source_record_id": str(
                                row["source_record_id"]
                            ),
                            "title": _text(row["title"]),
                            "evidence": (
                                f"与旧正史实体“{'、'.join(matched[:5])}”"
                                f"存在文本关联：{_text(row['evidence'])[:500]}"
                            ),
                            "risk_level": "high",
                            "recommended_action": (
                                "核对新正史是否仍支持这条事实或人物知情；"
                                "不要自动沿用。"
                            ),
                            "decision": "recheck",
                        }
                    )
                threads = connection.execute(
                    f"""
                    SELECT 'plot_thread' AS item_type, t.id AS source_record_id,
                           t.chapter_id, t.thread_name AS title,
                           t.update_text || ' ' || t.promise || ' ' ||
                           t.target_payoff AS evidence
                    FROM plot_threads t
                    WHERE t.chapter_id IN ({placeholders})
                      AND t.record_status='canon'
                    UNION ALL
                    SELECT 'foreshadowing', f.id, f.chapter_id, f.hook_name,
                           f.description || ' ' || f.intended_payoff
                    FROM foreshadowing f
                    WHERE f.chapter_id IN ({placeholders})
                      AND f.record_status='canon'
                    ORDER BY item_type, title
                    LIMIT 120
                    """,
                    (*downstream_ids, *downstream_ids),
                ).fetchall()
                for row in threads:
                    items.append(
                        {
                            "item_type": str(row["item_type"]),
                            "downstream_chapter_id": str(row["chapter_id"]),
                            "source_record_id": str(
                                row["source_record_id"]
                            ),
                            "title": _text(row["title"]),
                            "evidence": _text(row["evidence"])[:600],
                            "risk_level": "medium",
                            "recommended_action": (
                                "核对这条剧情线或伏笔的成立条件、推进方式"
                                "与预定回收是否仍然有效。"
                            ),
                            "decision": "recheck",
                        }
                    )
            report_id = uuid.uuid4().hex
            summary = (
                f"拟将第 {target['position']} 章《{target['title']}》从旧正史"
                f"版本 {old_version_id[:8]} 切换为 {proposed_version_id[:8]}。"
                f"检测到 {len(downstream)} 个后续正史章节和 "
                f"{max(0, len(items) - len(downstream))} 条可能依赖项。"
                "报告只标记与建议，不会自动改写后续正文。"
            )
            connection.execute(
                """
                INSERT INTO canon_impact_reports(
                    id, project_id, chapter_id, old_version_id,
                    proposed_version_id, status, summary,
                    downstream_count, item_count, override_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    project_id,
                    chapter_id,
                    old_version_id,
                    proposed_version_id,
                    summary,
                    len(downstream),
                    len(items),
                    clean_override[:1000],
                    now,
                    now,
                ),
            )
            for position, item in enumerate(items, start=1):
                connection.execute(
                    """
                    INSERT INTO canon_impact_items(
                        id, report_id, position, item_type,
                        downstream_chapter_id, source_record_id, title,
                        evidence, risk_level, recommended_action,
                        decision, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        report_id,
                        position,
                        item["item_type"],
                        item["downstream_chapter_id"],
                        item["source_record_id"],
                        item["title"],
                        item["evidence"],
                        item["risk_level"],
                        item["recommended_action"],
                        item["decision"],
                        now,
                    ),
                )
            connection.commit()
        return report_id

    def update_decisions(
        self,
        *,
        user_id: int,
        report_id: str,
        decisions: Mapping[str, Mapping[str, str]],
    ) -> bool:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            report = connection.execute(
                """
                SELECT r.id
                FROM canon_impact_reports r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND p.user_id=? AND r.status='pending'
                """,
                (report_id, user_id),
            ).fetchone()
            if not report:
                connection.rollback()
                return False
            valid_ids = {
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM canon_impact_items WHERE report_id=?",
                    (report_id,),
                ).fetchall()
            }
            if set(decisions) - valid_ids:
                raise ValueError("影响项已经变化，请刷新报告后重试")
            for item_id, choice in decisions.items():
                decision = str(choice.get("decision") or "recheck")
                if decision not in IMPACT_DECISIONS:
                    raise ValueError("不支持的影响处理方式")
                note = str(choice.get("note") or "").strip()[:1000]
                connection.execute(
                    """
                    UPDATE canon_impact_items
                    SET decision=?, decision_note=?, decided_at=?
                    WHERE id=? AND report_id=?
                    """,
                    (decision, note, now, item_id, report_id),
                )
            connection.execute(
                "UPDATE canon_impact_reports SET updated_at=? WHERE id=?",
                (now, report_id),
            )
            connection.commit()
        return True

    def mark_applied(
        self, *, user_id: int, report_id: str
    ) -> bool:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            report = connection.execute(
                """
                SELECT r.*
                FROM canon_impact_reports r
                JOIN novel_projects p ON p.id=r.project_id
                JOIN novel_chapters ch ON ch.id=r.chapter_id
                WHERE r.id=? AND p.user_id=? AND r.status='pending'
                  AND ch.canonical_version_id=r.proposed_version_id
                """,
                (report_id, user_id),
            ).fetchone()
            if not report:
                connection.rollback()
                return False
            chapters = connection.execute(
                """
                SELECT downstream_chapter_id,
                       MAX(CASE WHEN decision IN
                           ('recheck', 'rewrite_future', 'retire')
                           THEN 1 ELSE 0 END) AS requires_recheck
                FROM canon_impact_items
                WHERE report_id=? AND downstream_chapter_id IS NOT NULL
                GROUP BY downstream_chapter_id
                """,
                (report_id,),
            ).fetchall()
            for chapter in chapters:
                if not bool(chapter["requires_recheck"]):
                    connection.execute(
                        """
                        UPDATE novel_chapters
                        SET needs_recheck=0, updated_at=?
                        WHERE id=? AND project_id=?
                        """,
                        (
                            now,
                            chapter["downstream_chapter_id"],
                            report["project_id"],
                        ),
                    )
            connection.execute(
                """
                UPDATE canon_impact_reports
                SET status='applied', updated_at=?, decided_at=?
                WHERE id=?
                """,
                (now, now, report_id),
            )
            connection.commit()
        return True

    def cancel_report(
        self, *, user_id: int, report_id: str
    ) -> Optional[Dict[str, str]]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            report = connection.execute(
                """
                SELECT r.project_id, r.chapter_id
                FROM canon_impact_reports r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND p.user_id=? AND r.status='pending'
                """,
                (report_id, user_id),
            ).fetchone()
            if not report:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE canon_impact_reports
                SET status='canceled', updated_at=?, decided_at=?
                WHERE id=?
                """,
                (now, now, report_id),
            )
            connection.commit()
        return {
            "project_id": str(report["project_id"]),
            "chapter_id": str(report["chapter_id"]),
        }

    def mark_stale(
        self, *, user_id: int, report_id: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE canon_impact_reports
                SET status='stale', updated_at=?, decided_at=?
                WHERE id=? AND status='pending' AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=canon_impact_reports.project_id
                      AND p.user_id=?
                )
                """,
                (utc_now(), utc_now(), report_id, user_id),
            )
            connection.commit()
        return cursor.rowcount == 1
