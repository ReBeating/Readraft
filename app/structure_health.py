from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .db import Database
from .structure_link_schema import CAUSAL_RELATION_LABELS


STRUCTURE_ROLES = {
    "setup": "建立",
    "escalation": "升级",
    "reversal": "反转",
    "payoff": "兑现",
    "transition": "转场",
}
ARC_TYPE_LABELS = {
    "main": "主线",
    "subplot": "支线",
    "character": "人物线",
    "relationship": "关系线",
    "mystery": "悬疑线",
    "world": "世界线",
}
ARC_LIFECYCLE_LABELS = {
    "planned": "计划中",
    "active": "推进中",
    "paused": "暂停",
    "resolved": "已兑现",
    "abandoned": "已放弃",
}
SEVERITY_LABELS = {
    "blocking": "硬缺口",
    "warning": "待核对",
    "info": "观察项",
}
SEVERITY_ORDER = {"blocking": 0, "warning": 1, "info": 2}


def _load_list(raw: Any) -> List[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _key_points(raw: Any) -> List[str]:
    return [
        line.strip()
        for line in str(raw or "").splitlines()
        if line.strip()
    ]


def _application_option_label(raw: Any, option_index: int) -> str:
    try:
        payload = json.loads(str(raw or "{}"))
        options = payload.get("options") or []
        option = options[option_index]
        label = str(option.get("label") or "").strip()
    except (AttributeError, IndexError, TypeError, ValueError):
        label = ""
    return label


def _chapter_action(project_id: str, chapter_id: str) -> str:
    return (
        f"/novels/{project_id}/chapters/{chapter_id}/task-card"
    )


def _volume_action(project_id: str, volume_id: str) -> str:
    return (
        f"/novels/{project_id}/workbench"
        "?view=archive&archive_tab=creative&settings_tab=structure"
    )


def _arc_action(project_id: str, arc_id: str) -> str:
    return (
        f"/novels/{project_id}/workbench"
        "?view=archive&archive_tab=creative&settings_tab=structure"
    )


def _source_kind(application_id: str) -> str:
    return f"application:{application_id}" if application_id else "manual"


def _longest_uncovered_run(
    chapters: Sequence[Mapping[str, Any]],
    covered_positions: set[int],
) -> Dict[str, int]:
    best: Dict[str, int] = {"count": 0, "start": 0, "end": 0}
    start = 0
    previous = 0
    count = 0
    for chapter in chapters:
        position = int(chapter["position"])
        if (
            position in covered_positions
            or (previous and position != previous + 1)
        ):
            if count > best["count"]:
                best = {
                    "count": count,
                    "start": start,
                    "end": previous,
                }
            count = 0
            start = 0
        if position not in covered_positions:
            if not count:
                start = position
            count += 1
        previous = position
    if count > best["count"]:
        best = {"count": count, "start": start, "end": previous}
    return best


class StructureHealthService:
    """Build a deterministic, read-only audit of the planned book structure."""

    def __init__(self, database: Database):
        self.database = database

    def get_report(
        self, *, user_id: int, project_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            project_row = connection.execute(
                """
                SELECT p.*
                FROM novel_projects p
                WHERE p.id=? AND p.user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project_row:
                return None

            chapter_rows = connection.execute(
                """
                SELECT ch.id, ch.position, ch.title, ch.outline,
                       ch.key_points, ch.status, ch.volume_id,
                       ch.canonical_version_id, ch.needs_recheck,
                       ch.skeleton_role,
                       ch.skeleton_arc_titles_json,
                       ch.skeleton_ending_hook,
                       ch.skeleton_application_id,
                       v.position AS volume_position,
                       v.title AS volume_title,
                       cp.id AS task_card_id,
                       cp.status AS task_card_status,
                       cp.source AS task_card_source
                FROM novel_chapters ch
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
                WHERE ch.project_id=?
                ORDER BY ch.position
                """,
                (project_id,),
            ).fetchall()
            volume_rows = connection.execute(
                """
                SELECT v.*
                FROM novel_volumes v
                WHERE v.project_id=?
                ORDER BY v.position
                """,
                (project_id,),
            ).fetchall()
            blueprint_row = connection.execute(
                """
                SELECT confirmed_version_id
                FROM novel_story_blueprint_heads
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            arc_rows = connection.execute(
                """
                SELECT a.id AS arc_id, a.position AS arc_position,
                       version.arc_type, version.title,
                       version.lifecycle_status, version.priority,
                       version.dramatic_question, version.promise,
                       version.target_payoff
                FROM novel_plot_arcs a
                JOIN novel_plot_arc_versions version
                  ON version.id=a.confirmed_version_id
                WHERE a.project_id=?
                ORDER BY a.position
                """,
                (project_id,),
            ).fetchall()
            application_rows = connection.execute(
                """
                SELECT application.id, application.suggestion_id,
                       application.option_index, application.status,
                       application.created_at, application.reverted_at,
                       suggestion.result_json
                FROM story_structure_applications application
                JOIN story_structure_suggestions suggestion
                  ON suggestion.id=application.suggestion_id
                WHERE application.project_id=?
                ORDER BY application.created_at DESC
                """,
                (project_id,),
            ).fetchall()
            causal_link_rows = connection.execute(
                """
                SELECT link.*,
                       source.position AS source_position,
                       source.title AS source_title,
                       source.skeleton_role AS source_role,
                       source.skeleton_arc_titles_json
                           AS source_arcs_json,
                       source.canonical_version_id
                           AS source_canonical_version_id,
                       target.position AS target_position,
                       target.title AS target_title,
                       target.skeleton_role AS target_role,
                       target.skeleton_arc_titles_json
                           AS target_arcs_json,
                       target.canonical_version_id
                           AS target_canonical_version_id
                FROM novel_chapter_causal_links link
                JOIN novel_chapters source
                  ON source.id=link.source_chapter_id
                JOIN novel_chapters target
                  ON target.id=link.target_chapter_id
                WHERE link.project_id=? AND link.status='active'
                ORDER BY target.position, source.position, link.created_at
                """,
                (project_id,),
            ).fetchall()

        project = dict(project_row)
        chapters = self._decode_chapters(chapter_rows)
        volumes = [dict(row) for row in volume_rows]
        applications = self._decode_applications(application_rows)
        causal_links = self._decode_causal_links(causal_link_rows)
        application_map = {
            str(item["id"]): item for item in applications
        }
        self._label_chapter_sources(chapters, application_map)

        canonical_chapters = [
            item for item in chapters if item["is_canonical"]
        ]
        current_position = max(
            (int(item["position"]) for item in canonical_chapters),
            default=0,
        )
        shadow_chapters = [
            item
            for item in chapters
            if not item["is_canonical"]
            and int(item["position"]) <= current_position
        ]
        future_chapters = [
            item
            for item in chapters
            if not item["is_canonical"]
            and int(item["position"]) > current_position
        ]
        last_canonical = max(
            canonical_chapters,
            key=lambda item: int(item["position"]),
            default=None,
        )
        for link in causal_links:
            link["planning_status"] = (
                "realized"
                if link["target_is_canonical"]
                or int(link["target_position"]) <= current_position
                else "active"
            )
        planned_causal_links = [
            item
            for item in causal_links
            if item["planning_status"] == "active"
        ]

        arcs = self._decode_arcs(arc_rows)
        arcs_by_title: Dict[str, List[Dict[str, Any]]] = {}
        for arc in arcs:
            arcs_by_title.setdefault(str(arc["title"]), []).append(arc)

        findings: List[Dict[str, Any]] = []
        seen_findings: set[str] = set()

        def add_finding(
            *,
            code: str,
            key: str,
            severity: str,
            title: str,
            description: str,
            evidence: str,
            action_url: str,
            action_label: str,
            chapter_positions: Optional[List[int]] = None,
            chapter_ids: Optional[List[str]] = None,
            volume_id: str = "",
            arc_id: str = "",
        ) -> str:
            finding_id = f"{code}:{key}"
            if finding_id in seen_findings:
                return finding_id
            seen_findings.add(finding_id)
            findings.append(
                {
                    "id": finding_id,
                    "code": code,
                    "severity": severity,
                    "severity_label": SEVERITY_LABELS[severity],
                    "title": title,
                    "description": description,
                    "evidence": evidence,
                    "action_url": action_url,
                    "action_label": action_label,
                    "chapter_positions": chapter_positions or [],
                    "chapter_ids": chapter_ids or [],
                    "volume_id": volume_id,
                    "arc_id": arc_id,
                }
            )
            return finding_id

        if shadow_chapters:
            positions = [int(item["position"]) for item in shadow_chapters]
            add_finding(
                code="NON_CANON_BEHIND_BOUNDARY",
                key="-".join(str(item) for item in positions),
                severity="blocking",
                title="正史边界后方仍有未确认章节",
                description=(
                    "这些章节的位置不晚于当前最高正史章，却仍不是正史。"
                    "继续生成窗口前，需要先决定它们应进入正史、重排还是保留为草稿。"
                ),
                evidence="位置：" + "、".join(
                    f"{item:03d}" for item in positions
                ),
                action_url=_chapter_action(
                    project_id, str(shadow_chapters[0]["id"])
                ),
                action_label="检查最早一章",
                chapter_positions=positions,
                chapter_ids=[
                    str(item["id"]) for item in shadow_chapters
                ],
            )

        if future_chapters:
            self._audit_foundations(
                project_id=project_id,
                has_confirmed_blueprint=bool(
                    blueprint_row
                    and blueprint_row["confirmed_version_id"]
                ),
                arcs=arcs,
                arcs_by_title=arcs_by_title,
                add_finding=add_finding,
            )
            self._audit_position_and_volume_order(
                project_id=project_id,
                current_position=current_position,
                last_canonical=last_canonical,
                future_chapters=future_chapters,
                add_finding=add_finding,
            )
            self._audit_chapter_skeletons(
                project_id=project_id,
                future_chapters=future_chapters,
                arcs_by_title=arcs_by_title,
                add_finding=add_finding,
            )

        volume_health = self._audit_volumes(
            project_id=project_id,
            volumes=volumes,
            chapters=chapters,
            future_chapters=future_chapters,
            add_finding=add_finding,
        )
        self._audit_role_rhythm(
            project_id=project_id,
            future_chapters=future_chapters,
            add_finding=add_finding,
        )
        arc_health = self._audit_arc_coverage(
            project_id=project_id,
            arcs=arcs,
            future_chapters=future_chapters,
            add_finding=add_finding,
        )
        self._audit_causal_coverage(
            project_id=project_id,
            future_chapters=future_chapters,
            causal_links=planned_causal_links,
            add_finding=add_finding,
        )
        boundaries = self._audit_boundaries(
            project_id=project_id,
            last_canonical=last_canonical,
            future_chapters=future_chapters,
            causal_links=planned_causal_links,
            add_finding=add_finding,
        )

        findings.sort(
            key=lambda item: (
                SEVERITY_ORDER[str(item["severity"])],
                min(item["chapter_positions"] or [10**9]),
                str(item["code"]),
            )
        )
        finding_by_id = {
            str(item["id"]): item for item in findings
        }
        self._attach_finding_state(
            future_chapters=future_chapters,
            volume_health=volume_health,
            boundaries=boundaries,
            findings=findings,
            finding_by_id=finding_by_id,
        )

        for application in applications:
            covered = [
                int(item["position"])
                for item in future_chapters
                if item["skeleton_application_id"]
                == application["id"]
            ]
            application["covered_positions"] = covered
            application["covered_count"] = len(covered)

        counts = {
            "blocking": sum(
                item["severity"] == "blocking" for item in findings
            ),
            "warning": sum(
                item["severity"] == "warning" for item in findings
            ),
            "info": sum(
                item["severity"] == "info" for item in findings
            ),
            "canonical_chapters": len(canonical_chapters),
            "future_chapters": len(future_chapters),
            "window_boundaries": len(boundaries),
            "confirmed_arcs": len(arcs),
            "causal_links": len(planned_causal_links),
            "cross_line_causal_links": sum(
                bool(item["cross_line"])
                for item in planned_causal_links
            ),
        }
        if not future_chapters and not shadow_chapters:
            report_status = "empty"
        elif counts["blocking"]:
            report_status = "blocked"
        elif counts["warning"]:
            report_status = "attention"
        else:
            report_status = "healthy"

        return {
            "project": project,
            "status": report_status,
            "status_label": {
                "empty": "尚未规划",
                "blocked": "先补硬缺口",
                "attention": "可续写，建议先核对",
                "healthy": "结构证据完整",
            }[report_status],
            "scope": {
                "current_canonical_position": current_position,
                "first_future_position": (
                    int(future_chapters[0]["position"])
                    if future_chapters
                    else None
                ),
                "last_future_position": (
                    int(future_chapters[-1]["position"])
                    if future_chapters
                    else None
                ),
                "next_window_position": (
                    int(future_chapters[-1]["position"]) + 1
                    if future_chapters
                    else current_position + 1
                ),
            },
            "counts": counts,
            "findings": findings,
            "chapters": future_chapters,
            "shadow_chapters": shadow_chapters,
            "volumes": volume_health,
            "arcs": arc_health,
            "boundaries": boundaries,
            "causal_links": causal_links,
            "causal_source_chapters": [
                {
                    "id": str(item["id"]),
                    "position": int(item["position"]),
                    "title": str(item["title"]),
                    "is_canonical": bool(item["is_canonical"]),
                    "arc_titles": list(item["arc_titles"]),
                }
                for item in chapters
                if (
                    item["is_canonical"]
                    or int(item["position"]) > current_position
                )
                and int(item["position"])
                < int(
                    future_chapters[-1]["position"]
                    if future_chapters
                    else 0
                )
            ],
            "causal_target_chapters": [
                {
                    "id": str(item["id"]),
                    "position": int(item["position"]),
                    "title": str(item["title"]),
                    "arc_titles": list(item["arc_titles"]),
                }
                for item in future_chapters
            ],
            "causal_relation_options": [
                {"value": value, "label": label}
                for value, label in CAUSAL_RELATION_LABELS.items()
            ],
            "applications": applications,
            "has_confirmed_blueprint": bool(
                blueprint_row
                and blueprint_row["confirmed_version_id"]
            ),
        }

    @staticmethod
    def _decode_chapters(rows) -> List[Dict[str, Any]]:
        chapters: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["arc_titles"] = _load_list(
                item.pop("skeleton_arc_titles_json", "[]")
            )
            item["key_point_items"] = _key_points(
                item.get("key_points")
            )
            item["role"] = str(item.get("skeleton_role") or "")
            item["role_label"] = STRUCTURE_ROLES.get(
                item["role"], item["role"] or "待定义"
            )
            item["ending_hook"] = str(
                item.get("skeleton_ending_hook") or ""
            ).strip()
            item["skeleton_application_id"] = str(
                item.get("skeleton_application_id") or ""
            )
            item["is_canonical"] = bool(
                item.get("canonical_version_id")
            )
            item["finding_ids"] = []
            item["health_severity"] = ""
            chapters.append(item)
        return chapters

    @staticmethod
    def _decode_arcs(rows) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item.pop("arc_id"))
            item["position"] = int(item.pop("arc_position"))
            item["title"] = str(item.get("title") or "").strip()
            item["priority"] = int(item.get("priority") or 0)
            item["arc_type_label"] = ARC_TYPE_LABELS.get(
                str(item.get("arc_type") or ""),
                str(item.get("arc_type") or ""),
            )
            item["lifecycle_label"] = ARC_LIFECYCLE_LABELS.get(
                str(item.get("lifecycle_status") or ""),
                str(item.get("lifecycle_status") or ""),
            )
            result.append(item)
        return result

    @staticmethod
    def _decode_applications(rows) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            option_index = int(item["option_index"])
            option_label = _application_option_label(
                item.pop("result_json", "{}"), option_index
            )
            item["option_label"] = option_label
            item["source_label"] = (
                option_label
                or f"结构采纳 {str(item['id'])[:6].upper()}"
            )
            result.append(item)
        return result

    @staticmethod
    def _decode_causal_links(rows) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            source_arcs = _load_list(
                item.pop("source_arcs_json", "[]")
            )
            target_arcs = _load_list(
                item.pop("target_arcs_json", "[]")
            )
            shared_arcs = sorted(set(source_arcs) & set(target_arcs))
            item["source_arc_titles"] = source_arcs
            item["target_arc_titles"] = target_arcs
            item["shared_arc_titles"] = shared_arcs
            item["cross_line"] = bool(
                source_arcs and target_arcs and not shared_arcs
            )
            item["relation_label"] = CAUSAL_RELATION_LABELS.get(
                str(item.get("relation_type") or ""),
                str(item.get("relation_type") or ""),
            )
            item["source_is_canonical"] = bool(
                item.pop("source_canonical_version_id", None)
            )
            item["target_is_canonical"] = bool(
                item.pop("target_canonical_version_id", None)
            )
            result.append(item)
        return result

    @staticmethod
    def _label_chapter_sources(
        chapters: List[Dict[str, Any]],
        application_map: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for chapter in chapters:
            application_id = str(
                chapter["skeleton_application_id"] or ""
            )
            chapter["source_kind"] = _source_kind(application_id)
            if not application_id:
                chapter["source_label"] = "作者手动骨架"
                continue
            application = application_map.get(application_id)
            chapter["source_label"] = (
                str(application["source_label"])
                if application
                else f"结构窗口 {application_id[:6].upper()}"
            )

    @staticmethod
    def _audit_foundations(
        *,
        project_id: str,
        has_confirmed_blueprint: bool,
        arcs: List[Dict[str, Any]],
        arcs_by_title: Mapping[str, List[Dict[str, Any]]],
        add_finding,
    ) -> None:
        if not has_confirmed_blueprint:
            add_finding(
                code="CONFIRMED_BLUEPRINT_MISSING",
                key="project",
                severity="blocking",
                title="未来骨架没有确认版全书蓝图作基线",
                description=(
                    "草稿蓝图不会进入后续 Planner。先确认核心悬问、"
                    "冲突引擎、终局状态和必须兑现项，再继续扩展窗口。"
                ),
                evidence="当前项目没有 confirmed blueprint",
                action_url=(
                    f"/novels/{project_id}/workbench"
                    "?view=archive&archive_tab=creative&settings_tab=structure"
                ),
                action_label="检查全书蓝图",
            )
        active_main = [
            arc
            for arc in arcs
            if arc["arc_type"] == "main"
            and arc["lifecycle_status"] in {"planned", "active"}
        ]
        if not active_main:
            add_finding(
                code="ACTIVE_MAIN_ARC_MISSING",
                key="project",
                severity="blocking",
                title="没有可推进的确认主线",
                description=(
                    "未来章节必须引用作者确认、且处于计划中或推进中的主线；"
                    "暂停、已兑现和已放弃的线不能充当新窗口发动机。"
                ),
                evidence="确认剧情线中未找到 planned / active main",
                action_url=(
                    f"/novels/{project_id}/workbench"
                    "?view=archive&archive_tab=creative&settings_tab=structure"
                ),
                action_label="检查规划剧情线",
            )
        duplicate_titles = [
            title
            for title, matches in arcs_by_title.items()
            if title and len(matches) > 1
        ]
        if duplicate_titles:
            add_finding(
                code="DUPLICATE_CONFIRMED_ARC_TITLE",
                key="|".join(duplicate_titles),
                severity="blocking",
                title="确认剧情线存在重名，章节引用无法唯一定位",
                description=(
                    "章节骨架用精确标题绑定剧情线。重名会让 Writer "
                    "无法判断应该读取哪条承诺和目标回报。"
                ),
                evidence="重名：" + "、".join(duplicate_titles),
                action_url=(
                    f"/novels/{project_id}/workbench"
                    "?view=archive&archive_tab=creative&settings_tab=structure"
                ),
                action_label="为剧情线改成唯一标题",
            )

    @staticmethod
    def _audit_position_and_volume_order(
        *,
        project_id: str,
        current_position: int,
        last_canonical: Optional[Mapping[str, Any]],
        future_chapters: List[Dict[str, Any]],
        add_finding,
    ) -> None:
        expected = current_position + 1
        for chapter in future_chapters:
            position = int(chapter["position"])
            if position != expected:
                add_finding(
                    code="FUTURE_POSITION_GAP",
                    key=f"{expected}-{position}",
                    severity="blocking",
                    title="正史之后的章节位置不连续",
                    description=(
                        "滚动窗口必须从正史下一章连续展开，否则前后任务卡、"
                        "记忆边界和下一窗口起点都会产生歧义。"
                    ),
                    evidence=(
                        f"应接第 {expected:03d} 章，实际下一章为 "
                        f"{position:03d}"
                    ),
                    action_url=_chapter_action(
                        project_id, str(chapter["id"])
                    ),
                    action_label="检查断点后的章节",
                    chapter_positions=[position],
                    chapter_ids=[str(chapter["id"])],
                )
            expected = position + 1

        first_future = future_chapters[0]
        first_volume = first_future.get("volume_position")
        canonical_volume = (
            last_canonical.get("volume_position")
            if last_canonical
            else None
        )
        if first_volume is not None and canonical_volume is None:
            first_volume_position = int(first_volume)
            if first_volume_position != 1:
                add_finding(
                    code="VOLUME_SEQUENCE_START",
                    key=str(first_future["position"]),
                    severity="blocking",
                    title="未来结构从第 1 卷之后开始，却没有前卷承载正史",
                    description=(
                        "没有可作为基线的正史分卷时，第一段章节应从第 1 卷"
                        "开始；否则前面的卷位没有章节承载阶段目标。"
                    ),
                    evidence=(
                        f"第 {int(first_future['position']):03d} 章"
                        f"直接归入第 {first_volume_position} 卷"
                    ),
                    action_url=_chapter_action(
                        project_id, str(first_future["id"])
                    ),
                    action_label="修订首章卷归属",
                    chapter_positions=[
                        int(first_future["position"])
                    ],
                    chapter_ids=[str(first_future["id"])],
                )

        prior = last_canonical
        for chapter in future_chapters:
            if prior is not None:
                left_volume = prior.get("volume_position")
                right_volume = chapter.get("volume_position")
                if left_volume is not None and right_volume is not None:
                    left_position = int(left_volume)
                    right_position = int(right_volume)
                    if right_position < left_position:
                        add_finding(
                            code="VOLUME_ORDER_REGRESSION",
                            key=(
                                f"{prior['position']}-"
                                f"{chapter['position']}"
                            ),
                            severity="blocking",
                            title="章节顺序回退到了更早分卷",
                            description=(
                                "分卷位置必须随章节单调前进；窗口不能从后卷"
                                "重新跳回前卷。"
                            ),
                            evidence=(
                                f"第 {int(prior['position']):03d} 章在第 "
                                f"{left_position} 卷，第 "
                                f"{int(chapter['position']):03d} 章回到第 "
                                f"{right_position} 卷"
                            ),
                            action_url=_chapter_action(
                                project_id, str(chapter["id"])
                            ),
                            action_label="修订后一个章节",
                            chapter_positions=[
                                int(prior["position"]),
                                int(chapter["position"]),
                            ],
                            chapter_ids=[
                                str(prior["id"]),
                                str(chapter["id"]),
                            ],
                        )
                    elif right_position > left_position + 1:
                        add_finding(
                            code="VOLUME_POSITION_GAP",
                            key=(
                                f"{prior['position']}-"
                                f"{chapter['position']}"
                            ),
                            severity="blocking",
                            title="章节跨过了未承载内容的分卷位置",
                            description=(
                                "连续章节最多进入下一卷；直接跨过一个或多个"
                                "卷位会让阶段目标与回报失去承载范围。"
                            ),
                            evidence=(
                                f"第 {left_position} 卷直接跳到第 "
                                f"{right_position} 卷"
                            ),
                            action_url=_chapter_action(
                                project_id, str(chapter["id"])
                            ),
                            action_label="检查卷归属",
                            chapter_positions=[
                                int(prior["position"]),
                                int(chapter["position"]),
                            ],
                            chapter_ids=[
                                str(prior["id"]),
                                str(chapter["id"]),
                            ],
                        )
            prior = chapter

    @staticmethod
    def _audit_chapter_skeletons(
        *,
        project_id: str,
        future_chapters: List[Dict[str, Any]],
        arcs_by_title: Mapping[str, List[Dict[str, Any]]],
        add_finding,
    ) -> None:
        for chapter in future_chapters:
            position = int(chapter["position"])
            chapter_id = str(chapter["id"])
            action_url = _chapter_action(project_id, chapter_id)
            role = str(chapter["role"])
            if role not in STRUCTURE_ROLES:
                add_finding(
                    code="CHAPTER_ROLE_MISSING",
                    key=str(position),
                    severity="blocking",
                    title=f"第 {position:03d} 章没有有效结构职责",
                    description=(
                        "每章必须明确承担建立、升级、反转、兑现或转场之一，"
                        "才能判断连续节奏是否真的推进。"
                    ),
                    evidence=f"当前值：{role or '空'}",
                    action_url=action_url,
                    action_label="补全章节职责",
                    chapter_positions=[position],
                    chapter_ids=[chapter_id],
                )
            if not str(chapter.get("outline") or "").strip():
                add_finding(
                    code="CHAPTER_PURPOSE_MISSING",
                    key=str(position),
                    severity="blocking",
                    title=f"第 {position:03d} 章缺少可核对的推进目的",
                    description=(
                        "章节目的要说明人物采取什么行动，以及局面发生"
                        "什么不可逆变化；只有标题不能支撑任务卡。"
                    ),
                    evidence="章节 purpose / outline 为空",
                    action_url=action_url,
                    action_label="补全章节目的",
                    chapter_positions=[position],
                    chapter_ids=[chapter_id],
                )
            point_count = len(chapter["key_point_items"])
            if point_count < 2:
                add_finding(
                    code="CHAPTER_KEY_POINTS_INCOMPLETE",
                    key=str(position),
                    severity="blocking",
                    title=f"第 {position:03d} 章关键推进点不足",
                    description=(
                        "滚动骨架至少需要两个可验证推进点，才能继续拆成"
                        "场景阻力、行动和结果。"
                    ),
                    evidence=f"当前只有 {point_count} 条关键推进点",
                    action_url=action_url,
                    action_label="补全关键推进点",
                    chapter_positions=[position],
                    chapter_ids=[chapter_id],
                )
            if not chapter.get("volume_id"):
                add_finding(
                    code="CHAPTER_VOLUME_MISSING",
                    key=str(position),
                    severity="blocking",
                    title=f"第 {position:03d} 章尚未归入分卷",
                    description=(
                        "没有卷归属就无法核对阶段目标、卷内反转和阶段回报。"
                    ),
                    evidence="volume_id 为空",
                    action_url=action_url,
                    action_label="选择所属分卷",
                    chapter_positions=[position],
                    chapter_ids=[chapter_id],
                )
            arc_titles = list(chapter["arc_titles"])
            if not arc_titles:
                add_finding(
                    code="CHAPTER_ARC_MISSING",
                    key=str(position),
                    severity="blocking",
                    title=f"第 {position:03d} 章没有绑定精确剧情线",
                    description=(
                        "Writer 只读取本章明确选择的确认剧情线。"
                        "缺少绑定会让章节目标失去长期承诺来源。"
                    ),
                    evidence="skeleton_arc_titles 为空",
                    action_url=action_url,
                    action_label="选择本章推进线",
                    chapter_positions=[position],
                    chapter_ids=[chapter_id],
                )
            else:
                unknown = [
                    title
                    for title in arc_titles
                    if title not in arcs_by_title
                ]
                if unknown:
                    add_finding(
                        code="CHAPTER_ARC_UNKNOWN",
                        key=str(position),
                        severity="blocking",
                        title=(
                            f"第 {position:03d} 章引用了未确认或不存在的剧情线"
                        ),
                        description=(
                            "骨架必须用精确标题引用作者确认的剧情线；"
                            "草稿名、旧名和自由文本都不会进入后续上下文。"
                        ),
                        evidence="无法定位：" + "、".join(unknown),
                        action_url=action_url,
                        action_label="改用确认剧情线",
                        chapter_positions=[position],
                        chapter_ids=[chapter_id],
                    )
                inactive = sorted(
                    {
                        title
                        for title in arc_titles
                        for arc in arcs_by_title.get(title, [])
                        if arc["lifecycle_status"]
                        not in {"planned", "active"}
                    }
                )
                if inactive:
                    add_finding(
                        code="CHAPTER_ARC_INACTIVE",
                        key=str(position),
                        severity="warning",
                        title=(
                            f"第 {position:03d} 章仍在推进已暂停或已结束的剧情线"
                        ),
                        description=(
                            "如果是有意重启，请先修改并确认剧情线生命周期；"
                            "否则应把章节绑定改到当前可推进的线。"
                        ),
                        evidence="当前不可推进：" + "、".join(inactive),
                        action_url=action_url,
                        action_label="核对剧情线绑定",
                        chapter_positions=[position],
                        chapter_ids=[chapter_id],
                    )
            if not chapter["ending_hook"]:
                add_finding(
                    code="CHAPTER_HOOK_MISSING",
                    key=str(position),
                    severity="warning",
                    title=f"第 {position:03d} 章缺少章末推动",
                    description=(
                        "章末不必强行悬崖，但需要留下新的证据、后果、"
                        "选择或明确下一目标，供下一章承接。"
                    ),
                    evidence="skeleton_ending_hook 为空",
                    action_url=action_url,
                    action_label="补全章末推动",
                    chapter_positions=[position],
                    chapter_ids=[chapter_id],
                )

    @staticmethod
    def _audit_volumes(
        *,
        project_id: str,
        volumes: List[Dict[str, Any]],
        chapters: List[Dict[str, Any]],
        future_chapters: List[Dict[str, Any]],
        add_finding,
    ) -> List[Dict[str, Any]]:
        all_by_volume: Dict[str, List[Dict[str, Any]]] = {}
        future_by_volume: Dict[str, List[Dict[str, Any]]] = {}
        for chapter in chapters:
            if chapter.get("volume_id"):
                all_by_volume.setdefault(
                    str(chapter["volume_id"]), []
                ).append(chapter)
        for chapter in future_chapters:
            if chapter.get("volume_id"):
                future_by_volume.setdefault(
                    str(chapter["volume_id"]), []
                ).append(chapter)

        result: List[Dict[str, Any]] = []
        for raw_volume in volumes:
            volume = dict(raw_volume)
            volume_id = str(volume["id"])
            all_items = all_by_volume.get(volume_id, [])
            future_items = future_by_volume.get(volume_id, [])
            if not future_items:
                continue
            volume["chapter_positions"] = [
                int(item["position"]) for item in all_items
            ]
            volume["future_positions"] = [
                int(item["position"]) for item in future_items
            ]
            volume["canonical_count"] = sum(
                item["is_canonical"] for item in all_items
            )
            volume["future_count"] = len(future_items)
            volume["role_counts"] = dict(
                Counter(str(item["role"]) for item in future_items)
            )
            volume["finding_ids"] = []
            missing_fields = [
                label
                for field, label in (
                    ("goal", "本卷目标"),
                    ("start_state", "开卷状态"),
                    ("end_state", "收卷状态"),
                    ("major_conflict", "主要冲突"),
                    ("payoff", "本卷回报"),
                )
                if not str(volume.get(field) or "").strip()
            ]
            if missing_fields:
                finding_id = add_finding(
                    code="VOLUME_METADATA_INCOMPLETE",
                    key=volume_id,
                    severity="warning",
                    title=(
                        f"第 {int(volume['position'])} 卷阶段证据不完整"
                    ),
                    description=(
                        "卷级目标用于判断这些章节是否在同一阶段内累积，"
                        "缺项会削弱下一窗口的分卷衔接。"
                    ),
                    evidence="待补：" + "、".join(missing_fields),
                    action_url=_volume_action(project_id, volume_id),
                    action_label="编辑分卷目标",
                    chapter_positions=volume["future_positions"],
                    chapter_ids=[
                        str(item["id"]) for item in future_items
                    ],
                    volume_id=volume_id,
                )
                volume["finding_ids"].append(finding_id)
            if (
                len(future_items) >= 3
                and not any(
                    item["role"] == "payoff"
                    for item in future_items
                )
            ):
                finding_id = add_finding(
                    code="VOLUME_PAYOFF_MISSING",
                    key=volume_id,
                    severity="warning",
                    title=(
                        f"第 {int(volume['position'])} 卷没有规划兑现章"
                    ),
                    description=(
                        "当前未来范围已在本卷安排至少三章，却没有一章"
                        "明确承担兑现。读者可能只看到持续加压而没有阶段回报。"
                    ),
                    evidence=(
                        f"未来 {len(future_items)} 章职责中没有 payoff"
                    ),
                    action_url=_chapter_action(
                        project_id, str(future_items[-1]["id"])
                    ),
                    action_label="检查本卷后段",
                    chapter_positions=volume["future_positions"],
                    chapter_ids=[
                        str(item["id"]) for item in future_items
                    ],
                    volume_id=volume_id,
                )
                volume["finding_ids"].append(finding_id)
            if (
                len(future_items) >= 5
                and not any(
                    item["role"] == "reversal"
                    for item in future_items
                )
            ):
                finding_id = add_finding(
                    code="VOLUME_REVERSAL_MISSING",
                    key=volume_id,
                    severity="warning",
                    title=(
                        f"第 {int(volume['position'])} 卷缺少认知或局势反转"
                    ),
                    description=(
                        "五章以上的阶段如果只有线性推进，调查、升级或关系"
                        "变化容易显得重复。反转应改变解释、策略或代价。"
                    ),
                    evidence=(
                        f"未来 {len(future_items)} 章职责中没有 reversal"
                    ),
                    action_url=_chapter_action(
                        project_id, str(future_items[len(future_items) // 2]["id"])
                    ),
                    action_label="检查本卷中段",
                    chapter_positions=volume["future_positions"],
                    chapter_ids=[
                        str(item["id"]) for item in future_items
                    ],
                    volume_id=volume_id,
                )
                volume["finding_ids"].append(finding_id)
            result.append(volume)
        return result

    @staticmethod
    def _audit_role_rhythm(
        *,
        project_id: str,
        future_chapters: List[Dict[str, Any]],
        add_finding,
    ) -> None:
        if not future_chapters:
            return
        runs: List[List[Dict[str, Any]]] = []
        current_run: List[Dict[str, Any]] = []
        for chapter in future_chapters:
            if (
                current_run
                and (
                    chapter["role"] != current_run[-1]["role"]
                    or int(chapter["position"])
                    != int(current_run[-1]["position"]) + 1
                )
            ):
                runs.append(current_run)
                current_run = []
            current_run.append(chapter)
        if current_run:
            runs.append(current_run)
        for run in runs:
            role = str(run[0]["role"])
            positions = [int(item["position"]) for item in run]
            if role in STRUCTURE_ROLES and len(run) >= 4:
                add_finding(
                    code="ROLE_RUN_TOO_LONG",
                    key=f"{role}-{positions[0]}-{positions[-1]}",
                    severity="warning",
                    title=(
                        f"连续 {len(run)} 章都承担“"
                        f"{STRUCTURE_ROLES[role]}”职责"
                    ),
                    description=(
                        "相同结构职责连续过久，容易让章节虽有事件却没有"
                        "阅读节奏变化。至少检查其中一章能否承担反转或兑现。"
                    ),
                    evidence=(
                        f"第 {positions[0]:03d}–{positions[-1]:03d} 章"
                    ),
                    action_url=_chapter_action(
                        project_id, str(run[len(run) // 2]["id"])
                    ),
                    action_label="检查连续段中部",
                    chapter_positions=positions,
                    chapter_ids=[str(item["id"]) for item in run],
                )
            if role == "transition" and len(run) >= 2:
                add_finding(
                    code="TRANSITION_RUN",
                    key=f"{positions[0]}-{positions[-1]}",
                    severity="warning",
                    title="连续转场章可能让故事失去发动机",
                    description=(
                        "转场用于重新定位人物、目标和局势；连续两章都只转场，"
                        "通常意味着至少一章缺少明确行动结果。"
                    ),
                    evidence=(
                        f"第 {positions[0]:03d}–{positions[-1]:03d} 章"
                    ),
                    action_url=_chapter_action(
                        project_id, str(run[-1]["id"])
                    ),
                    action_label="检查第二个转场",
                    chapter_positions=positions,
                    chapter_ids=[str(item["id"]) for item in run],
                )

        without_turn: List[Dict[str, Any]] = []
        no_turn_runs: List[List[Dict[str, Any]]] = []
        for chapter in future_chapters:
            if (
                chapter["role"] in {"reversal", "payoff"}
                or (
                    without_turn
                    and int(chapter["position"])
                    != int(without_turn[-1]["position"]) + 1
                )
            ):
                if without_turn:
                    no_turn_runs.append(without_turn)
                without_turn = []
            if chapter["role"] not in {"reversal", "payoff"}:
                without_turn.append(chapter)
        if without_turn:
            no_turn_runs.append(without_turn)
        for run in no_turn_runs:
            if len(run) < 6:
                continue
            positions = [int(item["position"]) for item in run]
            add_finding(
                code="TURNING_POINT_GAP",
                key=f"{positions[0]}-{positions[-1]}",
                severity="warning",
                title=f"连续 {len(run)} 章没有反转或兑现",
                description=(
                    "长段只有建立、升级或转场时，信息可能增加，"
                    "但读者对局势的理解与回报预期没有发生结构变化。"
                ),
                evidence=(
                    f"第 {positions[0]:03d}–{positions[-1]:03d} 章"
                ),
                action_url=_chapter_action(
                    project_id, str(run[len(run) // 2]["id"])
                ),
                action_label="检查连续段中部",
                chapter_positions=positions,
                chapter_ids=[str(item["id"]) for item in run],
            )

    @staticmethod
    def _audit_arc_coverage(
        *,
        project_id: str,
        arcs: List[Dict[str, Any]],
        future_chapters: List[Dict[str, Any]],
        add_finding,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for raw_arc in arcs:
            arc = dict(raw_arc)
            positions = [
                int(chapter["position"])
                for chapter in future_chapters
                if arc["title"] in chapter["arc_titles"]
            ]
            covered = set(positions)
            gap = _longest_uncovered_run(
                future_chapters, covered
            )
            arc["chapter_positions"] = positions
            arc["touch_count"] = len(positions)
            arc["longest_gap"] = gap
            arc["finding_ids"] = []
            active = arc["lifecycle_status"] in {
                "planned",
                "active",
            }
            if active and future_chapters and not positions:
                severity = (
                    "warning"
                    if arc["arc_type"] == "main"
                    or int(arc["priority"]) >= 4
                    or arc["lifecycle_status"] == "active"
                    else "info"
                )
                finding_id = add_finding(
                    code="ARC_NOT_ADVANCED",
                    key=str(arc["id"]),
                    severity=severity,
                    title=f"“{arc['title']}”未进入任何未来章节",
                    description=(
                        "这条确认剧情线处于可推进状态，却没有被当前全部"
                        "未来骨架引用。请决定是安排一次推进，还是主动暂停。"
                    ),
                    evidence=(
                        f"{arc['arc_type_label']} · "
                        f"{arc['lifecycle_label']} · "
                        f"优先级 {arc['priority']}"
                    ),
                    action_url=_arc_action(
                        project_id, str(arc["id"])
                    ),
                    action_label="检查剧情线计划",
                    arc_id=str(arc["id"]),
                )
                arc["finding_ids"].append(finding_id)
                arc["coverage_status"] = "missing"
            elif (
                arc["arc_type"] == "main"
                and arc["lifecycle_status"] == "active"
                and int(gap["count"]) >= 6
            ):
                finding_id = add_finding(
                    code="ACTIVE_MAIN_ARC_GAP",
                    key=(
                        f"{arc['id']}-{gap['start']}-{gap['end']}"
                    ),
                    severity="warning",
                    title=(
                        f"推进中的主线“{arc['title']}”连续 "
                        f"{gap['count']} 章没有动作"
                    ),
                    description=(
                        "支线可以阶段性接管焦点，但推进中的主线长时间"
                        "完全不触达，会削弱读者对核心悬问的追踪。"
                    ),
                    evidence=(
                        f"第 {gap['start']:03d}–{gap['end']:03d} 章"
                    ),
                    action_url=_chapter_action(
                        project_id,
                        str(
                            next(
                                item["id"]
                                for item in future_chapters
                                if int(item["position"])
                                == int(gap["start"])
                            )
                        ),
                    ),
                    action_label="检查主线空窗起点",
                    chapter_positions=list(
                        range(int(gap["start"]), int(gap["end"]) + 1)
                    ),
                    arc_id=str(arc["id"]),
                )
                arc["finding_ids"].append(finding_id)
                arc["coverage_status"] = "thin"
            else:
                arc["coverage_status"] = (
                    "covered" if positions else "inactive"
                )
            result.append(arc)
        return result

    @staticmethod
    def _audit_causal_coverage(
        *,
        project_id: str,
        future_chapters: List[Dict[str, Any]],
        causal_links: List[Dict[str, Any]],
        add_finding,
    ) -> None:
        incoming_targets = {
            str(item["target_chapter_id"]) for item in causal_links
        }
        for chapter in future_chapters:
            if chapter["role"] not in {"reversal", "payoff"}:
                continue
            if str(chapter["id"]) in incoming_targets:
                continue
            add_finding(
                code="CAUSAL_BRIDGE_MISSING",
                key=str(chapter["id"]),
                severity="info",
                title=(
                    f"第 {int(chapter['position']):03d} 章"
                    f"{chapter['role_label']}尚无显式起因"
                ),
                description=(
                    "这不是硬性错误；若该变化依赖更早章节的行动，"
                    "可建立作者确认的因果链接，让 Planner 与 Writer "
                    "明确承接，而不是凭语感补造原因。"
                ),
                evidence=(
                    f"《{chapter['title']}》当前结构职责为"
                    f"{chapter['role_label']}"
                ),
                action_url=(
                    f"/novels/{project_id}/workbench"
                    "?view=archive&archive_tab=creative&settings_tab=structure"
                ),
                action_label="建立因果链接",
                chapter_positions=[int(chapter["position"])],
                chapter_ids=[str(chapter["id"])],
            )

    @staticmethod
    def _audit_boundaries(
        *,
        project_id: str,
        last_canonical: Optional[Dict[str, Any]],
        future_chapters: List[Dict[str, Any]],
        causal_links: List[Dict[str, Any]],
        add_finding,
    ) -> List[Dict[str, Any]]:
        pairs: List[tuple[Dict[str, Any], Dict[str, Any], str]] = []
        if (
            last_canonical
            and future_chapters
            and int(future_chapters[0]["position"])
            == int(last_canonical["position"]) + 1
        ):
            pairs.append(
                (last_canonical, future_chapters[0], "canon_to_future")
            )
        for left, right in zip(
            future_chapters, future_chapters[1:]
        ):
            if int(right["position"]) != int(left["position"]) + 1:
                continue
            if left["source_kind"] != right["source_kind"]:
                pairs.append((left, right, "window_change"))

        boundaries: List[Dict[str, Any]] = []
        for left, right, boundary_kind in pairs:
            left_position = int(left["position"])
            right_position = int(right["position"])
            left_source = (
                "已确认正史"
                if boundary_kind == "canon_to_future"
                else str(left["source_label"])
            )
            right_source = str(right["source_label"])
            if boundary_kind == "canon_to_future":
                boundary_label = "正史 → 当前未来骨架"
            elif (
                left["source_kind"].startswith("application:")
                and right["source_kind"] == "manual"
            ):
                boundary_label = "结构方案 → 作者接管"
            elif (
                left["source_kind"] == "manual"
                and right["source_kind"].startswith("application:")
            ):
                boundary_label = "作者骨架 → 新结构窗口"
            elif (
                left["source_kind"].startswith("application:")
                and right["source_kind"].startswith("application:")
            ):
                boundary_label = "结构窗口切换"
            else:
                boundary_label = "骨架来源切换"

            shared_arcs = sorted(
                set(left["arc_titles"]) & set(right["arc_titles"])
            )
            causal_bridges = [
                item
                for item in causal_links
                if int(item["source_position"]) <= left_position
                and int(item["target_position"]) >= right_position
            ]
            same_volume = bool(
                left.get("volume_id")
                and left.get("volume_id") == right.get("volume_id")
            )
            finding_ids: List[str] = []
            if boundary_kind != "canon_to_future":
                if (
                    same_volume
                    and left["arc_titles"]
                    and right["arc_titles"]
                    and not shared_arcs
                    and not causal_bridges
                ):
                    finding_ids.append(
                        add_finding(
                            code="WINDOW_ARC_HARD_CUT",
                            key=f"{left_position}-{right_position}",
                            severity="warning",
                            title=(
                                f"第 {left_position:03d} → "
                                f"{right_position:03d} 章同卷却完全换线"
                            ),
                            description=(
                                "相邻窗口在同一卷内没有共享剧情线。"
                                "如果这是有意切线，应让前章后果或后章行动"
                                "明确说明为什么此刻换焦点。"
                            ),
                            evidence=(
                                "前章："
                                + "、".join(left["arc_titles"])
                                + "；后章："
                                + "、".join(right["arc_titles"])
                            ),
                            action_url=_chapter_action(
                                project_id, str(right["id"])
                            ),
                            action_label="检查后一个窗口起点",
                            chapter_positions=[
                                left_position,
                                right_position,
                            ],
                            chapter_ids=[
                                str(left["id"]),
                                str(right["id"]),
                            ],
                        )
                    )
                if same_volume and right["role"] == "setup":
                    finding_ids.append(
                        add_finding(
                            code="WINDOW_RESTARTS_SAME_VOLUME",
                            key=f"{left_position}-{right_position}",
                            severity="warning",
                            title=(
                                f"第 {right_position:03d} 章在同卷窗口边界"
                                "重新“建立”"
                            ),
                            description=(
                                "同卷进入新窗口通常应承接既有行动后果。"
                                "重新建立并非一定错误，但要避免把已经建立的"
                                "目标、关系或谜面再次介绍一遍。"
                            ),
                            evidence=(
                                f"第 {left_position:03d} 与 "
                                f"{right_position:03d} 章属于同一卷"
                            ),
                            action_url=_chapter_action(
                                project_id, str(right["id"])
                            ),
                            action_label="核对新窗口首章",
                            chapter_positions=[
                                left_position,
                                right_position,
                            ],
                            chapter_ids=[
                                str(left["id"]),
                                str(right["id"]),
                            ],
                        )
                    )
                if (
                    left.get("volume_position") is not None
                    and right.get("volume_position") is not None
                    and int(right["volume_position"])
                    == int(left["volume_position"]) + 1
                    and left["role"] not in {"payoff", "transition"}
                ):
                    finding_ids.append(
                        add_finding(
                            code="WINDOW_CROSSES_VOLUME_WITHOUT_CLOSE",
                            key=f"{left_position}-{right_position}",
                            severity="warning",
                            title="新结构窗口跨卷，但前卷未以兑现或转场收束",
                            description=(
                                "跨卷前章仍承担建立、升级或反转时，阶段回报"
                                "可能被卷界截断。请核对是否应调整卷界或前章职责。"
                            ),
                            evidence=(
                                f"第 {left['volume_position']} 卷 "
                                f"{left['role_label']} → 第 "
                                f"{right['volume_position']} 卷 "
                                f"{right['role_label']}"
                            ),
                            action_url=_chapter_action(
                                project_id, str(left["id"])
                            ),
                            action_label="检查前卷收束章",
                            chapter_positions=[
                                left_position,
                                right_position,
                            ],
                            chapter_ids=[
                                str(left["id"]),
                                str(right["id"]),
                            ],
                        )
                    )

            severity = "warning" if finding_ids else "healthy"
            boundaries.append(
                {
                    "id": f"{left_position}-{right_position}",
                    "kind": boundary_kind,
                    "label": boundary_label,
                    "left": {
                        "id": str(left["id"]),
                        "position": left_position,
                        "title": str(left["title"]),
                        "role": str(left["role"]),
                        "role_label": str(left["role_label"]),
                        "source_label": left_source,
                        "ending_hook": str(left["ending_hook"]),
                        "arc_titles": list(left["arc_titles"]),
                        "volume_position": left.get("volume_position"),
                    },
                    "right": {
                        "id": str(right["id"]),
                        "position": right_position,
                        "title": str(right["title"]),
                        "role": str(right["role"]),
                        "role_label": str(right["role_label"]),
                        "source_label": right_source,
                        "purpose": str(right.get("outline") or ""),
                        "arc_titles": list(right["arc_titles"]),
                        "volume_position": right.get("volume_position"),
                    },
                    "same_volume": same_volume,
                    "shared_arcs": shared_arcs,
                    "causal_bridges": causal_bridges,
                    "finding_ids": finding_ids,
                    "health_severity": severity,
                }
            )
        return boundaries

    @staticmethod
    def _attach_finding_state(
        *,
        future_chapters: List[Dict[str, Any]],
        volume_health: List[Dict[str, Any]],
        boundaries: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        finding_by_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        chapter_map = {
            int(item["position"]): item for item in future_chapters
        }
        volume_map = {
            str(item["id"]): item for item in volume_health
        }
        for finding in findings:
            for position in finding["chapter_positions"]:
                chapter = chapter_map.get(int(position))
                if chapter is not None:
                    chapter["finding_ids"].append(finding["id"])
            if finding["volume_id"] in volume_map:
                volume = volume_map[finding["volume_id"]]
                if finding["id"] not in volume["finding_ids"]:
                    volume["finding_ids"].append(finding["id"])
        for chapter in future_chapters:
            severities = [
                str(finding_by_id[finding_id]["severity"])
                for finding_id in chapter["finding_ids"]
                if finding_id in finding_by_id
            ]
            chapter["health_severity"] = (
                min(severities, key=lambda value: SEVERITY_ORDER[value])
                if severities
                else "healthy"
            )
        for volume in volume_health:
            severities = [
                str(finding_by_id[finding_id]["severity"])
                for finding_id in volume["finding_ids"]
                if finding_id in finding_by_id
            ]
            volume["health_severity"] = (
                min(severities, key=lambda value: SEVERITY_ORDER[value])
                if severities
                else "healthy"
            )
        for boundary in boundaries:
            severities = [
                str(finding_by_id[finding_id]["severity"])
                for finding_id in boundary["finding_ids"]
                if finding_id in finding_by_id
            ]
            boundary["health_severity"] = (
                min(severities, key=lambda value: SEVERITY_ORDER[value])
                if severities
                else "healthy"
            )
