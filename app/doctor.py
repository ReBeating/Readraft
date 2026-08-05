from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .text_metrics import effective_char_count


@dataclass
class IntegrityIssue:
    severity: str
    code: str
    message: str
    path: str = ""
    repaired: bool = False


@dataclass
class IntegrityReport:
    database_path: str
    checked_files: int
    issues: list[IntegrityIssue]
    repaired_files: int = 0
    pruned_files: int = 0

    @property
    def ok(self) -> bool:
        return not any(
            issue.severity == "error" and not issue.repaired
            for issue in self.issues
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "database_path": self.database_path,
            "checked_files": self.checked_files,
            "repaired_files": self.repaired_files,
            "pruned_files": self.pruned_files,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_path(raw_path: str, data_dir: Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path.resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_utf8(path: Path) -> str:
    # ``Path.read_text`` performs universal-newline translation. Version
    # hashes and character counts are recorded from the exact submitted text,
    # so preserve CRLF bytes while still validating UTF-8.
    return path.read_bytes().decode("utf-8")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.doctor-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def inspect_integrity(
    settings: Settings,
    *,
    repair: bool = False,
    prune_orphans: bool = False,
) -> IntegrityReport:
    """Audit the live repository without changing authoritative history.

    ``repair`` only refreshes the non-authoritative chapter cache from a
    verified HEAD. ``prune_orphans`` additionally removes unreferenced files
    from ``versions/`` directories and interrupted ``*.tmp`` writes.
    """

    database_path = settings.database_path.resolve()
    report = IntegrityReport(str(database_path), checked_files=0, issues=[])
    if not database_path.is_file():
        report.issues.append(
            IntegrityIssue("error", "database_missing", "数据库文件不存在")
        )
        return report

    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    referenced_files: set[Path] = set()
    verified_text: dict[Path, str] = {}

    def issue(
        severity: str,
        code: str,
        message: str,
        path: Path | None = None,
        *,
        repaired: bool = False,
    ) -> None:
        report.issues.append(
            IntegrityIssue(
                severity,
                code,
                message,
                str(path) if path else "",
                repaired,
            )
        )

    def check_file(
        raw_path: str,
        *,
        label: str,
        expected_hash: str = "",
        expected_chars: int | None = None,
        expected_effective_chars: int | None = None,
        require_inside_data: bool = True,
    ) -> tuple[Path, str | None]:
        path = _resolve_path(raw_path, settings.data_dir)
        referenced_files.add(path)
        if require_inside_data and not _within(path, settings.data_dir.resolve()):
            issue(
                "error",
                "path_outside_data",
                f"{label} 指向数据目录之外",
                path,
            )
            return path, None
        try:
            text = _read_utf8(path)
        except FileNotFoundError:
            issue("error", "file_missing", f"{label} 文件不存在", path)
            return path, None
        except (OSError, UnicodeError):
            issue("error", "file_unreadable", f"{label} 不是可读 UTF-8 文件", path)
            return path, None
        report.checked_files += 1
        verified_text[path] = text
        actual_hash = _sha256(text)
        if expected_hash and actual_hash != expected_hash:
            issue("error", "hash_mismatch", f"{label} 内容哈希不一致", path)
        if expected_chars is not None and len(text) != expected_chars:
            issue(
                "error",
                "char_count_mismatch",
                f"{label} 字符数记录为 {expected_chars}，实际为 {len(text)}",
                path,
            )
        if (
            expected_effective_chars is not None
            and effective_char_count(text) != expected_effective_chars
        ):
            issue(
                "warning",
                "effective_char_count_mismatch",
                f"{label} 有效字符数与文件不一致",
                path,
            )
        return path, text

    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if len(integrity) != 1 or str(integrity[0][0]).lower() != "ok":
            issue(
                "error",
                "sqlite_integrity",
                "SQLite 完整性检查失败："
                + "；".join(str(row[0]) for row in integrity[:10]),
            )
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        for row in foreign_keys:
            issue(
                "error",
                "foreign_key_violation",
                f"外键异常：表 {row[0]} 行 {row[1]} 引用 {row[2]}",
            )

        tables = _tables(connection)
        required = {
            "novel_projects",
            "novel_chapters",
            "novel_chapter_versions",
            "novel_chapter_edit_buffers",
            "work_versions",
            "work_tag_chapter_heads",
            "documents",
            "chapters",
        }
        missing_tables = sorted(required - tables)
        for table in missing_tables:
            issue("error", "table_missing", f"缺少必要数据表 {table}")
        if missing_tables:
            return report

        version_columns = _columns(connection, "novel_chapter_versions")
        effective_expression = (
            "version.effective_char_count"
            if "effective_char_count" in version_columns
            else "NULL"
        )
        versions = connection.execute(
            f"""
            SELECT version.id, version.chapter_id, version.parent_version_id,
                   version.content_path, version.content_hash,
                   version.char_count,
                   {effective_expression} AS effective_char_count,
                   chapter.head_version_id,
                   chapter.content_path AS cache_content_path
            FROM novel_chapter_versions version
            JOIN novel_chapters chapter ON chapter.id=version.chapter_id
            ORDER BY version.created_at, version.rowid
            """
        ).fetchall()
        version_chapters = {
            str(row["id"]): str(row["chapter_id"]) for row in versions
        }
        for row in versions:
            version_id = str(row["id"])
            chapter_id = str(row["chapter_id"])
            parent_id = str(row["parent_version_id"] or "")
            if parent_id and version_chapters.get(parent_id) != chapter_id:
                issue(
                    "error",
                    "cross_chapter_parent",
                    f"版本 {version_id} 的父版本不属于同一章",
                )
            check_file(
                str(row["content_path"]),
                label=f"章节版本 {version_id}",
                expected_hash=str(row["content_hash"] or ""),
                expected_chars=int(row["char_count"] or 0),
                expected_effective_chars=(
                    int(row["effective_char_count"])
                    if row["effective_char_count"] is not None
                    else None
                ),
            )

        chapters = connection.execute(
            """
            SELECT chapter.id, chapter.head_version_id, chapter.content_path,
                   chapter.char_count,
                   head.chapter_id AS head_chapter_id,
                   head.content_path AS head_content_path,
                   head.content_hash AS head_content_hash
            FROM novel_chapters chapter
            LEFT JOIN novel_chapter_versions head
              ON head.id=chapter.head_version_id
            ORDER BY chapter.project_id, chapter.position
            """
        ).fetchall()
        for row in chapters:
            chapter_id = str(row["id"])
            head_id = str(row["head_version_id"] or "")
            if head_id and str(row["head_chapter_id"] or "") != chapter_id:
                issue(
                    "error",
                    "cross_chapter_head",
                    f"章节 {chapter_id} 的 HEAD 不属于该章",
                )
                continue
            cache_path = _resolve_path(str(row["content_path"]), settings.data_dir)
            referenced_files.add(cache_path)
            if not head_id:
                if cache_path.exists():
                    check_file(
                        str(cache_path),
                        label=f"空章节 {chapter_id} 缓存",
                        expected_chars=int(row["char_count"] or 0),
                    )
                continue
            head_path = _resolve_path(
                str(row["head_content_path"]), settings.data_dir
            )
            head_text = verified_text.get(head_path)
            if head_text is None:
                continue
            cache_text: str | None
            try:
                cache_text = _read_utf8(cache_path)
                report.checked_files += 1
            except (OSError, UnicodeError):
                cache_text = None
            if cache_text != head_text:
                repaired = False
                if repair:
                    _atomic_write(cache_path, head_text)
                    report.repaired_files += 1
                    repaired = True
                issue(
                    "warning" if repaired else "error",
                    "chapter_cache_stale",
                    f"章节 {chapter_id} 的显示缓存与 HEAD 不一致",
                    cache_path,
                    repaired=repaired,
                )
            if int(row["char_count"] or 0) != len(head_text):
                issue(
                    "error",
                    "chapter_head_count_mismatch",
                    f"章节 {chapter_id} 的 HEAD 字符数与章节记录不一致",
                )

        buffers = connection.execute(
            """
            SELECT buffer.chapter_id, buffer.base_version_id,
                   buffer.content, buffer.content_hash,
                   chapter.head_version_id,
                   base.chapter_id AS base_chapter_id
            FROM novel_chapter_edit_buffers buffer
            JOIN novel_chapters chapter ON chapter.id=buffer.chapter_id
            LEFT JOIN novel_chapter_versions base
              ON base.id=buffer.base_version_id
            """
        ).fetchall()
        for row in buffers:
            chapter_id = str(row["chapter_id"])
            base_id = str(row["base_version_id"] or "")
            if base_id and str(row["base_chapter_id"] or "") != chapter_id:
                issue(
                    "error",
                    "cross_chapter_buffer_base",
                    f"章节 {chapter_id} 的暂存稿基线不属于该章",
                )
            if base_id != str(row["head_version_id"] or ""):
                issue(
                    "warning",
                    "stale_edit_buffer",
                    f"章节 {chapter_id} 的暂存稿基于旧 HEAD",
                )
            if _sha256(str(row["content"])) != str(row["content_hash"] or ""):
                issue(
                    "error",
                    "edit_buffer_hash_mismatch",
                    f"章节 {chapter_id} 的暂存稿哈希不一致",
                )

        documents = connection.execute(
            "SELECT id, source_path, char_count FROM documents"
        ).fetchall()
        for row in documents:
            check_file(
                str(row["source_path"]),
                label=f"导入文档 {row['id']}",
                expected_chars=int(row["char_count"] or 0),
            )
        document_chapters = connection.execute(
            """
            SELECT id, document_id, position, content_path, char_count
            FROM chapters ORDER BY document_id, position
            """
        ).fetchall()
        document_chapter_map = {
            str(row["id"]): dict(row) for row in document_chapters
        }
        for row in document_chapters:
            check_file(
                str(row["content_path"]),
                label=f"导入章节 {row['id']}",
                expected_chars=int(row["char_count"] or 0),
            )

        tags = connection.execute(
            """
            SELECT id, document_id, ref_name, intent
            FROM work_versions
            WHERE ref_type='tag' AND document_id IS NOT NULL
            """
        ).fetchall()
        for tag in tags:
            tag_id = str(tag["id"])
            rows = connection.execute(
                """
                SELECT * FROM work_tag_chapter_heads
                WHERE work_version_id=? ORDER BY position
                """,
                (tag_id,),
            ).fetchall()
            tag_chapters = [
                row
                for row in document_chapters
                if str(row["document_id"]) == str(tag["document_id"])
            ]
            if str(tag["intent"] or "") == "snapshot" and len(rows) != len(
                tag_chapters
            ):
                issue(
                    "error",
                    "tag_manifest_incomplete",
                    f"Tag {tag['ref_name']} 的章节清单不完整",
                )
            for snapshot in rows:
                chapter = document_chapter_map.get(
                    str(snapshot["document_chapter_id"])
                )
                if (
                    not chapter
                    or str(chapter["document_id"]) != str(tag["document_id"])
                    or int(chapter["position"]) != int(snapshot["position"])
                ):
                    issue(
                        "error",
                        "tag_manifest_target_mismatch",
                        f"Tag {tag['ref_name']} 的章节清单指向错误",
                    )
                    continue
                path = _resolve_path(
                    str(chapter["content_path"]), settings.data_dir
                )
                content = verified_text.get(path)
                if content is not None and _sha256(content) != str(
                    snapshot["content_hash"] or ""
                ):
                    issue(
                        "error",
                        "tag_manifest_hash_mismatch",
                        f"Tag {tag['ref_name']} 的第 {snapshot['position']} 章哈希不一致",
                        path,
                    )

        project_roots = [
            settings.novels_dir / str(row["user_id"]) / str(row["id"])
            for row in connection.execute(
                "SELECT id, user_id FROM novel_projects"
            ).fetchall()
        ]
        for project_root in project_roots:
            if not project_root.exists():
                continue
            for path in project_root.rglob("*"):
                if not path.is_file():
                    continue
                resolved = path.resolve(strict=False)
                is_interrupted_write = (
                    path.name.startswith(".") and path.name.endswith(".tmp")
                )
                is_unreferenced_version = (
                    path.parent.name == "versions"
                    and resolved not in referenced_files
                )
                if not (is_interrupted_write or is_unreferenced_version):
                    continue
                pruned = False
                if prune_orphans:
                    path.unlink(missing_ok=True)
                    report.pruned_files += 1
                    pruned = True
                issue(
                    "warning",
                    "orphan_file",
                    "已清理无引用文件" if pruned else "发现无引用文件",
                    path,
                    repaired=pruned,
                )
    except sqlite3.Error as exc:
        issue("error", "database_query_failed", f"数据库检查失败：{exc}")
    finally:
        connection.close()
    return report


def _print_human(report: IntegrityReport) -> None:
    state = "通过" if report.ok else "发现错误"
    print(
        f"Readraft 完整性检查：{state}；"
        f"检查 {report.checked_files} 个文件，"
        f"修复 {report.repaired_files} 个，清理 {report.pruned_files} 个。"
    )
    for item in report.issues:
        marker = "已处理" if item.repaired else item.severity.upper()
        suffix = f" ({item.path})" if item.path else ""
        print(f"[{marker}] {item.code}: {item.message}{suffix}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 Readraft 数据库、正文历史、Tag 与文件完整性"
    )
    parser.add_argument("--database", type=Path, help="覆盖数据库路径")
    parser.add_argument("--data-dir", type=Path, help="覆盖数据目录")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="从已验证 HEAD 刷新非权威章节缓存",
    )
    parser.add_argument(
        "--prune-orphans",
        action="store_true",
        help="删除 versions/ 中无数据库引用的文件和中断临时文件",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    settings = Settings.from_env()
    if arguments.data_dir:
        settings = replace(settings, data_dir=arguments.data_dir.resolve())
    if arguments.database:
        settings = replace(settings, database_path=arguments.database.resolve())
    report = inspect_integrity(
        settings,
        repair=bool(arguments.repair),
        prune_orphans=bool(arguments.prune_orphans),
    )
    if arguments.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
