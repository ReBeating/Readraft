from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .archive_support import (
    ACTIVE_QUEUE_TABLES,
    ArchiveError,
    FORBIDDEN_FIELDS,
    MANIFEST_NAME,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_ROWS,
    MAX_MANIFEST_BYTES,
    SHA256_RE,
    TableInfo,
    add_rows as _add_rows,
    chunks as _chunks,
    insert_archive_rows as _insert_archive_rows,
    is_zip_symlink as _is_zip_symlink,
    prune_external_dependencies as _prune_external_dependencies,
    quote_identifier as _quote_identifier,
    remap_json_value as _remap_json_value,
    row_key as _row_key,
    safe_archive_name as _safe_archive_name,
    schema_version as _schema_version,
    sha256_file as _sha256_file,
    sha256_zip_entry as _sha256_zip_entry,
    table_metadata as _table_metadata,
    utc_now as _utc_now,
)
from .db import Database, has_active_project_ai_task


WORK_ARCHIVE_FORMAT = "novelai-work"
WORK_ARCHIVE_VERSION = 2
WORK_PATH_PREFIX = "work://"
WORK_EXCLUDED_TABLES = {
    "users",
    "api_credentials",
    "api_models",
    "user_model_preferences",
    "schema_migrations",
    "story_memory_fts",
    "story_memory_fts_config",
    "story_memory_fts_data",
    "story_memory_fts_docsize",
    "story_memory_fts_idx",
}


WorkArchiveError = ArchiveError


@dataclass(frozen=True)
class WorkArchiveSummary:
    work_id: str
    title: str
    version_count: int
    table_count: int
    row_count: int
    file_count: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class ImportedWork:
    work_id: str
    title: str
    version_count: int
    row_count: int
    file_count: int


def detect_archive_format(archive_path: Path) -> str:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise WorkArchiveError("作品归档不存在")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = [
                info
                for info in archive.infolist()
                if info.filename == MANIFEST_NAME
            ]
            if len(infos) != 1:
                raise WorkArchiveError("作品归档缺少唯一的 manifest.json")
            if infos[0].file_size > MAX_MANIFEST_BYTES:
                raise WorkArchiveError("作品归档清单过大")
            try:
                with archive.open(infos[0], "r") as handle:
                    manifest = json.load(handle)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkArchiveError(
                    "作品归档清单不是有效 JSON"
                ) from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise WorkArchiveError("作品归档 ZIP 已损坏") from exc
    if not isinstance(manifest, dict):
        raise WorkArchiveError("作品归档清单格式不正确")
    archive_format = str(manifest.get("format") or "")
    if not archive_format:
        raise WorkArchiveError("作品归档缺少格式标识")
    return archive_format


def _work_tables(
    connection: sqlite3.Connection,
) -> Dict[str, TableInfo]:
    return _table_metadata(
        connection,
        excluded_tables=WORK_EXCLUDED_TABLES,
    )


def _add_table_rows(
    *,
    tables: Mapping[str, TableInfo],
    selected: Dict[str, List[Dict[str, Any]]],
    selected_keys: Dict[str, set[tuple[Any, ...]]],
    table_name: str,
    candidates: Sequence[Mapping[str, Any]],
) -> bool:
    table = tables.get(table_name)
    if not table:
        raise WorkArchiveError(f"当前数据库缺少必要数据表：{table_name}")
    return _add_rows(
        table=table,
        destination=selected[table_name],
        keys=selected_keys[table_name],
        candidates=candidates,
    )


def _select_work_rows(
    connection: sqlite3.Connection,
    *,
    work_id: str,
    user_id: int,
    tables: Mapping[str, TableInfo],
) -> Dict[str, List[Dict[str, Any]]]:
    work = connection.execute(
        "SELECT * FROM works WHERE id=? AND user_id=?",
        (work_id, user_id),
    ).fetchone()
    if not work:
        raise WorkArchiveError("作品不存在或不属于当前账号")
    versions = connection.execute(
        """
        SELECT * FROM work_versions
        WHERE work_id=?
        ORDER BY created_at, id
        """,
        (work_id,),
    ).fetchall()
    if not versions:
        raise WorkArchiveError("作品没有可归档的版本")

    project_ids = sorted(
        {
            str(row["project_id"])
            for row in versions
            if row["project_id"]
        }
    )
    document_ids = sorted(
        {
            str(row["document_id"])
            for row in versions
            if row["document_id"]
        }
    )
    for project_id in project_ids:
        if has_active_project_ai_task(
            connection,
            user_id=user_id,
            project_id=project_id,
        ):
            raise WorkArchiveError(
                "作品有 AI 任务正在排队或运行，请完成后再导出"
            )
    if document_ids:
        for batch in _chunks(document_ids):
            placeholders = ",".join("?" for _ in batch)
            active_analysis = connection.execute(
                "SELECT 1 FROM analysis_jobs "
                f"WHERE document_id IN ({placeholders}) "
                "AND status IN ('queued', 'running') LIMIT 1",
                tuple(batch),
            ).fetchone()
            if active_analysis:
                raise WorkArchiveError(
                    "作品有阅读分析正在排队或运行，请完成后再导出"
                )

    selected = {name: [] for name in tables}
    selected_keys = {name: set() for name in tables}
    _add_table_rows(
        tables=tables,
        selected=selected,
        selected_keys=selected_keys,
        table_name="works",
        candidates=[work],
    )
    _add_table_rows(
        tables=tables,
        selected=selected,
        selected_keys=selected_keys,
        table_name="work_versions",
        candidates=versions,
    )
    if project_ids:
        for batch in _chunks(project_ids):
            placeholders = ",".join("?" for _ in batch)
            projects = connection.execute(
                "SELECT * FROM novel_projects "
                f"WHERE id IN ({placeholders}) AND user_id=?",
                (*batch, user_id),
            ).fetchall()
            _add_table_rows(
                tables=tables,
                selected=selected,
                selected_keys=selected_keys,
                table_name="novel_projects",
                candidates=projects,
            )
    if document_ids:
        for batch in _chunks(document_ids):
            placeholders = ",".join("?" for _ in batch)
            documents = connection.execute(
                "SELECT * FROM documents "
                f"WHERE id IN ({placeholders}) AND user_id=?",
                (*batch, user_id),
            ).fetchall()
            _add_table_rows(
                tables=tables,
                selected=selected,
                selected_keys=selected_keys,
                table_name="documents",
                candidates=documents,
            )

    selected_projects = {
        str(row["id"]) for row in selected.get("novel_projects", [])
    }
    selected_documents = {
        str(row["id"]) for row in selected.get("documents", [])
    }
    if selected_projects != set(project_ids):
        raise WorkArchiveError("作品包含不可访问的创作版本")
    if selected_documents != set(document_ids):
        raise WorkArchiveError("作品包含不可访问的只读 Tag")

    changed = True
    while changed:
        changed = False
        for child in tables.values():
            for foreign_key in child.foreign_keys:
                parent_rows = selected.get(foreign_key.parent_table) or []
                if not parent_rows:
                    continue
                values = sorted(
                    {
                        row[foreign_key.parent_column]
                        for row in parent_rows
                        if row.get(foreign_key.parent_column) is not None
                    },
                    key=str,
                )
                for batch in _chunks(values):
                    placeholders = ",".join("?" for _ in batch)
                    rows = connection.execute(
                        f"SELECT * FROM {_quote_identifier(child.name)} "
                        f"WHERE {_quote_identifier(foreign_key.column)} "
                        f"IN ({placeholders})",
                        tuple(batch),
                    ).fetchall()
                    changed |= _add_rows(
                        table=child,
                        destination=selected[child.name],
                        keys=selected_keys[child.name],
                        candidates=rows,
                    )

        technique_ids = {
            str(row["technique_id"])
            for row in selected.get("novel_technique_bindings", [])
            if row.get("technique_id")
        }
        technique_table = tables.get("reference_technique_cards")
        if technique_ids and technique_table:
            for batch in _chunks(sorted(technique_ids)):
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT * FROM reference_technique_cards "
                    f"WHERE id IN ({placeholders}) AND user_id=?",
                    (*batch, user_id),
                ).fetchall()
                changed |= _add_rows(
                    table=technique_table,
                    destination=selected["reference_technique_cards"],
                    keys=selected_keys["reference_technique_cards"],
                    candidates=rows,
                )

    _prune_external_dependencies(selected, tables)
    rows = {
        name: sorted(
            items,
            key=lambda row: tuple(
                str(part) for part in _row_key(tables[name], row)
            ),
        )
        for name, items in selected.items()
        if items
    }
    row_count = sum(len(items) for items in rows.values())
    if row_count > MAX_ARCHIVE_ROWS:
        raise WorkArchiveError("作品归档包含过多数据行")
    _validate_inactive_rows(rows)
    return rows


def _validate_inactive_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for table_name in {*ACTIVE_QUEUE_TABLES, "analysis_jobs"}:
        for row in rows.get(table_name, []):
            if str(row.get("status") or "") in {"queued", "running"}:
                raise WorkArchiveError("作品归档包含未完成的 AI 任务")
    for row in rows.get("assistant_messages", []):
        if (
            str(row.get("role") or "") == "assistant"
            and str(row.get("status") or "") in {"queued", "running"}
        ):
            raise WorkArchiveError("作品归档包含未完成的 AI 对话")


def _source_roots(
    *,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    novels_dir: Path,
    documents_dir: Path,
    user_id: int,
) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    for project in rows.get("novel_projects", []):
        project_id = str(project["id"])
        roots[f"projects/{project_id}"] = (
            novels_dir / str(user_id) / project_id
        )
    for document in rows.get("documents", []):
        document_id = str(document["id"])
        roots[f"documents/{document_id}"] = (
            documents_dir / str(user_id) / document_id
        )
    return roots


def _work_file_payload(
    *,
    rows: Dict[str, List[Dict[str, Any]]],
    tables: Mapping[str, TableInfo],
    source_roots: Mapping[str, Path],
) -> tuple[List[Dict[str, Any]], int]:
    resolved_roots = {
        name: path.resolve() for name, path in source_roots.items()
    }
    files: Dict[str, Dict[str, Any]] = {}
    total_size = 0
    for table_name, table_rows in rows.items():
        path_columns = [
            name
            for name in tables[table_name].columns
            if name.endswith("_path")
        ]
        for row in table_rows:
            for column in path_columns:
                raw_value = str(row.get(column) or "")
                if not raw_value:
                    continue
                path = Path(raw_value)
                if path.is_symlink():
                    raise WorkArchiveError(
                        f"作品文件不能是符号链接：{path.name}"
                    )
                try:
                    resolved = path.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise WorkArchiveError(
                        f"作品文件缺失：{path.name}"
                    ) from exc
                if not resolved.is_file():
                    raise WorkArchiveError(
                        f"作品路径不是普通文件：{path.name}"
                    )
                matches: List[tuple[str, PurePosixPath]] = []
                for root_name, root in resolved_roots.items():
                    try:
                        relative = resolved.relative_to(root)
                    except ValueError:
                        continue
                    matches.append(
                        (root_name, PurePosixPath(relative.as_posix()))
                    )
                if len(matches) != 1:
                    raise WorkArchiveError(
                        f"作品文件超出当前作品目录：{path.name}"
                    )
                root_name, relative = matches[0]
                relative_name = _safe_archive_name(relative.as_posix())
                work_path = _safe_archive_name(
                    f"{root_name}/{relative_name}"
                )
                archive_name = f"files/{work_path}"
                row[column] = WORK_PATH_PREFIX + work_path
                if archive_name in files:
                    continue
                size = resolved.stat().st_size
                files[archive_name] = {
                    "path": archive_name,
                    "work_path": work_path,
                    "size": size,
                    "sha256": _sha256_file(resolved),
                    "_source_path": resolved,
                }
                total_size += size
    return [files[name] for name in sorted(files)], total_size


def create_work_archive(
    *,
    database: Database,
    novels_dir: Path,
    documents_dir: Path,
    user_id: int,
    work_id: str,
    destination: Path,
    max_uncompressed_bytes: int,
) -> WorkArchiveSummary:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with database.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            tables = _work_tables(connection)
            rows = _select_work_rows(
                connection,
                work_id=work_id,
                user_id=user_id,
                tables=tables,
            )
            work = rows["works"][0]
            source_roots = _source_roots(
                rows=rows,
                novels_dir=novels_dir,
                documents_dir=documents_dir,
                user_id=user_id,
            )
            files, file_bytes = _work_file_payload(
                rows=rows,
                tables=tables,
                source_roots=source_roots,
            )
            public_files = [
                {
                    key: value
                    for key, value in item.items()
                    if not key.startswith("_")
                }
                for item in files
            ]
            project_ids = sorted(
                str(row["id"]) for row in rows.get("novel_projects", [])
            )
            document_ids = sorted(
                str(row["id"]) for row in rows.get("documents", [])
            )
            row_count = sum(len(items) for items in rows.values())
            manifest = {
                "format": WORK_ARCHIVE_FORMAT,
                "version": WORK_ARCHIVE_VERSION,
                "created_at": _utc_now(),
                "source_schema_version": _schema_version(connection),
                "work": {
                    "id": work_id,
                    "title": str(work.get("title") or ""),
                },
                "roots": {
                    "projects": project_ids,
                    "documents": document_ids,
                },
                "tables": rows,
                "files": public_files,
                "counts": {
                    "versions": len(rows.get("work_versions", [])),
                    "tables": len(rows),
                    "rows": row_count,
                    "files": len(files),
                },
            }
            manifest_bytes = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            uncompressed_bytes = len(manifest_bytes) + file_bytes
            if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise WorkArchiveError("作品结构数据过大，无法创建归档")
            if uncompressed_bytes > max_uncompressed_bytes:
                raise WorkArchiveError(
                    "作品归档超过允许大小，建议先清理不再需要的版本"
                )
            with zipfile.ZipFile(
                destination,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                archive.writestr(MANIFEST_NAME, manifest_bytes)
                for item in files:
                    archive.write(
                        item["_source_path"],
                        arcname=item["path"],
                    )
            verify_work_archive(
                destination,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            destination.unlink(missing_ok=True)
            raise
    os.chmod(destination, 0o600)
    return WorkArchiveSummary(
        work_id=work_id,
        title=str(work.get("title") or ""),
        version_count=len(rows.get("work_versions", [])),
        table_count=len(rows),
        row_count=row_count,
        file_count=len(files),
        uncompressed_bytes=uncompressed_bytes,
    )


def _validate_manifest_shape(manifest: Any) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        raise WorkArchiveError("作品归档清单格式不正确")
    if manifest.get("format") != WORK_ARCHIVE_FORMAT:
        raise WorkArchiveError("这不是 novelAI 完整作品归档")
    if manifest.get("version") != WORK_ARCHIVE_VERSION:
        raise WorkArchiveError("暂不支持这个完整作品归档版本")
    work = manifest.get("work")
    roots = manifest.get("roots")
    tables = manifest.get("tables")
    files = manifest.get("files")
    if not isinstance(work, dict) or not isinstance(work.get("id"), str):
        raise WorkArchiveError("作品归档缺少作品标识")
    if not isinstance(roots, dict):
        raise WorkArchiveError("作品归档缺少版本根目录")
    project_ids = roots.get("projects")
    document_ids = roots.get("documents")
    if not isinstance(project_ids, list) or not isinstance(
        document_ids, list
    ):
        raise WorkArchiveError("作品归档的版本根目录格式不正确")
    if any(not isinstance(value, str) for value in [*project_ids, *document_ids]):
        raise WorkArchiveError("作品归档的版本标识格式不正确")
    if len(project_ids) != len(set(project_ids)) or len(document_ids) != len(
        set(document_ids)
    ):
        raise WorkArchiveError("作品归档包含重复版本标识")
    if not isinstance(tables, dict) or not isinstance(files, list):
        raise WorkArchiveError("作品归档清单缺少数据表或文件列表")
    if set(tables).intersection(WORK_EXCLUDED_TABLES):
        raise WorkArchiveError("作品归档不得包含账号或模型凭据")

    total_rows = 0
    for table_name, table_rows in tables.items():
        _quote_identifier(str(table_name))
        if not isinstance(table_rows, list):
            raise WorkArchiveError("作品归档的数据表格式不正确")
        total_rows += len(table_rows)
        if total_rows > MAX_ARCHIVE_ROWS:
            raise WorkArchiveError("作品归档包含过多数据行")
        for row in table_rows:
            if not isinstance(row, dict):
                raise WorkArchiveError("作品归档的数据行格式不正确")
            lowered = {str(key).lower() for key in row}
            if lowered.intersection(FORBIDDEN_FIELDS):
                raise WorkArchiveError("作品归档包含禁止的敏感字段")

    work_rows = tables.get("works")
    if (
        not isinstance(work_rows, list)
        or len(work_rows) != 1
        or work_rows[0].get("id") != work["id"]
    ):
        raise WorkArchiveError("作品归档的根作品记录不唯一")
    versions = tables.get("work_versions")
    if not isinstance(versions, list) or not versions:
        raise WorkArchiveError("作品归档没有版本记录")
    version_ids = {
        str(row.get("id") or "")
        for row in versions
        if isinstance(row, dict)
    }
    if "" in version_ids or len(version_ids) != len(versions):
        raise WorkArchiveError("作品归档包含无效或重复版本")
    for version in versions:
        if version.get("work_id") != work["id"]:
            raise WorkArchiveError("作品归档包含其他作品的版本")
        project_id = version.get("project_id")
        document_id = version.get("document_id")
        if bool(project_id) == bool(document_id):
            raise WorkArchiveError("作品归档包含无效版本指向")
        if project_id and project_id not in project_ids:
            raise WorkArchiveError("创作版本缺少对应作品数据")
        if document_id and document_id not in document_ids:
            raise WorkArchiveError("只读 Tag 缺少对应文档数据")
        base_version_id = version.get("base_version_id")
        if base_version_id and base_version_id not in version_ids:
            raise WorkArchiveError("基础版本不在当前作品归档中")

    archived_project_ids = {
        str(row.get("id") or "")
        for row in tables.get("novel_projects", [])
    }
    archived_document_ids = {
        str(row.get("id") or "")
        for row in tables.get("documents", [])
    }
    if archived_project_ids != set(project_ids):
        raise WorkArchiveError("作品归档的创作版本清单不一致")
    if archived_document_ids != set(document_ids):
        raise WorkArchiveError("作品归档的 Tag 清单不一致")
    _validate_inactive_rows(tables)
    return manifest


def verify_work_archive(
    archive_path: Path,
    *,
    max_uncompressed_bytes: int,
) -> Dict[str, Any]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise WorkArchiveError("作品归档不存在")
    if archive_path.stat().st_size > max_uncompressed_bytes:
        raise WorkArchiveError("作品归档文件超过允许大小")
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise WorkArchiveError("作品归档 ZIP 已损坏") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise WorkArchiveError("作品归档包含过多文件")
        names = [_safe_archive_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise WorkArchiveError("作品归档包含重复文件名")
        if any(info.is_dir() or _is_zip_symlink(info) for info in infos):
            raise WorkArchiveError("作品归档不能包含目录项或符号链接")
        total_size = sum(info.file_size for info in infos)
        if total_size > max_uncompressed_bytes:
            raise WorkArchiveError("作品归档解压后超过允许大小")
        info_by_name = {info.filename: info for info in infos}
        manifest_info = info_by_name.get(MANIFEST_NAME)
        if not manifest_info:
            raise WorkArchiveError("作品归档缺少 manifest.json")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise WorkArchiveError("作品归档清单过大")
        try:
            with archive.open(manifest_info, "r") as handle:
                manifest = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkArchiveError(
                "作品归档清单不是有效 JSON"
            ) from exc
        manifest = _validate_manifest_shape(manifest)
        project_ids = set(manifest["roots"]["projects"])
        document_ids = set(manifest["roots"]["documents"])
        file_items: Dict[str, Dict[str, Any]] = {}
        for item in manifest["files"]:
            if not isinstance(item, dict):
                raise WorkArchiveError("作品归档文件清单格式不正确")
            name = _safe_archive_name(str(item.get("path") or ""))
            work_path = _safe_archive_name(
                str(item.get("work_path") or "")
            )
            if name != f"files/{work_path}":
                raise WorkArchiveError("作品归档文件路径映射不一致")
            parts = PurePosixPath(work_path).parts
            if (
                len(parts) < 3
                or parts[0] not in {"projects", "documents"}
                or (
                    parts[0] == "projects"
                    and parts[1] not in project_ids
                )
                or (
                    parts[0] == "documents"
                    and parts[1] not in document_ids
                )
            ):
                raise WorkArchiveError("作品归档文件不属于已登记版本")
            if name in file_items:
                raise WorkArchiveError("作品归档文件清单存在重复项")
            try:
                size = int(item["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkArchiveError("作品归档文件大小无效") from exc
            digest = str(item.get("sha256") or "")
            if size < 0 or not SHA256_RE.fullmatch(digest):
                raise WorkArchiveError("作品归档文件校验信息无效")
            info = info_by_name.get(name)
            if not info or info.file_size != size:
                raise WorkArchiveError(
                    "作品归档文件大小与清单不一致"
                )
            file_items[name] = item
        expected_names = {MANIFEST_NAME, *file_items}
        if set(info_by_name) != expected_names:
            raise WorkArchiveError("作品归档包含未登记的文件")
        for name, item in file_items.items():
            digest = _sha256_zip_entry(archive, info_by_name[name])
            if digest != item["sha256"]:
                raise WorkArchiveError(
                    f"作品归档文件校验失败：{name}"
                )
        referenced_paths = set()
        for table_rows in manifest["tables"].values():
            for row in table_rows:
                for key, value in row.items():
                    if not str(key).endswith("_path") or not value:
                        continue
                    value = str(value)
                    if not value.startswith(WORK_PATH_PREFIX):
                        raise WorkArchiveError(
                            "作品归档包含绝对路径或未知路径格式"
                        )
                    work_path = _safe_archive_name(
                        value[len(WORK_PATH_PREFIX) :]
                    )
                    referenced_paths.add(f"files/{work_path}")
        if referenced_paths != set(file_items):
            raise WorkArchiveError(
                "作品归档的文件引用与清单不一致"
            )
        return manifest


def _remap_source_ref(value: str, id_map: Mapping[str, str]) -> str:
    prefix, separator, source_id = value.partition(":")
    if separator and source_id in id_map:
        return f"{prefix}:{id_map[source_id]}"
    return value


def _parse_work_path(value: str) -> tuple[str, str, PurePosixPath]:
    relative = _safe_archive_name(value[len(WORK_PATH_PREFIX) :])
    parts = PurePosixPath(relative).parts
    if len(parts) < 3 or parts[0] not in {"projects", "documents"}:
        raise WorkArchiveError("作品归档包含无效文件映射")
    return parts[0], parts[1], PurePosixPath(*parts[2:])


def _remap_import_rows(
    *,
    manifest: Mapping[str, Any],
    tables: Mapping[str, TableInfo],
    user_id: int,
    novels_dir: Path,
    documents_dir: Path,
) -> tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, str],
    Dict[str, Path],
]:
    archive_tables = manifest["tables"]
    unknown_tables = set(archive_tables).difference(tables)
    if unknown_tables:
        raise WorkArchiveError(
            "当前 novelAI 版本缺少归档所需的数据表，请先升级"
        )
    id_map: Dict[str, str] = {}
    for table_name, rows in archive_tables.items():
        table = tables[table_name]
        for row in rows:
            unknown_columns = set(row).difference(table.columns)
            if unknown_columns:
                raise WorkArchiveError(
                    f"当前版本不认识归档字段：{table_name}"
                )
            for column in table.primary_key:
                value = row.get(column)
                if isinstance(value, str) and value:
                    id_map.setdefault(value, uuid.uuid4().hex)

    source_work_id = str(manifest["work"]["id"])
    if source_work_id not in id_map:
        raise WorkArchiveError("作品归档缺少可映射的作品标识")
    target_roots: Dict[str, Path] = {}
    for source_project_id in manifest["roots"]["projects"]:
        target_project_id = id_map.get(source_project_id)
        if not target_project_id:
            raise WorkArchiveError("作品归档缺少创作版本标识映射")
        target_roots[f"projects/{source_project_id}"] = (
            novels_dir / str(user_id) / target_project_id
        )
    for source_document_id in manifest["roots"]["documents"]:
        target_document_id = id_map.get(source_document_id)
        if not target_document_id:
            raise WorkArchiveError("作品归档缺少 Tag 标识映射")
        target_roots[f"documents/{source_document_id}"] = (
            documents_dir / str(user_id) / target_document_id
        )

    remapped: Dict[str, List[Dict[str, Any]]] = {}
    for table_name, rows in archive_tables.items():
        output_rows = []
        for source in rows:
            row: Dict[str, Any] = {}
            for column, value in source.items():
                if column == "user_id":
                    row[column] = user_id
                    continue
                if (
                    isinstance(value, str)
                    and value.startswith(WORK_PATH_PREFIX)
                ):
                    root_kind, source_root_id, relative = _parse_work_path(
                        value
                    )
                    root = target_roots.get(
                        f"{root_kind}/{source_root_id}"
                    )
                    if not root:
                        raise WorkArchiveError(
                            "作品归档文件缺少目标版本"
                        )
                    row[column] = str(root.joinpath(*relative.parts))
                    continue
                if isinstance(value, str) and value in id_map:
                    value = id_map[value]
                elif column == "source_ref" and isinstance(value, str):
                    value = _remap_source_ref(value, id_map)
                if (
                    isinstance(value, str)
                    and "json" in column.lower()
                    and value.strip()
                ):
                    try:
                        decoded = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    else:
                        value = json.dumps(
                            _remap_json_value(decoded, id_map),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                row[column] = value
            if "claim_token" in row:
                row["claim_token"] = None
            if "lease_expires_at" in row:
                row["lease_expires_at"] = None
            output_rows.append(row)
        remapped[table_name] = output_rows
    return remapped, id_map, target_roots


def _extract_work_files(
    *,
    archive_path: Path,
    manifest: Mapping[str, Any],
    target_roots: Mapping[str, Path],
    transaction_id: str,
) -> tuple[Dict[str, Path], Dict[str, Path]]:
    staging_roots: Dict[str, Path] = {}
    try:
        for root_name, target in target_roots.items():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(target.parent, 0o700)
            if target.exists():
                raise WorkArchiveError(
                    "导入目标目录已经存在，请重试"
                )
            staging = target.parent / (
                f".work-import-{transaction_id}-{target.name}"
            )
            staging.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(staging, 0o700)
            (staging / "chapters").mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )
            staging_roots[root_name] = staging

        with zipfile.ZipFile(archive_path, "r") as archive:
            for item in manifest["files"]:
                work_path = str(item["work_path"])
                root_kind, source_root_id, relative = _parse_work_path(
                    WORK_PATH_PREFIX + work_path
                )
                staging = staging_roots.get(
                    f"{root_kind}/{source_root_id}"
                )
                if not staging:
                    raise WorkArchiveError(
                        "作品归档文件缺少暂存目标"
                    )
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                    mode=0o700,
                )
                os.chmod(destination.parent, 0o700)
                with archive.open(str(item["path"]), "r") as source:
                    with destination.open("wb") as target:
                        shutil.copyfileobj(
                            source,
                            target,
                            length=1024 * 1024,
                        )
                os.chmod(destination, 0o600)
    except Exception:
        for path in staging_roots.values():
            shutil.rmtree(path, ignore_errors=True)
        raise
    return staging_roots, dict(target_roots)


def _validate_document_quota(
    connection: sqlite3.Connection,
    *,
    user_id: int,
    document_rows: Sequence[Mapping[str, Any]],
    max_documents: Optional[int],
    max_stored_chars: Optional[int],
) -> None:
    usage = connection.execute(
        """
        SELECT COUNT(*) AS document_count,
               COALESCE(SUM(char_count), 0) AS stored_chars
        FROM documents WHERE user_id=?
        """,
        (user_id,),
    ).fetchone()
    imported_count = len(document_rows)
    imported_chars = sum(
        int(row.get("char_count") or 0) for row in document_rows
    )
    if (
        max_documents is not None
        and int(usage["document_count"] or 0) + imported_count
        > max_documents
    ):
        raise WorkArchiveError(
            f"每个账号最多保存 {max_documents} 本阅读文档"
        )
    if (
        max_stored_chars is not None
        and int(usage["stored_chars"] or 0) + imported_chars
        > max_stored_chars
    ):
        raise WorkArchiveError(
            f"账号累计正文不能超过 {max_stored_chars:,} 字"
        )


def import_work_archive(
    *,
    database: Database,
    novels_dir: Path,
    documents_dir: Path,
    user_id: int,
    archive_path: Path,
    max_uncompressed_bytes: int,
    max_documents: Optional[int] = None,
    max_stored_chars: Optional[int] = None,
) -> ImportedWork:
    manifest = verify_work_archive(
        archive_path,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    transaction_id = uuid.uuid4().hex
    staging_roots: Dict[str, Path] = {}
    target_roots: Dict[str, Path] = {}
    committed = False
    try:
        with database.connection() as connection:
            tables = _work_tables(connection)
            source_schema_version = int(
                manifest.get("source_schema_version") or 0
            )
            if _schema_version(connection) < source_schema_version:
                raise WorkArchiveError(
                    "归档来自更新的 novelAI 数据结构，请先升级当前程序"
                )
            _validate_document_quota(
                connection,
                user_id=user_id,
                document_rows=manifest["tables"].get("documents", []),
                max_documents=max_documents,
                max_stored_chars=max_stored_chars,
            )
            remapped_rows, id_map, mapped_roots = _remap_import_rows(
                manifest=manifest,
                tables=tables,
                user_id=user_id,
                novels_dir=novels_dir,
                documents_dir=documents_dir,
            )
            staging_roots, target_roots = _extract_work_files(
                archive_path=archive_path,
                manifest=manifest,
                target_roots=mapped_roots,
                transaction_id=transaction_id,
            )
            for root_name, staging in staging_roots.items():
                os.replace(staging, target_roots[root_name])

            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN IMMEDIATE")
            try:
                _insert_archive_rows(
                    connection,
                    rows=remapped_rows,
                    tables=tables,
                )
                violations = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                if violations:
                    sample = violations[0]
                    raise WorkArchiveError(
                        "作品归档引用不完整，无法安全导入："
                        f"{sample['table']}"
                    )
                connection.commit()
                committed = True
            except Exception:
                connection.rollback()
                raise
            finally:
                try:
                    connection.execute("PRAGMA foreign_keys = ON")
                except sqlite3.Error:
                    if not committed:
                        raise
    except Exception:
        for path in staging_roots.values():
            shutil.rmtree(path, ignore_errors=True)
        if not committed:
            for path in target_roots.values():
                shutil.rmtree(path, ignore_errors=True)
        raise

    source_work_id = str(manifest["work"]["id"])
    target_work_id = id_map[source_work_id]
    return ImportedWork(
        work_id=target_work_id,
        title=str(manifest["work"].get("title") or ""),
        version_count=len(manifest["tables"]["work_versions"]),
        row_count=sum(
            len(rows) for rows in manifest["tables"].values()
        ),
        file_count=len(manifest["files"]),
    )
