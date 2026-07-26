from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .db import Database, has_active_project_ai_task


ARCHIVE_FORMAT = "novelai-project"
ARCHIVE_VERSION = 1
MANIFEST_NAME = "manifest.json"
PROJECT_PATH_PREFIX = "project://"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ROWS = 200_000
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_PATH_BYTES = 1_024
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

EXCLUDED_TABLES = {
    "users",
    "api_credentials",
    "api_models",
    "documents",
    "chapters",
    "analysis_jobs",
    "chapter_analyses",
    "works",
    "work_editions",
    "work_archive_entries",
    "schema_migrations",
    "story_memory_fts",
    "story_memory_fts_config",
    "story_memory_fts_data",
    "story_memory_fts_docsize",
    "story_memory_fts_idx",
}
FORBIDDEN_FIELDS = {
    "password",
    "password_hash",
    "api_key",
    "encrypted_key",
    "secret",
    "secret_key",
}
ACTIVE_QUEUE_TABLES = {
    "generation_jobs",
    "story_plan_suggestions",
    "story_structure_suggestions",
    "novel_causal_link_suggestions",
    "novel_causal_branch_simulations",
    "voice_profile_suggestions",
    "editing_preference_suggestions",
}


class ProjectArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ForeignKeyInfo:
    column: str
    parent_table: str
    parent_column: str


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    not_null: bool
    default: Optional[str]
    primary_key_position: int


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: Mapping[str, ColumnInfo]
    primary_key: Sequence[str]
    foreign_keys: Sequence[ForeignKeyInfo]


@dataclass(frozen=True)
class ArchiveSummary:
    project_id: str
    title: str
    table_count: int
    row_count: int
    file_count: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class ImportedProject:
    project_id: str
    title: str
    row_count: int
    file_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _quote_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ProjectArchiveError("归档包含非法数据库标识符")
    return f'"{value}"'


def _safe_archive_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or len(name.encode("utf-8")) > MAX_ARCHIVE_PATH_BYTES
    ):
        raise ProjectArchiveError("归档包含不安全的文件路径")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ProjectArchiveError("归档包含不安全的文件路径")
    normalized = path.as_posix()
    if normalized != name:
        raise ProjectArchiveError("归档文件路径未规范化")
    return normalized


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_zip_entry(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_metadata(connection: sqlite3.Connection) -> Dict[str, TableInfo]:
    tables: Dict[str, TableInfo] = {}
    rows = connection.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    for row in rows:
        name = str(row["name"])
        sql = str(row["sql"] or "")
        if name in EXCLUDED_TABLES or sql.upper().startswith(
            "CREATE VIRTUAL TABLE"
        ):
            continue
        _quote_identifier(name)
        column_rows = connection.execute(
            f"PRAGMA table_info({_quote_identifier(name)})"
        ).fetchall()
        columns = {
            str(item["name"]): ColumnInfo(
                name=str(item["name"]),
                type=str(item["type"] or ""),
                not_null=bool(item["notnull"]),
                default=(
                    str(item["dflt_value"])
                    if item["dflt_value"] is not None
                    else None
                ),
                primary_key_position=int(item["pk"] or 0),
            )
            for item in column_rows
        }
        primary_key = tuple(
            column.name
            for column in sorted(
                (
                    item
                    for item in columns.values()
                    if item.primary_key_position
                ),
                key=lambda item: item.primary_key_position,
            )
        )
        if not primary_key:
            continue
        foreign_key_rows = connection.execute(
            f"PRAGMA foreign_key_list({_quote_identifier(name)})"
        ).fetchall()
        if any(int(item["seq"] or 0) != 0 for item in foreign_key_rows):
            raise ProjectArchiveError(
                f"暂不支持复合外键表：{name}"
            )
        foreign_keys = tuple(
            ForeignKeyInfo(
                column=str(item["from"]),
                parent_table=str(item["table"]),
                parent_column=str(item["to"]),
            )
            for item in foreign_key_rows
        )
        tables[name] = TableInfo(
            name=name,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )
    return tables


def _row_key(table: TableInfo, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in table.primary_key)


def _chunks(values: Sequence[Any], size: int = 500) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _add_rows(
    *,
    table: TableInfo,
    destination: List[Dict[str, Any]],
    keys: set[tuple[Any, ...]],
    candidates: Iterable[Mapping[str, Any]],
) -> bool:
    changed = False
    for candidate in candidates:
        row = dict(candidate)
        key = _row_key(table, row)
        if key in keys:
            continue
        destination.append(row)
        keys.add(key)
        changed = True
    return changed


def _select_project_rows(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    user_id: int,
    tables: Mapping[str, TableInfo],
) -> Dict[str, List[Dict[str, Any]]]:
    project = connection.execute(
        """
        SELECT * FROM novel_projects
        WHERE id=? AND user_id=?
        """,
        (project_id, user_id),
    ).fetchone()
    if not project:
        raise ProjectArchiveError("作品不存在或不属于当前账号")
    if has_active_project_ai_task(
        connection, user_id=user_id, project_id=project_id
    ):
        raise ProjectArchiveError(
            "作品有 AI 任务正在排队或运行，请完成后再导出"
        )

    selected = {name: [] for name in tables}
    selected_keys = {name: set() for name in tables}
    project_table = tables.get("novel_projects")
    if not project_table:
        raise ProjectArchiveError("当前数据库缺少作品表")
    _add_rows(
        table=project_table,
        destination=selected["novel_projects"],
        keys=selected_keys["novel_projects"],
        candidates=[project],
    )

    for table in tables.values():
        if table.name == "novel_projects" or "project_id" not in table.columns:
            continue
        rows = connection.execute(
            f"SELECT * FROM {_quote_identifier(table.name)} "
            "WHERE project_id=?",
            (project_id,),
        ).fetchall()
        if table.name == "assistant_message_quotes":
            rows = [
                row
                for row in rows
                if str(row["source_type"]) == "novel_version"
            ]
        _add_rows(
            table=table,
            destination=selected[table.name],
            keys=selected_keys[table.name],
            candidates=rows,
        )

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
                    query = (
                        f"SELECT * FROM {_quote_identifier(child.name)} "
                        f"WHERE {_quote_identifier(foreign_key.column)} "
                        f"IN ({placeholders})"
                    )
                    parameters: List[Any] = list(batch)
                    if "project_id" in child.columns:
                        query += " AND project_id=?"
                        parameters.append(project_id)
                    rows = connection.execute(
                        query, tuple(parameters)
                    ).fetchall()
                    if child.name == "assistant_message_quotes":
                        rows = [
                            row
                            for row in rows
                            if str(row["source_type"]) == "novel_version"
                        ]
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
    return {
        name: sorted(
            rows,
            key=lambda row: tuple(
                str(part) for part in _row_key(tables[name], row)
            ),
        )
        for name, rows in selected.items()
        if rows
    }


def _prune_external_dependencies(
    selected: Dict[str, List[Dict[str, Any]]],
    tables: Mapping[str, TableInfo],
) -> None:
    changed = True
    while changed:
        changed = False
        reference_values: Dict[tuple[str, str], set[Any]] = {}
        for table_name, rows in selected.items():
            for column in tables[table_name].columns:
                reference_values[(table_name, column)] = {
                    row[column] for row in rows if row.get(column) is not None
                }
        for table_name, rows in list(selected.items()):
            table = tables[table_name]
            retained = []
            for row in rows:
                drop = False
                for foreign_key in table.foreign_keys:
                    value = row.get(foreign_key.column)
                    if value is None or foreign_key.parent_table == "users":
                        continue
                    available = reference_values.get(
                        (
                            foreign_key.parent_table,
                            foreign_key.parent_column,
                        ),
                        set(),
                    )
                    if value in available:
                        continue
                    column = table.columns[foreign_key.column]
                    if column.not_null:
                        drop = True
                        break
                    row[foreign_key.column] = None
                if drop:
                    changed = True
                else:
                    retained.append(row)
            selected[table_name] = retained

    for row in selected.get("reference_technique_cards", []):
        row["source_document_id"] = None
        row["source_chapter_id"] = None
        row["source_analysis_id"] = None


def _project_file_payload(
    *,
    rows: Dict[str, List[Dict[str, Any]]],
    tables: Mapping[str, TableInfo],
    project_root: Path,
) -> tuple[List[Dict[str, Any]], int]:
    files: Dict[str, Dict[str, Any]] = {}
    total_size = 0
    root = project_root.resolve()
    for table_name, table_rows in rows.items():
        path_columns = [
            name for name in tables[table_name].columns if name.endswith("_path")
        ]
        for row in table_rows:
            for column in path_columns:
                raw_value = str(row.get(column) or "")
                if not raw_value:
                    continue
                path = Path(raw_value)
                if path.is_symlink():
                    raise ProjectArchiveError(
                        f"作品文件不能是符号链接：{path.name}"
                    )
                try:
                    resolved = path.resolve(strict=True)
                    relative = resolved.relative_to(root)
                except (FileNotFoundError, ValueError) as exc:
                    raise ProjectArchiveError(
                        f"作品文件缺失或超出作品目录：{path.name}"
                    ) from exc
                if not resolved.is_file():
                    raise ProjectArchiveError(
                        f"作品路径不是普通文件：{path.name}"
                    )
                relative_name = _safe_archive_name(relative.as_posix())
                archive_name = f"files/{relative_name}"
                row[column] = PROJECT_PATH_PREFIX + relative_name
                if archive_name in files:
                    continue
                size = resolved.stat().st_size
                files[archive_name] = {
                    "path": archive_name,
                    "project_path": relative_name,
                    "size": size,
                    "sha256": _sha256_file(resolved),
                    "_source_path": resolved,
                }
                total_size += size
    return [files[name] for name in sorted(files)], total_size


def _schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
    ).fetchone()
    return int(row["version"] or 0)


def create_project_archive(
    *,
    database: Database,
    novels_dir: Path,
    user_id: int,
    project_id: str,
    destination: Path,
    max_uncompressed_bytes: int,
) -> ArchiveSummary:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    project_root = novels_dir / str(user_id) / project_id
    with database.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            tables = _table_metadata(connection)
            rows = _select_project_rows(
                connection,
                project_id=project_id,
                user_id=user_id,
                tables=tables,
            )
            project_rows = rows.get("novel_projects") or []
            project = project_rows[0]
            files, file_bytes = _project_file_payload(
                rows=rows,
                tables=tables,
                project_root=project_root,
            )
            public_files = [
                {
                    key: value
                    for key, value in item.items()
                    if not key.startswith("_")
                }
                for item in files
            ]
            row_count = sum(len(items) for items in rows.values())
            manifest = {
                "format": ARCHIVE_FORMAT,
                "version": ARCHIVE_VERSION,
                "created_at": _utc_now(),
                "source_schema_version": _schema_version(connection),
                "project": {
                    "id": project_id,
                    "title": str(project.get("title") or ""),
                },
                "tables": rows,
                "files": public_files,
                "counts": {
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
                raise ProjectArchiveError("作品结构数据过大，无法创建归档")
            if uncompressed_bytes > max_uncompressed_bytes:
                raise ProjectArchiveError(
                    "作品归档超过允许大小，建议先清理不再需要的历史版本"
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
                        item["_source_path"], arcname=item["path"]
                    )
            verify_project_archive(
                destination,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            destination.unlink(missing_ok=True)
            raise
    os.chmod(destination, 0o600)
    return ArchiveSummary(
        project_id=project_id,
        title=str(project.get("title") or ""),
        table_count=len(rows),
        row_count=row_count,
        file_count=len(files),
        uncompressed_bytes=uncompressed_bytes,
    )


def _validate_manifest_shape(
    manifest: Any,
) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ProjectArchiveError("作品归档清单格式不正确")
    if manifest.get("format") != ARCHIVE_FORMAT:
        raise ProjectArchiveError("这不是 novelAI 作品归档")
    if manifest.get("version") != ARCHIVE_VERSION:
        raise ProjectArchiveError("暂不支持这个作品归档版本")
    project = manifest.get("project")
    tables = manifest.get("tables")
    files = manifest.get("files")
    if not isinstance(project, dict) or not isinstance(
        project.get("id"), str
    ):
        raise ProjectArchiveError("作品归档缺少作品标识")
    if not isinstance(tables, dict) or not isinstance(files, list):
        raise ProjectArchiveError("作品归档清单缺少数据表或文件列表")
    if set(tables).intersection(EXCLUDED_TABLES):
        raise ProjectArchiveError("作品归档不得包含账号、凭据或参考书数据")
    total_rows = 0
    for table_name, rows in tables.items():
        _quote_identifier(str(table_name))
        if not isinstance(rows, list):
            raise ProjectArchiveError("作品归档的数据表格式不正确")
        total_rows += len(rows)
        if total_rows > MAX_ARCHIVE_ROWS:
            raise ProjectArchiveError("作品归档包含过多数据行")
        for row in rows:
            if not isinstance(row, dict):
                raise ProjectArchiveError("作品归档的数据行格式不正确")
            lowered = {str(key).lower() for key in row}
            if lowered.intersection(FORBIDDEN_FIELDS):
                raise ProjectArchiveError("作品归档包含禁止的敏感字段")
    project_rows = tables.get("novel_projects")
    if (
        not isinstance(project_rows, list)
        or len(project_rows) != 1
        or project_rows[0].get("id") != project["id"]
    ):
        raise ProjectArchiveError("作品归档的根作品记录不唯一")
    for table_name in ACTIVE_QUEUE_TABLES:
        for row in tables.get(table_name, []):
            if str(row.get("status") or "") in {"queued", "running"}:
                raise ProjectArchiveError("作品归档包含未完成的 AI 任务")
    for row in tables.get("assistant_messages", []):
        if (
            str(row.get("role") or "") == "assistant"
            and str(row.get("status") or "") in {"queued", "running"}
        ):
            raise ProjectArchiveError("作品归档包含未完成的 AI 对话")
    return manifest


def verify_project_archive(
    archive_path: Path, *, max_uncompressed_bytes: int
) -> Dict[str, Any]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise ProjectArchiveError("作品归档不存在")
    if archive_path.stat().st_size > max_uncompressed_bytes:
        raise ProjectArchiveError("作品归档文件超过允许大小")
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProjectArchiveError("作品归档 ZIP 已损坏") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise ProjectArchiveError("作品归档包含过多文件")
        names = [_safe_archive_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise ProjectArchiveError("作品归档包含重复文件名")
        if any(info.is_dir() or _is_zip_symlink(info) for info in infos):
            raise ProjectArchiveError("作品归档不能包含目录项或符号链接")
        total_size = sum(info.file_size for info in infos)
        if total_size > max_uncompressed_bytes:
            raise ProjectArchiveError("作品归档解压后超过允许大小")
        info_by_name = {info.filename: info for info in infos}
        manifest_info = info_by_name.get(MANIFEST_NAME)
        if not manifest_info:
            raise ProjectArchiveError("作品归档缺少 manifest.json")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ProjectArchiveError("作品归档清单过大")
        try:
            with archive.open(manifest_info, "r") as handle:
                manifest = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectArchiveError("作品归档清单不是有效 JSON") from exc
        manifest = _validate_manifest_shape(manifest)
        file_items: Dict[str, Dict[str, Any]] = {}
        for item in manifest["files"]:
            if not isinstance(item, dict):
                raise ProjectArchiveError("作品归档文件清单格式不正确")
            name = _safe_archive_name(str(item.get("path") or ""))
            project_path = _safe_archive_name(
                str(item.get("project_path") or "")
            )
            if name != f"files/{project_path}":
                raise ProjectArchiveError("作品归档文件路径映射不一致")
            if name in file_items:
                raise ProjectArchiveError("作品归档文件清单存在重复项")
            try:
                size = int(item["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProjectArchiveError("作品归档文件大小无效") from exc
            digest = str(item.get("sha256") or "")
            if size < 0 or not SHA256_RE.fullmatch(digest):
                raise ProjectArchiveError("作品归档文件校验信息无效")
            info = info_by_name.get(name)
            if not info or info.file_size != size:
                raise ProjectArchiveError("作品归档文件大小与清单不一致")
            file_items[name] = item
        expected_names = {MANIFEST_NAME, *file_items}
        if set(info_by_name) != expected_names:
            raise ProjectArchiveError("作品归档包含未登记的文件")
        for name, item in file_items.items():
            digest = _sha256_zip_entry(archive, info_by_name[name])
            if digest != item["sha256"]:
                raise ProjectArchiveError(
                    f"作品归档文件校验失败：{name}"
                )
        referenced_paths = set()
        for table_rows in manifest["tables"].values():
            for row in table_rows:
                for key, value in row.items():
                    if not str(key).endswith("_path") or not value:
                        continue
                    value = str(value)
                    if not value.startswith(PROJECT_PATH_PREFIX):
                        raise ProjectArchiveError(
                            "作品归档包含绝对路径或未知路径格式"
                        )
                    relative = _safe_archive_name(
                        value[len(PROJECT_PATH_PREFIX) :]
                    )
                    referenced_paths.add(f"files/{relative}")
        if referenced_paths != set(file_items):
            raise ProjectArchiveError("作品归档的文件引用与清单不一致")
        return manifest


def _remap_json_value(value: Any, id_map: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return id_map.get(value, value)
    if isinstance(value, list):
        return [_remap_json_value(item, id_map) for item in value]
    if isinstance(value, dict):
        return {
            id_map.get(str(key), str(key)): _remap_json_value(item, id_map)
            for key, item in value.items()
        }
    return value


def _remap_import_rows(
    *,
    manifest: Mapping[str, Any],
    tables: Mapping[str, TableInfo],
    user_id: int,
    target_project_id: str,
    target_root: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    archive_tables = manifest["tables"]
    unknown_tables = set(archive_tables).difference(tables)
    if unknown_tables:
        raise ProjectArchiveError(
            "当前 novelAI 版本缺少归档所需的数据表，请先升级"
        )
    source_project_id = str(manifest["project"]["id"])
    id_map: Dict[str, str] = {source_project_id: target_project_id}
    for table_name, rows in archive_tables.items():
        table = tables[table_name]
        for row in rows:
            unknown_columns = set(row).difference(table.columns)
            if unknown_columns:
                raise ProjectArchiveError(
                    f"当前版本不认识归档字段：{table_name}"
                )
            for column in table.primary_key:
                value = row.get(column)
                if isinstance(value, str) and value:
                    id_map.setdefault(value, uuid.uuid4().hex)

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
                    and value.startswith(PROJECT_PATH_PREFIX)
                ):
                    relative = _safe_archive_name(
                        value[len(PROJECT_PATH_PREFIX) :]
                    )
                    row[column] = str(target_root / relative)
                    continue
                if isinstance(value, str) and value in id_map:
                    value = id_map[value]
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
    return remapped


def _extract_project_files(
    *,
    archive_path: Path,
    manifest: Mapping[str, Any],
    staging_root: Path,
) -> None:
    staging_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(staging_root, 0o700)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for item in manifest["files"]:
            relative = PurePosixPath(str(item["project_path"]))
            destination = staging_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(destination.parent, 0o700)
            with archive.open(str(item["path"]), "r") as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            os.chmod(destination, 0o600)


def _insert_archive_rows(
    connection: sqlite3.Connection,
    *,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    tables: Mapping[str, TableInfo],
) -> None:
    for table_name in sorted(rows):
        table = tables[table_name]
        for row in rows[table_name]:
            columns = list(row)
            if not columns:
                raise ProjectArchiveError("作品归档包含空数据行")
            for required in table.columns.values():
                if (
                    required.name not in row
                    and required.not_null
                    and required.default is None
                    and not required.primary_key_position
                ):
                    raise ProjectArchiveError(
                        f"作品归档缺少必要字段：{table_name}.{required.name}"
                    )
            quoted_columns = ",".join(
                _quote_identifier(column) for column in columns
            )
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO {_quote_identifier(table_name)} "
                f"({quoted_columns}) VALUES ({placeholders})",
                tuple(row[column] for column in columns),
            )


def import_project_archive(
    *,
    database: Database,
    novels_dir: Path,
    user_id: int,
    archive_path: Path,
    max_uncompressed_bytes: int,
) -> ImportedProject:
    manifest = verify_project_archive(
        archive_path, max_uncompressed_bytes=max_uncompressed_bytes
    )
    target_project_id = uuid.uuid4().hex
    user_root = novels_dir / str(user_id)
    user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(user_root, 0o700)
    staging_root = user_root / f".import-{uuid.uuid4().hex}"
    target_root = user_root / target_project_id
    _extract_project_files(
        archive_path=archive_path,
        manifest=manifest,
        staging_root=staging_root,
    )
    (staging_root / "chapters").mkdir(
        parents=True, exist_ok=True, mode=0o700
    )
    committed = False
    try:
        os.replace(staging_root, target_root)
        with database.connection() as connection:
            tables = _table_metadata(connection)
            source_schema_version = int(
                manifest.get("source_schema_version") or 0
            )
            if _schema_version(connection) < source_schema_version:
                raise ProjectArchiveError(
                    "归档来自更新的 novelAI 数据结构，请先升级当前程序"
                )
            remapped_rows = _remap_import_rows(
                manifest=manifest,
                tables=tables,
                user_id=user_id,
                target_project_id=target_project_id,
                target_root=target_root,
            )
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN IMMEDIATE")
            try:
                _insert_archive_rows(
                    connection, rows=remapped_rows, tables=tables
                )
                violations = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                if violations:
                    sample = violations[0]
                    raise ProjectArchiveError(
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
        shutil.rmtree(staging_root, ignore_errors=True)
        if not committed:
            shutil.rmtree(target_root, ignore_errors=True)
        raise
    project_row = manifest["tables"]["novel_projects"][0]
    return ImportedProject(
        project_id=target_project_id,
        title=str(project_row.get("title") or ""),
        row_count=sum(
            len(rows) for rows in manifest["tables"].values()
        ),
        file_count=len(manifest["files"]),
    )
