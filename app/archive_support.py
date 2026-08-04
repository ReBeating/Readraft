from __future__ import annotations

import hashlib
import re
import sqlite3
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ROWS = 200_000
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_PATH_BYTES = 1_024
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
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


class ArchiveError(ValueError):
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quote_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ArchiveError("归档包含非法数据库标识符")
    return f'"{value}"'


def safe_archive_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or len(name.encode("utf-8")) > MAX_ARCHIVE_PATH_BYTES
    ):
        raise ArchiveError("归档包含不安全的文件路径")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveError("归档包含不安全的文件路径")
    normalized = path.as_posix()
    if normalized != name:
        raise ArchiveError("归档文件路径未规范化")
    return normalized


def is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(mode) == stat.S_IFLNK


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_entry(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_metadata(
    connection: sqlite3.Connection,
    *,
    excluded_tables: set[str],
) -> Dict[str, TableInfo]:
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
        if name in excluded_tables or sql.upper().startswith(
            "CREATE VIRTUAL TABLE"
        ):
            continue
        quote_identifier(name)
        column_rows = connection.execute(
            f"PRAGMA table_info({quote_identifier(name)})"
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
            f"PRAGMA foreign_key_list({quote_identifier(name)})"
        ).fetchall()
        if any(int(item["seq"] or 0) != 0 for item in foreign_key_rows):
            raise ArchiveError(f"暂不支持复合外键表：{name}")
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


def row_key(
    table: TableInfo,
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    return tuple(row[column] for column in table.primary_key)


def chunks(
    values: Sequence[Any],
    size: int = 500,
) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def add_rows(
    *,
    table: TableInfo,
    destination: List[Dict[str, Any]],
    keys: set[tuple[Any, ...]],
    candidates: Iterable[Mapping[str, Any]],
) -> bool:
    changed = False
    for candidate in candidates:
        row = dict(candidate)
        key = row_key(table, row)
        if key in keys:
            continue
        destination.append(row)
        keys.add(key)
        changed = True
    return changed


def prune_external_dependencies(
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


def schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
    ).fetchone()
    return int(row["version"] or 0)


def remap_json_value(value: Any, id_map: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return id_map.get(value, value)
    if isinstance(value, list):
        return [remap_json_value(item, id_map) for item in value]
    if isinstance(value, dict):
        return {
            id_map.get(str(key), str(key)): remap_json_value(item, id_map)
            for key, item in value.items()
        }
    return value


def insert_archive_rows(
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
                raise ArchiveError("作品归档包含空数据行")
            for required in table.columns.values():
                if (
                    required.name not in row
                    and required.not_null
                    and required.default is None
                    and not required.primary_key_position
                ):
                    raise ArchiveError(
                        "作品归档缺少必要字段："
                        f"{table_name}.{required.name}"
                    )
            quoted_columns = ",".join(
                quote_identifier(column) for column in columns
            )
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO {quote_identifier(table_name)} "
                f"({quoted_columns}) VALUES ({placeholders})",
                tuple(row[column] for column in columns),
            )
