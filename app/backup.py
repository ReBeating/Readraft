from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .config import Settings
from .process_lock import ProcessLock


BACKUP_FORMAT = "novelai-full-backup"
BACKUP_VERSION = 1
DATABASE_ARCHIVE_PATH = "database.sqlite3"
MANIFEST_ARCHIVE_PATH = "manifest.json"
DATA_PATH_TOKEN = "__NOVELAI_DATA__/"
MAX_ARCHIVE_ENTRIES = 100_000
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024 * 1024
PATH_COLUMNS = (
    ("chapters", "content_path"),
    ("documents", "source_path"),
    ("novel_chapter_versions", "content_path"),
    ("novel_chapters", "content_path"),
    ("novel_scene_versions", "content_path"),
    ("story_structure_applications", "recovery_path"),
)
REQUIRED_TABLES = frozenset(
    {
        "users",
        "novel_projects",
        "novel_chapters",
        "novel_chapter_versions",
        "documents",
        "chapters",
    }
)


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackupSummary:
    path: Path
    created_at: str
    file_count: int
    total_bytes: int
    safety_backup: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or value != path.as_posix()
    ):
        raise BackupError(f"备份包含不安全路径：{value!r}")
    return value


def _assert_archive_outside_data(
    settings: Settings, destination: Path
) -> None:
    data_dir = settings.data_dir.resolve()
    resolved = destination.resolve()
    try:
        resolved.relative_to(data_dir)
    except ValueError:
        return
    raise BackupError("备份文件必须保存在应用数据目录之外")


def _assert_database_integrity(path: Path) -> None:
    if not path.is_file():
        raise BackupError("备份数据库不存在")
    try:
        # SQLite databases created from a WAL-mode source may need to create
        # transient sidecar files while being checked. All callers pass a
        # private staging copy, so opening that copy read-write is safe and
        # avoids false failures on otherwise valid snapshots.
        connection = sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise BackupError("无法打开备份数据库") from exc
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or str(integrity[0]).lower() != "ok":
            raise BackupError("备份数据库完整性检查失败")
        tables = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                """
            ).fetchall()
        }
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise BackupError(
                "备份数据库缺少必要数据表：" + "、".join(missing)
            )
    except sqlite3.Error as exc:
        raise BackupError("备份数据库结构检查失败") from exc
    finally:
        connection.close()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table,),
    ).fetchone()
    return bool(row)


def _normalize_snapshot_paths(
    snapshot_path: Path, settings: Settings
) -> None:
    data_dir = settings.data_dir.resolve()
    allowed_roots = (
        settings.documents_dir.resolve(),
        settings.novels_dir.resolve(),
    )
    connection = sqlite3.connect(snapshot_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for table, column in PATH_COLUMNS:
            if not _table_exists(connection, table):
                continue
            rows = connection.execute(
                f'SELECT rowid, "{column}" FROM "{table}"'
            ).fetchall()
            for rowid, raw_value in rows:
                value = str(raw_value or "").strip()
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = data_dir / path
                resolved = path.resolve(strict=False)
                try:
                    relative = resolved.relative_to(data_dir)
                except ValueError as exc:
                    raise BackupError(
                        f"{table}.{column} 指向数据目录之外，"
                        "无法安全备份"
                    ) from exc
                if not any(
                    _is_relative_to(resolved, root)
                    for root in allowed_roots
                ):
                    raise BackupError(
                        f"{table}.{column} 不在正文或参考资料目录中"
                    )
                if not resolved.exists():
                    raise BackupError(
                        f"{table}.{column} 指向的文件或目录不存在"
                    )
                connection.execute(
                    f'UPDATE "{table}" SET "{column}"=? WHERE rowid=?',
                    (DATA_PATH_TOKEN + relative.as_posix(), rowid),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _restore_snapshot_paths(
    snapshot_path: Path, settings: Settings
) -> None:
    data_dir = settings.data_dir.resolve()
    connection = sqlite3.connect(snapshot_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for table, column in PATH_COLUMNS:
            if not _table_exists(connection, table):
                continue
            rows = connection.execute(
                f'SELECT rowid, "{column}" FROM "{table}"'
            ).fetchall()
            for rowid, raw_value in rows:
                value = str(raw_value or "").strip()
                if not value:
                    continue
                if not value.startswith(DATA_PATH_TOKEN):
                    raise BackupError(
                        f"备份中的 {table}.{column} 不是可迁移路径"
                    )
                relative_value = value[len(DATA_PATH_TOKEN) :]
                relative = PurePosixPath(relative_value)
                _safe_archive_name(relative.as_posix())
                restored = data_dir.joinpath(*relative.parts).resolve(
                    strict=False
                )
                if not _is_relative_to(restored, data_dir):
                    raise BackupError("备份正文路径越过目标数据目录")
                connection.execute(
                    f'UPDATE "{table}" SET "{column}"=? WHERE rowid=?',
                    (str(restored), rowid),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _iter_data_files(settings: Settings) -> Iterable[tuple[str, Path]]:
    data_dir = settings.data_dir.resolve()
    for root in (settings.documents_dir, settings.novels_dir):
        resolved_root = root.resolve()
        if not resolved_root.exists():
            continue
        if not _is_relative_to(resolved_root, data_dir):
            raise BackupError(
                "正文和参考资料目录必须位于应用数据目录内"
            )
        for path in sorted(resolved_root.rglob("*")):
            if path.is_symlink():
                raise BackupError(
                    f"数据目录包含不受支持的符号链接：{path}"
                )
            if not path.is_file():
                continue
            relative = path.resolve().relative_to(data_dir).as_posix()
            yield f"files/{relative}", path


def _create_database_snapshot(source_path: Path, target_path: Path) -> None:
    if not source_path.is_file():
        raise BackupError(f"数据库不存在：{source_path}")
    source = sqlite3.connect(
        f"file:{source_path.resolve()}?mode=ro", uri=True, timeout=30
    )
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        target.commit()
    except sqlite3.Error as exc:
        raise BackupError("创建 SQLite 一致性快照失败") from exc
    finally:
        target.close()
        source.close()


def _write_archive(
    settings: Settings,
    destination: Path,
    *,
    overwrite: bool,
) -> BackupSummary:
    destination = destination.resolve()
    _assert_archive_outside_data(settings, destination)
    if destination.exists() and not overwrite:
        raise BackupError(f"备份文件已存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created_at = _utc_now()
    temporary_archive = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    try:
        with tempfile.TemporaryDirectory(
            prefix="novelai-backup-"
        ) as temporary:
            snapshot = Path(temporary) / DATABASE_ARCHIVE_PATH
            _create_database_snapshot(settings.database_path, snapshot)
            _normalize_snapshot_paths(snapshot, settings)
            _assert_database_integrity(snapshot)

            sources: list[tuple[str, Path]] = [
                (DATABASE_ARCHIVE_PATH, snapshot),
                *_iter_data_files(settings),
            ]
            with zipfile.ZipFile(
                temporary_archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for archive_name, source_path in sources:
                    safe_name = _safe_archive_name(archive_name)
                    size = source_path.stat().st_size
                    total_bytes += size
                    if total_bytes > MAX_ARCHIVE_BYTES:
                        raise BackupError("备份内容超过允许的最大体积")
                    archive.write(source_path, safe_name)
                    entries.append(
                        {
                            "path": safe_name,
                            "size": size,
                            "sha256": _sha256_path(source_path),
                        }
                    )
                manifest = {
                    "format": BACKUP_FORMAT,
                    "version": BACKUP_VERSION,
                    "created_at": created_at,
                    "database_entry": DATABASE_ARCHIVE_PATH,
                    "data_roots": ["documents", "novels"],
                    "entries": entries,
                }
                archive.writestr(
                    MANIFEST_ARCHIVE_PATH,
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ).encode("utf-8"),
                )
        os.replace(temporary_archive, destination)
        destination.chmod(0o600)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise
    return BackupSummary(
        path=destination,
        created_at=created_at,
        file_count=len(entries),
        total_bytes=total_bytes,
    )


def create_backup(settings: Settings, destination: Path) -> BackupSummary:
    lock = ProcessLock(settings.data_dir / ".worker.lock")
    lock.acquire()
    try:
        return _write_archive(settings, destination, overwrite=False)
    finally:
        lock.release()


def _read_manifest(archive: zipfile.ZipFile) -> Mapping[str, Any]:
    try:
        info = archive.getinfo(MANIFEST_ARCHIVE_PATH)
    except KeyError as exc:
        raise BackupError("备份缺少 manifest.json") from exc
    if info.file_size > 5 * 1024 * 1024:
        raise BackupError("备份清单异常过大")
    try:
        payload = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError("备份清单无法解析") from exc
    if not isinstance(payload, dict):
        raise BackupError("备份清单格式不正确")
    if payload.get("format") != BACKUP_FORMAT:
        raise BackupError("这不是 novelAI 完整备份")
    if payload.get("version") != BACKUP_VERSION:
        raise BackupError("暂不支持这个备份版本")
    return payload


def _validate_archive_members(
    archive: zipfile.ZipFile, manifest: Mapping[str, Any]
) -> Sequence[Mapping[str, Any]]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise BackupError("备份包含过多文件")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise BackupError("备份包含重复路径")
    for info in infos:
        _safe_archive_name(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if stat.S_ISLNK(mode):
            raise BackupError("备份不能包含符号链接")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise BackupError("备份清单没有文件记录")
    entries: list[Mapping[str, Any]] = []
    expected_names = {MANIFEST_ARCHIVE_PATH}
    declared_total = 0
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise BackupError("备份文件记录格式不正确")
        path = _safe_archive_name(str(raw.get("path") or ""))
        digest = str(raw.get("sha256") or "")
        size = raw.get("size")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise BackupError(f"备份文件记录无效：{path}")
        declared_total += size
        if declared_total > MAX_ARCHIVE_BYTES:
            raise BackupError("备份声明的内容体积过大")
        if path in expected_names:
            raise BackupError(f"备份清单包含重复路径：{path}")
        expected_names.add(path)
        entries.append(raw)
    if set(names) != expected_names:
        raise BackupError("备份实际文件与清单不一致")
    if manifest.get("database_entry") != DATABASE_ARCHIVE_PATH:
        raise BackupError("备份数据库入口不正确")
    if DATABASE_ARCHIVE_PATH not in expected_names:
        raise BackupError("备份缺少数据库快照")
    return entries


def _extract_and_verify(
    archive_path: Path, destination: Path
) -> tuple[Mapping[str, Any], int, int]:
    if not archive_path.is_file():
        raise BackupError(f"备份文件不存在：{archive_path}")
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BackupError("备份归档已损坏或不是 ZIP 文件") from exc
    total_bytes = 0
    try:
        manifest = _read_manifest(archive)
        entries = _validate_archive_members(archive, manifest)
        for entry in entries:
            archive_name = str(entry["path"])
            target = destination.joinpath(*PurePosixPath(archive_name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(archive_name) as source, target.open("wb") as output:
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_ARCHIVE_BYTES:
                        raise BackupError("备份解压后体积过大")
                    digest.update(chunk)
                    output.write(chunk)
            if size != int(entry["size"]):
                raise BackupError(f"备份文件大小不匹配：{archive_name}")
            if digest.hexdigest() != str(entry["sha256"]):
                raise BackupError(f"备份文件校验失败：{archive_name}")
        database_path = destination / DATABASE_ARCHIVE_PATH
        _assert_database_integrity(database_path)
        return manifest, len(entries), total_bytes
    except (OSError, zipfile.BadZipFile) as exc:
        raise BackupError("读取备份归档失败") from exc
    finally:
        archive.close()


def verify_backup(archive_path: Path) -> BackupSummary:
    with tempfile.TemporaryDirectory(
        prefix="novelai-verify-"
    ) as temporary:
        manifest, file_count, total_bytes = _extract_and_verify(
            archive_path.resolve(), Path(temporary)
        )
    return BackupSummary(
        path=archive_path.resolve(),
        created_at=str(manifest["created_at"]),
        file_count=file_count,
        total_bytes=total_bytes,
    )


def _replace_directory(source: Path, target: Path, token: str) -> Path | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.with_name(f".{target.name}.pre-restore-{token}")
    if previous.exists():
        raise BackupError(f"恢复暂存目录已存在：{previous}")
    had_target = target.exists()
    if had_target:
        os.replace(target, previous)
    try:
        os.replace(source, target)
    except Exception:
        if had_target and previous.exists():
            os.replace(previous, target)
        raise
    return previous if had_target else None


def _apply_restored_data(
    settings: Settings, extracted: Path, token: str
) -> None:
    restored_database = extracted / DATABASE_ARCHIVE_PATH
    _restore_snapshot_paths(restored_database, settings)
    _assert_database_integrity(restored_database)

    staged_database = settings.database_path.with_name(
        f".{settings.database_path.name}.restore-{token}.tmp"
    )
    staged_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(restored_database, staged_database)
    staged_database.chmod(0o600)

    staged_roots: list[tuple[Path, Path]] = []
    for root_name, target in (
        ("documents", settings.documents_dir),
        ("novels", settings.novels_dir),
    ):
        source = extracted / "files" / root_name
        source.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.restore-{token}.tmp")
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(source, staged)
        staged_roots.append((staged, target))

    replaced_roots: list[tuple[Path | None, Path]] = []
    try:
        for staged, target in staged_roots:
            previous = _replace_directory(staged, target, token)
            replaced_roots.append((previous, target))
        os.replace(staged_database, settings.database_path)
    except Exception:
        staged_database.unlink(missing_ok=True)
        for staged, _target in staged_roots:
            if staged.exists():
                shutil.rmtree(staged)
        for previous, target in reversed(replaced_roots):
            if target.exists():
                shutil.rmtree(target)
            if previous is not None and previous.exists():
                os.replace(previous, target)
        raise

    for suffix in ("-wal", "-shm"):
        Path(f"{settings.database_path}{suffix}").unlink(missing_ok=True)
    for previous, _target in replaced_roots:
        if previous is not None:
            shutil.rmtree(previous)

    settings.ensure_directories()
    for root in (settings.documents_dir, settings.novels_dir):
        for directory in [root, *[item for item in root.rglob("*") if item.is_dir()]]:
            directory.chmod(0o700)
        for file_path in (item for item in root.rglob("*") if item.is_file()):
            file_path.chmod(0o600)


def restore_backup(
    settings: Settings,
    archive_path: Path,
    *,
    replace: bool = False,
) -> BackupSummary:
    if not replace:
        raise BackupError(
            "恢复会替换当前数据，必须显式传入 replace=True"
        )
    archive_path = archive_path.resolve()
    _assert_archive_outside_data(settings, archive_path)
    lock = ProcessLock(settings.data_dir / ".worker.lock")
    lock.acquire()
    safety_backup: Path | None = None
    try:
        token = f"{_timestamp_token()}-{uuid.uuid4().hex[:8]}"
        if settings.database_path.exists():
            safety_backup = archive_path.with_name(
                f"novelai-pre-restore-{token}.zip"
            )
            _write_archive(settings, safety_backup, overwrite=False)
        with tempfile.TemporaryDirectory(
            prefix="novelai-restore-",
            dir=settings.data_dir.parent,
        ) as temporary:
            extracted = Path(temporary)
            manifest, file_count, total_bytes = _extract_and_verify(
                archive_path, extracted
            )
            _apply_restored_data(settings, extracted, token)
        return BackupSummary(
            path=archive_path,
            created_at=str(manifest["created_at"]),
            file_count=file_count,
            total_bytes=total_bytes,
            safety_backup=safety_backup,
        )
    finally:
        lock.release()


def _summary_payload(summary: BackupSummary) -> dict[str, Any]:
    return {
        "path": str(summary.path),
        "created_at": summary.created_at,
        "file_count": summary.file_count,
        "total_bytes": summary.total_bytes,
        "safety_backup": (
            str(summary.safety_backup) if summary.safety_backup else None
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novelai-backup",
        description="创建、校验或恢复 novelAI 完整本地备份。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create", help="创建完整备份")
    create_parser.add_argument("archive", type=Path)
    verify_parser = subparsers.add_parser("verify", help="校验完整备份")
    verify_parser.add_argument("archive", type=Path)
    restore_parser = subparsers.add_parser("restore", help="恢复完整备份")
    restore_parser.add_argument("archive", type=Path)
    restore_parser.add_argument(
        "--replace",
        action="store_true",
        help="确认替换当前数据库与正文文件",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    try:
        if args.command == "create":
            summary = create_backup(settings, args.archive)
        elif args.command == "verify":
            summary = verify_backup(args.archive)
        else:
            summary = restore_backup(
                settings, args.archive, replace=bool(args.replace)
            )
    except BackupError as exc:
        parser.exit(2, f"错误：{exc}\n")
    print(json.dumps(_summary_payload(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
