"""Durable, user-scoped import previews.

Text uploads are kept immutable while the user reviews chapter boundaries.  The
preview metadata only stores offsets into ``source.txt``; every edit therefore
remains reversible and committing can prove that no source text was dropped or
duplicated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .chapter_splitter import ChapterChunk


PREVIEW_ID = re.compile(r"^[0-9a-f]{32}$")
PREVIEW_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ImportBoundary:
    title: str
    start: int
    end: int
    kind: str
    confidence: float
    reason: str
    title_source: str

    @property
    def char_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ImportPreview:
    id: str
    user_id: int
    title: str
    original_filename: str
    source_encoding: str
    source_hash: str
    source_text: str
    created_at: float
    updated_at: float
    boundaries: tuple[ImportBoundary, ...]

    @property
    def char_count(self) -> int:
        return len(self.source_text)


class ImportPreviewStore:
    """File-backed preview store with immutable source text and atomic edits."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_seconds: int = 24 * 60 * 60,
        max_previews_per_user: int = 5,
    ):
        self.root = root.resolve()
        self.ttl_seconds = ttl_seconds
        self.max_previews_per_user = max_previews_per_user
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def create(
        self,
        *,
        user_id: int,
        title: str,
        original_filename: str,
        source_encoding: str,
        source_text: str,
        chunks: Iterable[ChapterChunk],
    ) -> ImportPreview:
        chunk_list = list(chunks)
        if not source_text or not chunk_list:
            raise ValueError("没有可预览的导入内容")
        self.prune(user_id=user_id)
        preview_id = uuid.uuid4().hex
        directory = self._preview_dir(user_id, preview_id)
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        directory.chmod(0o700)
        now = time.time()
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        source_path = directory / "source.txt"
        source_path.write_text(source_text, encoding="utf-8")
        source_path.chmod(0o600)
        boundaries = tuple(
            ImportBoundary(
                title=str(chunk.title)[:120],
                start=int(chunk.source_start),
                end=int(chunk.source_end),
                kind=str(chunk.kind),
                confidence=max(0.0, min(1.0, float(chunk.split_confidence))),
                reason=str(chunk.split_reason),
                title_source=str(chunk.title_source),
            )
            for chunk in chunk_list
        )
        preview = ImportPreview(
            id=preview_id,
            user_id=user_id,
            title=title[:120],
            original_filename=original_filename[:255],
            source_encoding=source_encoding[:40],
            source_hash=source_hash,
            source_text=source_text,
            created_at=now,
            updated_at=now,
            boundaries=boundaries,
        )
        try:
            self._validate(preview)
            self._write_metadata(preview)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self._enforce_limit(user_id)
        return preview

    def get(self, *, user_id: int, preview_id: str) -> ImportPreview | None:
        directory = self._preview_dir(user_id, preview_id)
        metadata_path = directory / "metadata.json"
        source_path = directory / "source.txt"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_text = source_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return None
        try:
            preview = ImportPreview(
                id=str(metadata["id"]),
                user_id=int(metadata["user_id"]),
                title=str(metadata["title"]),
                original_filename=str(metadata["original_filename"]),
                source_encoding=str(metadata["source_encoding"]),
                source_hash=str(metadata["source_hash"]),
                source_text=source_text,
                created_at=float(metadata["created_at"]),
                updated_at=float(metadata["updated_at"]),
                boundaries=tuple(
                    ImportBoundary(
                        title=str(item["title"]),
                        start=int(item["start"]),
                        end=int(item["end"]),
                        kind=str(item["kind"]),
                        confidence=float(item["confidence"]),
                        reason=str(item["reason"]),
                        title_source=str(item["title_source"]),
                    )
                    for item in metadata["boundaries"]
                ),
            )
            self._validate(preview)
        except (KeyError, TypeError, ValueError):
            return None
        if preview.user_id != user_id or preview.id != preview_id:
            return None
        if time.time() - preview.updated_at > self.ttl_seconds:
            self.delete(user_id=user_id, preview_id=preview_id)
            return None
        return preview

    def rename(
        self, *, user_id: int, preview_id: str, index: int, title: str
    ) -> ImportPreview:
        clean_title = " ".join(str(title).split())[:120]
        if not clean_title:
            raise ValueError("章节名不能为空")
        preview = self._required(user_id, preview_id)
        boundaries = list(preview.boundaries)
        boundary = self._boundary(boundaries, index)
        boundaries[index] = ImportBoundary(
            title=clean_title,
            start=boundary.start,
            end=boundary.end,
            kind=boundary.kind,
            confidence=boundary.confidence,
            reason=boundary.reason,
            title_source="manual",
        )
        return self._replace(preview, boundaries)

    def merge(
        self, *, user_id: int, preview_id: str, index: int
    ) -> ImportPreview:
        preview = self._required(user_id, preview_id)
        boundaries = list(preview.boundaries)
        left = self._boundary(boundaries, index)
        if index + 1 >= len(boundaries):
            raise ValueError("最后一章后面没有可合并章节")
        right = boundaries[index + 1]
        if left.end != right.start:
            raise ValueError("章节边界不连续，无法合并")
        boundaries[index : index + 2] = [
            ImportBoundary(
                title=left.title,
                start=left.start,
                end=right.end,
                kind="manual",
                confidence=1.0,
                reason="用户手动合并相邻章节",
                title_source=left.title_source,
            )
        ]
        return self._replace(preview, boundaries)

    def split(
        self,
        *,
        user_id: int,
        preview_id: str,
        index: int,
        source_offset: int,
    ) -> ImportPreview:
        preview = self._required(user_id, preview_id)
        boundaries = list(preview.boundaries)
        boundary = self._boundary(boundaries, index)
        if source_offset <= boundary.start or source_offset >= boundary.end:
            raise ValueError("拆分位置必须位于章节正文内部")
        if source_offset - boundary.start < 1 or boundary.end - source_offset < 1:
            raise ValueError("拆分后两部分都必须包含正文")
        boundaries[index : index + 1] = [
            ImportBoundary(
                title=f"{boundary.title}（上）"[:120],
                start=boundary.start,
                end=source_offset,
                kind="manual",
                confidence=1.0,
                reason="用户手动选择拆分位置",
                title_source="generated",
            ),
            ImportBoundary(
                title=f"{boundary.title}（下）"[:120],
                start=source_offset,
                end=boundary.end,
                kind="manual",
                confidence=1.0,
                reason="用户手动选择拆分位置",
                title_source="generated",
            ),
        ]
        return self._replace(preview, boundaries)

    def to_chunks(self, preview: ImportPreview) -> list[ChapterChunk]:
        self._validate(preview)
        return [
            ChapterChunk(
                title=boundary.title,
                text=preview.source_text[boundary.start : boundary.end],
                kind=boundary.kind,
                source_start=boundary.start,
                source_end=boundary.end,
                split_confidence=boundary.confidence,
                split_reason=boundary.reason,
                title_source=boundary.title_source,
            )
            for boundary in preview.boundaries
        ]

    def delete(self, *, user_id: int, preview_id: str) -> bool:
        directory = self._preview_dir(user_id, preview_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        return True

    def prune(self, *, user_id: int | None = None) -> None:
        roots = [self._user_dir(user_id)] if user_id is not None else list(self.root.iterdir())
        cutoff = time.time() - self.ttl_seconds
        for user_root in roots:
            if not user_root.is_dir():
                continue
            for directory in user_root.iterdir():
                if not directory.is_dir() or not PREVIEW_ID.fullmatch(directory.name):
                    continue
                metadata = directory / "metadata.json"
                try:
                    updated_at = float(
                        json.loads(metadata.read_text(encoding="utf-8"))["updated_at"]
                    )
                except (OSError, ValueError, TypeError, KeyError):
                    updated_at = directory.stat().st_mtime
                if updated_at < cutoff:
                    shutil.rmtree(directory, ignore_errors=True)

    def _replace(
        self, preview: ImportPreview, boundaries: Iterable[ImportBoundary]
    ) -> ImportPreview:
        updated = ImportPreview(
            id=preview.id,
            user_id=preview.user_id,
            title=preview.title,
            original_filename=preview.original_filename,
            source_encoding=preview.source_encoding,
            source_hash=preview.source_hash,
            source_text=preview.source_text,
            created_at=preview.created_at,
            updated_at=time.time(),
            boundaries=tuple(boundaries),
        )
        self._validate(updated)
        self._write_metadata(updated)
        return updated

    def _required(self, user_id: int, preview_id: str) -> ImportPreview:
        preview = self.get(user_id=user_id, preview_id=preview_id)
        if preview is None:
            raise ValueError("导入预览不存在或已经过期")
        return preview

    @staticmethod
    def _boundary(boundaries: list[ImportBoundary], index: int) -> ImportBoundary:
        if index < 0 or index >= len(boundaries):
            raise ValueError("章节位置不存在")
        return boundaries[index]

    def _validate(self, preview: ImportPreview) -> None:
        if not PREVIEW_ID.fullmatch(preview.id):
            raise ValueError("导入预览编号无效")
        digest = hashlib.sha256(preview.source_text.encode("utf-8")).hexdigest()
        if digest != preview.source_hash:
            raise ValueError("导入原文校验失败，请重新上传")
        if not preview.boundaries:
            raise ValueError("导入预览没有章节")
        expected_start = 0
        for boundary in preview.boundaries:
            if boundary.start != expected_start or boundary.end <= boundary.start:
                raise ValueError("章节边界不连续")
            if boundary.end > len(preview.source_text):
                raise ValueError("章节边界超出原文")
            if not boundary.title.strip() or len(boundary.title) > 120:
                raise ValueError("章节名无效")
            if not 0.0 <= boundary.confidence <= 1.0:
                raise ValueError("章节边界置信度无效")
            expected_start = boundary.end
        if expected_start != len(preview.source_text):
            raise ValueError("章节边界没有完整覆盖原文")

    def _write_metadata(self, preview: ImportPreview) -> None:
        directory = self._preview_dir(preview.user_id, preview.id)
        payload = {
            "schema_version": PREVIEW_SCHEMA_VERSION,
            "id": preview.id,
            "user_id": preview.user_id,
            "title": preview.title,
            "original_filename": preview.original_filename,
            "source_encoding": preview.source_encoding,
            "source_hash": preview.source_hash,
            "char_count": preview.char_count,
            "created_at": preview.created_at,
            "updated_at": preview.updated_at,
            "boundaries": [asdict(item) for item in preview.boundaries],
        }
        path = directory / "metadata.json"
        temporary = directory / f".metadata.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)

    def _enforce_limit(self, user_id: int) -> None:
        directory = self._user_dir(user_id)
        previews = []
        for child in directory.iterdir():
            if child.is_dir() and PREVIEW_ID.fullmatch(child.name):
                previews.append((child.stat().st_mtime, child))
        previews.sort(reverse=True)
        for _modified, child in previews[self.max_previews_per_user :]:
            shutil.rmtree(child, ignore_errors=True)

    def _user_dir(self, user_id: int) -> Path:
        if int(user_id) <= 0:
            raise ValueError("用户编号无效")
        directory = self.root / str(int(user_id))
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        return directory

    def _preview_dir(self, user_id: int, preview_id: str) -> Path:
        if not PREVIEW_ID.fullmatch(str(preview_id)):
            raise ValueError("导入预览编号无效")
        return self._user_dir(user_id) / preview_id
