"""Routes for archive import and reviewable text import."""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .chapter_splitter import decode_upload, split_chapters
from .config import Settings
from .db import Database
from .import_preview import ImportPreview, ImportPreviewStore
from .security import verify_csrf
from .web_security import current_user, login_redirect
from .work_archive import (
    WORK_ARCHIVE_FORMAT,
    WorkArchiveError,
    detect_archive_format,
    import_work_archive,
)
from .work_library import create_reading_document_from_chunks


logger = logging.getLogger(__name__)
ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}

TemplateContext = Callable[..., dict[str, Any]]
TemplateRenderer = Callable[..., Response]


def build_import_router(
    *,
    database: Database,
    settings: Settings,
    preview_store: ImportPreviewStore,
    template_context: TemplateContext,
    render_template: TemplateRenderer,
) -> APIRouter:
    router = APIRouter()

    def import_context(request: Request, user: dict[str, Any], **extra: Any):
        return template_context(
            request,
            user=user,
            max_upload_mb=settings.max_upload_bytes // 1024 // 1024,
            max_archive_mb=settings.max_work_archive_bytes // 1024 // 1024,
            **extra,
        )

    def preview_items(preview: ImportPreview) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        source = preview.source_text
        for index, boundary in enumerate(preview.boundaries):
            text = source[boundary.start : boundary.end]
            excerpt = " ".join(text.strip().split())
            if len(excerpt) > 180:
                excerpt = excerpt[:177].rstrip() + "…"
            items.append(
                {
                    "index": index,
                    "number": index + 1,
                    "title": boundary.title,
                    "start": boundary.start,
                    "end": boundary.end,
                    "char_count": boundary.char_count,
                    "confidence": round(boundary.confidence * 100),
                    "low_confidence": boundary.confidence < 0.8,
                    "reason": boundary.reason,
                    "excerpt": excerpt,
                    "text": text,
                    "can_merge": index + 1 < len(preview.boundaries),
                }
            )
        return items

    def load_preview(user_id: int, preview_id: str) -> ImportPreview | None:
        try:
            return preview_store.get(user_id=user_id, preview_id=preview_id)
        except ValueError:
            return None

    def preview_redirect(
        preview_id: str, *, error: str = "", split: int | None = None
    ) -> RedirectResponse:
        query: list[str] = []
        if error:
            query.append("error=" + quote(error))
        if split is not None:
            query.append(f"split={split}")
        suffix = "?" + "&".join(query) if query else ""
        return RedirectResponse(
            f"/import/previews/{preview_id}{suffix}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.get("/import", response_class=HTMLResponse)
    async def import_page(request: Request, error: str | None = None):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        return render_template(
            "work_import.html",
            import_context(request, user, error=error),
        )

    @router.post("/import", response_class=HTMLResponse)
    async def upload_import(
        request: Request,
        work_file: UploadFile = File(...),
        title: str = Form(""),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        filename = Path(work_file.filename or "untitled.txt").name
        lower_filename = filename.lower()

        def render_error(message: str, status_code: int = 400):
            return render_template(
                "work_import.html",
                import_context(
                    request,
                    user,
                    error=message,
                    title=title.strip()[:120],
                ),
                status_code=status_code,
            )

        if lower_filename.endswith(".zip"):
            temporary = tempfile.NamedTemporaryFile(
                prefix="readraft-unified-import-",
                suffix=".zip",
                delete=False,
            )
            temporary_path = Path(temporary.name)
            total_bytes = 0
            imported_work_id = ""
            try:
                while True:
                    chunk = await work_file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > settings.max_work_archive_bytes:
                        raise WorkArchiveError("作品归档文件超过允许大小")
                    temporary.write(chunk)
                temporary.close()
                if total_bytes == 0:
                    raise WorkArchiveError("请选择非空的作品归档")
                if detect_archive_format(temporary_path) != WORK_ARCHIVE_FORMAT:
                    raise WorkArchiveError("只支持当前版本导出的完整作品归档")
                imported_work = import_work_archive(
                    database=database,
                    novels_dir=settings.novels_dir,
                    documents_dir=settings.documents_dir,
                    user_id=user_id,
                    archive_path=temporary_path,
                    max_uncompressed_bytes=settings.max_work_archive_bytes,
                    max_documents=settings.max_documents_per_user,
                    max_stored_chars=settings.max_stored_chars_per_user,
                )
                imported_work_id = imported_work.work_id
            except (WorkArchiveError, OSError, sqlite3.Error) as exc:
                logger.warning("unified archive import rejected: %s", exc)
                return render_error(str(exc))
            finally:
                temporary.close()
                temporary_path.unlink(missing_ok=True)
                await work_file.close()
            return RedirectResponse(
                f"/dashboard?imported=true&work={imported_work_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )

        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_TEXT_EXTENSIONS:
            await work_file.close()
            return render_error("支持 TXT、Markdown 和 .readraft.zip 作品归档")
        raw = await work_file.read(settings.max_upload_bytes + 1)
        await work_file.close()
        if len(raw) > settings.max_upload_bytes:
            return render_error(
                f"文件超过 {settings.max_upload_bytes // 1024 // 1024} MB 限制"
            )
        if not raw:
            return render_error("文件为空")
        try:
            text, encoding = decode_upload(raw)
            if not text.strip():
                raise ValueError("文件中没有可导入的正文")
            if len(text) > settings.max_text_chars:
                raise ValueError(f"正文超过 {settings.max_text_chars:,} 字限制")
            chunks = split_chapters(
                text,
                target_chars=settings.target_chapter_chars,
                max_chars=settings.max_chapter_chars,
            )
            if not chunks:
                raise ValueError("没有识别到可导入内容")
            preview = preview_store.create(
                user_id=user_id,
                title=(title.strip() or Path(filename).stem)[:120],
                original_filename=filename,
                source_encoding=encoding,
                source_text=text,
                chunks=chunks,
            )
        except ValueError as exc:
            return render_error(str(exc))
        except Exception:
            logger.exception("failed to prepare import preview")
            return render_error("无法创建导入预览，请稍后重试", 500)
        return preview_redirect(preview.id)

    @router.get("/import/previews/{preview_id}", response_class=HTMLResponse)
    async def review_import(
        request: Request,
        preview_id: str,
        split: int | None = None,
        error: str | None = None,
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        preview = load_preview(int(user["id"]), preview_id)
        if preview is None:
            return render_template(
                "not_found.html",
                template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        items = preview_items(preview)
        split_index = split if split is not None and 0 <= split < len(items) else None
        return render_template(
            "import_preview.html",
            template_context(
                request,
                user=user,
                preview=preview,
                items=items,
                split_index=split_index,
                split_item=(items[split_index] if split_index is not None else None),
                error=error,
            ),
        )

    @router.post("/import/previews/{preview_id}/chapters/{index}/rename")
    async def rename_chapter(
        request: Request,
        preview_id: str,
        index: int,
        title: str = Form(...),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            preview_store.rename(
                user_id=int(user["id"]),
                preview_id=preview_id,
                index=index,
                title=title,
            )
        except ValueError as exc:
            return preview_redirect(preview_id, error=str(exc))
        return preview_redirect(preview_id)

    @router.post("/import/previews/{preview_id}/chapters/{index}/merge")
    async def merge_chapter(
        request: Request,
        preview_id: str,
        index: int,
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            preview_store.merge(
                user_id=int(user["id"]), preview_id=preview_id, index=index
            )
        except ValueError as exc:
            return preview_redirect(preview_id, error=str(exc))
        return preview_redirect(preview_id)

    @router.post("/import/previews/{preview_id}/chapters/{index}/split")
    async def split_chapter(
        request: Request,
        preview_id: str,
        index: int,
        source_offset: int = Form(...),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            preview_store.split(
                user_id=int(user["id"]),
                preview_id=preview_id,
                index=index,
                source_offset=source_offset,
            )
        except ValueError as exc:
            return preview_redirect(preview_id, error=str(exc), split=index)
        return preview_redirect(preview_id)

    @router.post("/import/previews/{preview_id}/commit")
    async def commit_import(
        request: Request,
        preview_id: str,
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        preview = load_preview(user_id, preview_id)
        if preview is None:
            return preview_redirect(preview_id, error="导入预览不存在或已经过期")
        try:
            document_id = create_reading_document_from_chunks(
                database=database,
                documents_dir=settings.documents_dir,
                user_id=user_id,
                title=preview.title,
                original_filename=preview.original_filename,
                source_encoding=preview.source_encoding,
                source_text=preview.source_text,
                chunks=preview_store.to_chunks(preview),
                max_documents=settings.max_documents_per_user,
                max_stored_chars=settings.max_stored_chars_per_user,
            )
        except ValueError as exc:
            return preview_redirect(preview_id, error=str(exc))
        except Exception:
            logger.exception("failed to commit import preview id=%s", preview_id)
            return preview_redirect(preview_id, error="导入失败，请稍后重试")
        preview_store.delete(user_id=user_id, preview_id=preview_id)
        return RedirectResponse(
            f"/documents/{document_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.post("/import/previews/{preview_id}/cancel")
    async def cancel_import(
        request: Request,
        preview_id: str,
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            preview_store.delete(user_id=int(user["id"]), preview_id=preview_id)
        except ValueError:
            pass
        return RedirectResponse("/import", status_code=status.HTTP_303_SEE_OTHER)

    return router
