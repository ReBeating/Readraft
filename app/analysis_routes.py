"""HTTP boundary for layered reference analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .analysis_repository import AnalysisRepository
from .config import Settings
from .db import Database
from .security import verify_csrf
from .technique_service import TechniqueService
from .web_security import current_user, login_redirect


TemplateContext = Callable[..., dict[str, Any]]
TemplateRenderer = Callable[..., Response]


def build_analysis_router(
    *,
    database: Database,
    settings: Settings,
    repository: AnalysisRepository,
    technique_service: TechniqueService,
    template_context: TemplateContext,
    render_template: TemplateRenderer,
) -> APIRouter:
    router = APIRouter()

    @router.post("/documents/{document_id}/analyze")
    async def analyze_document(
        request: Request, document_id: str, csrf: str = Form(...)
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        document = database.get_document(user_id, document_id)
        if not document:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        credential = database.get_api_credential_summary(user_id)
        analyzer = request.app.state.analyzer
        if credential:
            provider = str(credential["provider"])
            model = str(credential["model"])
            credential_source = "personal"
        elif settings.model_api_key or settings.uses_test_models:
            provider = str(analyzer.provider)
            model = str(analyzer.model)
            credential_source = "default"
        else:
            return RedirectResponse(
                "/settings/api?error=" + quote("开始分析前，请先配置你的模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = repository.create_job(
                user_id=user_id,
                document_id=document_id,
                provider=provider,
                model=model,
                credential_source=credential_source,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/documents/{document_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @router.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_page(request: Request, job_id: str, error: str | None = None):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        user_id = int(user["id"])
        job = repository.get_job(user_id, job_id)
        if not job:
            return render_template(
                "not_found.html",
                template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        chapters = database.list_chapters(user_id, str(job["document_id"]), job_id)
        aggregate = repository.decode_object(job.get("aggregate_json")) or {}
        return render_template(
            "job.html",
            template_context(
                request,
                user=user,
                job=job,
                chapters=chapters,
                aggregate=aggregate,
                error=error,
            ),
        )

    @router.get("/api/jobs/{job_id}")
    async def job_status(request: Request, job_id: str):
        user = current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"}, status_code=status.HTTP_401_UNAUTHORIZED
            )
        job = repository.get_job(int(user["id"]), job_id)
        if not job:
            return JSONResponse(
                {"detail": "任务不存在"}, status_code=status.HTTP_404_NOT_FOUND
            )
        completed = int(job["completed_chapters"])
        failed = int(job["failed_chapters"])
        total = int(job["total_chapters"])
        return {
            "id": job_id,
            "status": job["status"],
            "completed": completed,
            "failed": failed,
            "total": total,
            "processed": completed + failed,
            "percent": round((completed + failed) / total * 100, 1) if total else 0,
            "terminal": job["status"] in {"completed", "partial", "failed"},
        }

    @router.post("/jobs/{job_id}/retry")
    async def retry_job(request: Request, job_id: str, csrf: str = Form(...)):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        job = repository.get_job(user_id, job_id)
        if not job:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if job.get("credential_source") == "personal" and not database.has_api_credential(
            user_id
        ):
            return RedirectResponse(
                "/settings/api?error="
                + quote("重试这个任务前，请先重新配置个人模型凭据"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if not repository.retry_failed(user_id, job_id):
            return RedirectResponse(
                f"/jobs/{job_id}?error="
                + quote("当前无法重试；请确认任务已结束且没有其他任务正在运行"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @router.get("/analyses/{analysis_id}", response_class=HTMLResponse)
    async def analysis_page(
        request: Request,
        analysis_id: str,
        evidence: str | None = None,
        error: str | None = None,
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        user_id = int(user["id"])
        analysis = repository.get_analysis(user_id, analysis_id)
        if not analysis:
            return render_template(
                "not_found.html",
                template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        result = repository.decode_object(analysis.get("result_json"))
        saved_names = technique_service.list_saved_names_for_analysis(
            user_id=user_id, analysis_id=analysis_id
        )
        source_text = Path(str(analysis["content_path"])).read_text(encoding="utf-8")
        evidence_parts: tuple[str, str, str] | None = None
        if evidence:
            try:
                raw_start, raw_end = evidence.split(":", 1)
                start, end = int(raw_start), int(raw_end)
                if 0 <= start < end <= len(source_text):
                    evidence_parts = (
                        source_text[:start],
                        source_text[start:end],
                        source_text[end:],
                    )
            except (TypeError, ValueError):
                evidence_parts = None
        return render_template(
            "analysis.html",
            template_context(
                request,
                user=user,
                analysis=analysis,
                result=result,
                layers=(result or {}).get("layers") or {},
                saved_technique_names=saved_names,
                source_text=source_text,
                evidence_parts=evidence_parts,
                error=error,
            ),
        )

    @router.post("/analyses/{analysis_id}/techniques/{technique_index}")
    async def save_analysis_technique(
        request: Request,
        analysis_id: str,
        technique_index: int,
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            technique_id, created = technique_service.create_from_analysis(
                user_id=int(user["id"]),
                analysis_id=analysis_id,
                technique_index=technique_index,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/analyses/{analysis_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?"
            + ("created=true" if created else "saved=true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.get("/jobs/{job_id}/export.json")
    async def export_job(request: Request, job_id: str):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        payload = repository.export_job(int(user["id"]), job_id)
        if not payload:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        filename = f"story-analysis-{job_id[:8]}.json"
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
