"""HTTP boundary for reusable writing-technique cards and bindings."""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import quote

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .security import verify_csrf
from .technique_service import TechniqueService
from .web_forms import _clean_field, _technique_observation_from_form
from .web_security import current_user, login_redirect


TemplateContext = Callable[..., dict[str, Any]]
TemplateRenderer = Callable[..., Response]


def build_technique_router(
    *,
    service: TechniqueService,
    template_context: TemplateContext,
    render_template: TemplateRenderer,
) -> APIRouter:
    router = APIRouter()

    @router.get("/techniques", response_class=HTMLResponse)
    async def technique_library(
        request: Request,
        error: str | None = None,
        created: bool = False,
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        cards = service.list_cards(user_id=int(user["id"]))
        return render_template(
            "techniques.html",
            template_context(
                request,
                user=user,
                cards=cards,
                error=error,
                created=created,
            ),
        )

    @router.post("/techniques")
    async def create_manual_technique(
        request: Request,
        name: str = Form(...),
        dimension: str = Form(...),
        source_location: str = Form("作者手动总结"),
        observation: str = Form(...),
        effect: str = Form(...),
        suitable_for: str = Form(""),
        unsuitable_for: str = Form(""),
        execution_rule: str = Form(...),
        originality_boundary: str = Form(...),
        author_note: str = Form(""),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            card_id = service.create_manual(
                user_id=int(user["id"]),
                observation=_technique_observation_from_form(
                    name=name,
                    dimension=dimension,
                    source_location=source_location,
                    observation=observation,
                    effect=effect,
                    suitable_for=suitable_for,
                    unsuitable_for=unsuitable_for,
                    execution_rule=execution_rule,
                    originality_boundary=originality_boundary,
                ),
                author_note=_clean_field(author_note, "作者备注", max_length=2000),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{card_id}?created=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.get("/techniques/{technique_id}", response_class=HTMLResponse)
    async def technique_card_page(
        request: Request,
        technique_id: str,
        error: str | None = None,
        saved: bool = False,
        created: bool = False,
        bound: bool = False,
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        user_id = int(user["id"])
        card = service.get_card(user_id=user_id, technique_id=technique_id)
        if not card:
            return render_template(
                "not_found.html",
                template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "technique_card.html",
            template_context(
                request,
                user=user,
                card=card,
                targets=service.binding_targets(user_id=user_id),
                error=error,
                saved=saved,
                created=created,
                bound=bound,
            ),
        )

    @router.post("/techniques/{technique_id}")
    async def update_technique_card(
        request: Request,
        technique_id: str,
        name: str = Form(...),
        dimension: str = Form(...),
        source_location: str = Form(...),
        observation: str = Form(...),
        effect: str = Form(...),
        suitable_for: str = Form(""),
        unsuitable_for: str = Form(""),
        execution_rule: str = Form(...),
        originality_boundary: str = Form(...),
        author_note: str = Form(""),
        card_status: str = Form("active"),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = service.update_card(
                user_id=int(user["id"]),
                technique_id=technique_id,
                observation=_technique_observation_from_form(
                    name=name,
                    dimension=dimension,
                    source_location=source_location,
                    observation=observation,
                    effect=effect,
                    suitable_for=suitable_for,
                    unsuitable_for=unsuitable_for,
                    execution_rule=execution_rule,
                    originality_boundary=originality_boundary,
                ),
                author_note=_clean_field(author_note, "作者备注", max_length=2000),
                status=card_status,
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques/{technique_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.post("/techniques/{technique_id}/bindings")
    async def bind_technique_card(
        request: Request,
        technique_id: str,
        target: str = Form(...),
        usage_modes: list[str] = Form(...),
        author_adaptation: str = Form(""),
        priority: int = Form(50),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            service.bind(
                user_id=int(user["id"]),
                technique_id=technique_id,
                target=target,
                usage_modes=usage_modes,
                author_adaptation=_clean_field(
                    author_adaptation,
                    "针对本书的改造",
                    max_length=1000,
                ),
                priority=priority,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques/{technique_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?bound=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.post("/technique-bindings/{binding_id}/status")
    async def set_technique_binding_status(
        request: Request,
        binding_id: str,
        binding_status: str = Form(...),
        csrf: str = Form(...),
    ):
        user = current_user(request)
        if not user:
            return login_redirect(request)
        verify_csrf(request, csrf)
        try:
            technique_id = service.set_binding_status(
                user_id=int(user["id"]),
                binding_id=binding_id,
                status=binding_status,
            )
            if not technique_id:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return router
