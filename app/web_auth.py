from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .config import Settings
from .db import Database
from .security import (
    csrf_token,
    hash_password,
    validate_password,
    validate_username,
    verify_csrf,
    verify_password,
)
from .web_paths import safe_next
from .web_security import SlidingWindowLimiter, client_address, current_user


logger = logging.getLogger(__name__)

TemplateContext = Callable[..., dict[str, Any]]
TemplateRenderer = Callable[..., Response]


def build_auth_router(
    *,
    database: Database,
    settings: Settings,
    template_context: TemplateContext,
    render_template: TemplateRenderer,
) -> APIRouter:
    """Build the account routes around one shared authentication policy."""
    router = APIRouter()
    limiter = SlidingWindowLimiter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        destination = "/dashboard" if current_user(request) else "/login"
        return RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, next: str = "/"):
        destination = safe_next(next)
        if current_user(request):
            return RedirectResponse(
                destination,
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return render_template(
            "login.html",
            template_context(
                request,
                next_path=destination,
                can_register=settings.allow_registration,
            ),
        )

    @router.post("/login", response_class=HTMLResponse)
    async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        csrf: str = Form(...),
        next: str = Form("/"),
    ):
        verify_csrf(request, csrf)
        address = client_address(request)
        clean_login = username.strip()
        if not limiter.allow(
            f"login-ip:{address}", limit=20, window_seconds=300
        ) or not limiter.allow(
            f"login-user:{address}:{clean_login.casefold()}",
            limit=10,
            window_seconds=300,
        ):
            return render_template(
                "login.html",
                template_context(
                    request,
                    error="尝试次数过多，请稍后再试",
                    next_path=safe_next(next),
                    can_register=settings.allow_registration,
                ),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        user = database.get_user_by_username(clean_login)
        password_ok = False
        if user:
            async with request.app.state.password_slots:
                password_ok = await asyncio.to_thread(
                    verify_password,
                    password,
                    str(user["password_hash"]),
                )
        if not user or not password_ok:
            return render_template(
                "login.html",
                template_context(
                    request,
                    error="用户名或密码不正确",
                    next_path=safe_next(next),
                    can_register=settings.allow_registration,
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        request.session.clear()
        request.session["user_id"] = int(user["id"])
        csrf_token(request)
        return RedirectResponse(
            safe_next(next),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request):
        if current_user(request):
            return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        if not settings.allow_registration:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        return render_template("register.html", template_context(request))

    @router.post("/register", response_class=HTMLResponse)
    async def register(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        password_confirm: str = Form(...),
        csrf: str = Form(...),
    ):
        verify_csrf(request, csrf)
        if not settings.allow_registration:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        if not limiter.allow(
            f"register:{client_address(request)}",
            limit=5,
            window_seconds=60 * 60,
        ):
            return render_template(
                "register.html",
                template_context(
                    request,
                    error="注册尝试次数过多，请稍后再试",
                    username=username,
                ),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        try:
            clean_username = validate_username(username)
            validate_password(password)
            if password != password_confirm:
                raise ValueError("两次输入的密码不一致")
            async with request.app.state.password_slots:
                password_hash = await asyncio.to_thread(hash_password, password)
            user_id = database.create_user(clean_username, password_hash)
        except ValueError as exc:
            return render_template(
                "register.html",
                template_context(request, error=str(exc), username=username),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            if not duplicate:
                logger.exception("failed to register user")
            return render_template(
                "register.html",
                template_context(
                    request,
                    error=(
                        "该用户名已存在" if duplicate else "创建账号失败，请稍后重试"
                    ),
                    username=username,
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        request.session.clear()
        request.session["user_id"] = user_id
        csrf_token(request)
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    @router.post("/logout")
    async def logout(request: Request, csrf: str = Form(...)):
        verify_csrf(request, csrf)
        request.session.clear()
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    return router
