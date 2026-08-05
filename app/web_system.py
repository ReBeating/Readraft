from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from .config import Settings
from .db import Database


logger = logging.getLogger(__name__)


def build_system_router(*, database: Database, settings: Settings) -> APIRouter:
    """Expose process health without coupling it to business routes."""
    router = APIRouter()

    @router.get("/healthz", include_in_schema=False)
    async def healthz(request: Request):
        worker = getattr(request.app.state, "worker", None)
        worker_task = getattr(request.app.state, "worker_task", None)
        try:
            database_ok = await asyncio.to_thread(database.ping)
        except Exception:
            logger.exception("health check database failure")
            database_ok = False
        worker_ok = bool(
            worker and worker.healthy and worker_task and not worker_task.done()
        )
        healthy = database_ok and worker_ok
        return JSONResponse(
            {
                "status": "ok" if healthy else "unhealthy",
                "model_provider": (
                    "mock"
                    if settings.uses_test_models
                    else (
                        settings.model_provider
                        if settings.model_api_key
                        else "personal-key-only"
                    )
                ),
                "database": "ok" if database_ok else "unavailable",
                "worker": "ok" if worker_ok else "unavailable",
            },
            status_code=(
                status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
        )

    return router
