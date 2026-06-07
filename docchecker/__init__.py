"""Document checker, embedded in ocr-rag's web app.

``register(app)`` wires the checker into ocr-rag's FastAPI app: session cookie,
a whole-app Byom-login gate, the checker JSON routers, and DB init. Call it from
web.py's main() before uvicorn.run().
"""
from __future__ import annotations

import logging

log = logging.getLogger("docchecker")

# Paths that bypass the login gate.
_OPEN_PREFIXES = ("/static", "/auth")
_OPEN_PATHS = {"/login", "/logout", "/healthz", "/openapi.json"}


class _AuthGate:
    """Pure-ASGI whole-app login gate (BaseHTTPMiddleware would break SSE streaming).

    Runs inside SessionMiddleware, so ``scope['session']`` is populated.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        open_ = (
            path in _OPEN_PATHS
            or any(path.startswith(x) for x in _OPEN_PREFIXES)
            or path.startswith("/api/docs")
        )
        if open_ or (scope.get("session") or {}).get("user"):
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse, RedirectResponse

        resp = (
            JSONResponse({"detail": "authentication required"}, status_code=401)
            if path.startswith("/api")
            else RedirectResponse("/login", status_code=303)
        )
        await resp(scope, receive, send)


def register(app) -> None:
    from starlette.middleware.sessions import SessionMiddleware

    from . import config
    from .db import init_databases
    from .routers import auth as auth_router
    from .routers import findings as findings_router
    from .routers import reference as reference_router
    from .routers import runs as runs_router
    from .routers import uploads as uploads_router

    config.ensure_dirs()
    init_databases()

    # Order: gate added first (inner), session added last (outer) → session runs
    # first and populates scope['session'] before the gate reads it.
    app.add_middleware(_AuthGate)
    app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET, same_site="lax")

    # --- checker routers ---
    app.include_router(auth_router.router)
    app.include_router(runs_router.router)
    app.include_router(uploads_router.router)
    app.include_router(findings_router.router)
    app.include_router(reference_router.router)

    @app.get("/healthz")
    def _healthz():
        import os

        ck_prov = (config.CHECKER_PROVIDER or "anthropic").strip().lower()
        if ck_prov == "grok":
            ck_model, ck_fast = config.CHECKER_GROK_MODEL, config.CHECKER_GROK_FAST_MODEL
            ck_key = bool(config.XAI_API_KEY)
        else:
            ck_model, ck_fast = config.CHECKER_MODEL, config.CHECKER_FAST_MODEL
            ck_key = bool(config.ANTHROPIC_API_KEY)

        chat_prov = (os.environ.get("OCR_RAG_CHAT_PROVIDER", "anthropic") or "anthropic").strip().lower()
        if chat_prov == "grok":
            chat_model = os.environ.get("GROK_CHAT_MODEL", "grok-4.3")
            chat_key = bool(os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"))
        else:
            chat_model = os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
            chat_key = bool(config.ANTHROPIC_API_KEY)

        return {
            "status": "ok",
            "auth_mode": config.AUTH_MODE,
            "checker": {
                "enabled": True, "provider": ck_prov,
                "model": ck_model, "fast_model": ck_fast, "key_present": ck_key,
            },
            "chat": {"provider": chat_prov, "model": chat_model, "key_present": chat_key},
        }

    log.info("docchecker registered (auth=%s, model=%s)", config.AUTH_MODE, config.CHECKER_MODEL)
