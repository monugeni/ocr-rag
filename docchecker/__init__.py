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
        return {"status": "ok", "checker": True, "auth_mode": config.AUTH_MODE}

    log.info("docchecker registered (auth=%s, model=%s)", config.AUTH_MODE, config.CHECKER_MODEL)
