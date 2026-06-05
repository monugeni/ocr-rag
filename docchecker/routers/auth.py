"""Auth routes: /login, /auth/callback, /logout.

Mock mode: /login renders a simple form; POST sets the session.
OIDC mode: /login redirects to the IdP; /auth/callback completes the flow.
"""
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, config

router = APIRouter(tags=["auth"])

_MOCK_LOGIN_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Sign in</title><style>body{font:15px system-ui;display:grid;place-items:center;height:100vh;margin:0;background:#f6f7f9}
form{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:28px;min-width:300px;display:flex;flex-direction:column;gap:12px}
input{padding:9px;border:1px solid #d4d7dd;border-radius:8px}button{padding:10px;border:0;border-radius:8px;background:#4f46e5;color:#fff;font-weight:600;cursor:pointer}</style></head>
<body><form method=post action=/login><h2 style=margin:0>Sign in (dev)</h2>
<input name=email type=email placeholder=you@company.com required>
<input name=display_name placeholder="Display name">
<button>Continue</button></form></body></html>"""


@router.get("/login")
async def login(request: Request):
    if config.AUTH_MODE == "oidc":
        oauth = auth.get_oauth()
        # Redirect back to whatever host the user is actually on (server, IP, domain),
        # so the session cookie set here matches the callback host.
        redirect_uri = config.OIDC_REDIRECT_URI or (
            str(request.base_url).rstrip("/") + "/auth/callback"
        )
        if request.base_url:
            redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
        return await oauth.company.authorize_redirect(request, redirect_uri)
    return HTMLResponse(_MOCK_LOGIN_HTML)


@router.post("/login")
async def login_mock(
    request: Request,
    email: str = Form(...),
    display_name: str = Form(""),
):
    """Mock-mode login: trust the submitted email as identity."""
    if config.AUTH_MODE == "oidc":
        # Don't allow form login in real-auth mode.
        return RedirectResponse("/login", status_code=303)
    sub = f"mock:{email.strip().lower()}"
    user = auth.upsert_user(sub, email.strip(), display_name.strip() or email.strip())
    auth.login_session(request, user)
    auth.record_audit("login", user_id=user["id"], payload={"mode": "mock"})
    return RedirectResponse("/", status_code=303)


@router.get("/auth/callback")
async def auth_callback(request: Request):
    """OIDC callback: exchange the code, validate, upsert the user, set session."""
    oauth = auth.get_oauth()
    token = await oauth.company.authorize_access_token(request)
    claims = token.get("userinfo") or {}
    sub = claims.get("sub") or claims.get("id") or claims.get("email")
    if not sub:
        # Byom is OAuth2 (no userinfo/id_token in the token response) and returns a
        # non-standard token_type that Authlib's client rejects, so fetch /userinfo
        # ourselves with a plain Bearer header.
        import httpx

        meta = await oauth.company.load_server_metadata()
        uinfo_url = meta.get("userinfo_endpoint")
        access_token = token.get("access_token")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(uinfo_url, headers={"Authorization": f"Bearer {access_token}"})
            r.raise_for_status()
            claims = r.json()
        sub = claims.get("sub") or claims.get("id") or claims.get("email")
    if not sub:
        raise HTTPException(status_code=400, detail="OAuth: no subject/email claim returned")
    user = auth.upsert_user(
        sub,
        claims.get("email"),
        claims.get("name") or claims.get("preferred_username"),
    )
    auth.login_session(request, user)
    auth.record_audit("login", user_id=user["id"], payload={"mode": "oidc"})
    return RedirectResponse("/", status_code=303)


@router.get("/api/me")
async def me(request: Request):
    user = auth.current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="not authenticated")
    return {
        "email": user.get("email"),
        "display_name": user.get("display_name"),
        "is_admin": auth.is_admin(user.get("oidc_sub")),
    }


@router.post("/logout")
async def logout(request: Request):
    user = auth.current_user(request)
    if user:
        auth.record_audit("logout", user_id=user["id"])
    auth.logout_session(request)
    return RedirectResponse("/login", status_code=303)
