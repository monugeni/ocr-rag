"""Authentication: company OIDC (Authlib) with a dev-mock fallback.

`AUTH_MODE=mock` (default) lets developers log in by entering an email so the rest
of the app can be built before the company IdP is provisioned. `AUTH_MODE=oidc`
uses Authlib's Starlette integration against `OIDC_ISSUER`.

Also holds user-upsert and audit helpers since both are auth-adjacent.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from . import config
from .db import get_conn

# ---------------------------------------------------------------------------
# OIDC client (lazy — only built in oidc mode)
# ---------------------------------------------------------------------------
_oauth = None


def get_oauth():
    """Return a configured Authlib OAuth registry (oidc mode only)."""
    global _oauth
    if _oauth is not None:
        return _oauth
    if not (config.OIDC_CLIENT_ID and config.OIDC_METADATA_URL):
        raise RuntimeError(
            "AUTH_MODE=oidc requires OIDC_CLIENT_ID and OIDC_METADATA_URL "
            "(or OIDC_ISSUER) in .env"
        )
    from authlib.integrations.starlette_client import OAuth

    # Public client (PKCE) when no secret is configured.
    is_public = not config.OIDC_CLIENT_SECRET
    client_kwargs = {
        "scope": config.OIDC_SCOPE,
        "code_challenge_method": "S256",
        "token_endpoint_auth_method": "none" if is_public else "client_secret_post",
    }
    oauth = OAuth()
    oauth.register(
        name="company",
        client_id=config.OIDC_CLIENT_ID,
        client_secret=config.OIDC_CLIENT_SECRET or None,
        server_metadata_url=config.OIDC_METADATA_URL,
        client_kwargs=client_kwargs,
    )
    _oauth = oauth
    return _oauth


# ---------------------------------------------------------------------------
# User store + audit
# ---------------------------------------------------------------------------
def upsert_user(oidc_sub: str, email: str | None, display_name: str | None) -> dict:
    """Insert or update a user by oidc_sub; bump last_login. Returns the row as dict."""
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO users (oidc_sub, email, display_name, last_login)
                 VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(oidc_sub) DO UPDATE SET
                 email=excluded.email,
                 display_name=excluded.display_name,
                 last_login=CURRENT_TIMESTAMP""",
            (oidc_sub, email, display_name),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE oidc_sub = ?", (oidc_sub,)
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def record_audit(
    event_type: str,
    *,
    user_id: int | None = None,
    run_id: str | None = None,
    payload: dict | None = None,
) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO audit_events (run_id, user_id, event_type, payload) VALUES (?, ?, ?, ?)",
            (run_id, user_id, event_type, json.dumps(payload or {})),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session helpers + dependencies
# ---------------------------------------------------------------------------
def login_session(request: Request, user: dict) -> None:
    request.session["user"] = {
        "id": user["id"],
        "oidc_sub": user["oidc_sub"],
        "email": user.get("email"),
        "display_name": user.get("display_name"),
        "is_admin": bool(user.get("is_admin")),
    }


def is_admin(oidc_sub: str) -> bool:
    """Fresh admin lookup from the DB (so grants take effect without re-login)."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT is_admin FROM users WHERE oidc_sub = ?", (oidc_sub,)).fetchone()
        return bool(row and row["is_admin"])
    finally:
        conn.close()


def get_user_email(user_id: int | None) -> str | None:
    """Email for a user id (for spend attribution), or None."""
    if user_id is None:
        return None
    conn = get_conn()
    try:
        row = conn.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
        return row["email"] if row else None
    finally:
        conn.close()


def logout_session(request: Request) -> None:
    request.session.pop("user", None)


def current_user(request: Request) -> Optional[dict]:
    """Return the logged-in user dict, or None."""
    return request.session.get("user")


def require_user(request: Request) -> dict:
    """FastAPI dependency: require an authenticated user (401 if missing)."""
    user = current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"Location": "/login"},
        )
    return user


CurrentUser = Depends(require_user)
