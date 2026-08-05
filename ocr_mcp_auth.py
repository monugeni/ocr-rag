"""OAuth protection for OCR-RAG's streamable-HTTP MCP transport.

The existing BYOM authorization server remains the source of truth.  Public
bearer tokens are checked through its introspection endpoint, so OCR-RAG does
not need a copy of BYOM's JWT signing secret.  In-process OCR-RAG components
use a separate random bearer token that never leaves the host.
"""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
import httpx
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send


load_dotenv()

log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


OAUTH_ENABLED = _env_bool("OCR_RAG_MCP_OAUTH_ENABLED", True)
OAUTH_ISSUER = os.environ.get(
    "OCR_RAG_OAUTH_ISSUER", "https://oauth.esteem.co.in"
).rstrip("/")
OAUTH_INTROSPECTION_URL = os.environ.get(
    "OCR_RAG_OAUTH_INTROSPECTION_URL", f"{OAUTH_ISSUER}/introspect"
).strip()
OAUTH_TOKEN_AUDIENCE = os.environ.get(
    "OCR_RAG_OAUTH_TOKEN_AUDIENCE", "https://byom.esteem.co.in"
).rstrip("/")
OAUTH_REQUIRED_SCOPE = os.environ.get(
    "OCR_RAG_OAUTH_REQUIRED_SCOPE", "imap"
).strip()
MCP_RESOURCE_URL = os.environ.get("OCR_RAG_MCP_RESOURCE_URL", "").rstrip("/")

# This credential is for trusted localhost clients such as the document checker.
# If deployment did not provide one, generate it once per process and propagate it
# through the environment so child MCP processes inherit the same value.
INTERNAL_MCP_TOKEN = os.environ.get("OCR_RAG_INTERNAL_MCP_TOKEN", "").strip()
if not INTERNAL_MCP_TOKEN:
    INTERNAL_MCP_TOKEN = secrets.token_urlsafe(48)
    os.environ["OCR_RAG_INTERNAL_MCP_TOKEN"] = INTERNAL_MCP_TOKEN


class InvalidToken(Exception):
    """The supplied bearer token is not an acceptable BYOM access token."""


class AuthorizationServerUnavailable(Exception):
    """The BYOM authorization server could not validate a token."""


def internal_mcp_headers() -> dict[str, str]:
    """Authorization headers for trusted OCR-RAG components on this host."""
    return {"Authorization": f"Bearer {INTERNAL_MCP_TOKEN}"}


def _jwt_claims_without_verification(token: str) -> dict[str, Any]:
    """Decode claims only after BYOM has declared this exact token active."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("not a JWT")
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        if not isinstance(claims, dict):
            raise ValueError("JWT payload is not an object")
        return claims
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidToken("Malformed access token") from exc


def _audiences(claim: Any) -> set[str]:
    if isinstance(claim, str):
        return {claim.rstrip("/")}
    if isinstance(claim, list):
        return {item.rstrip("/") for item in claim if isinstance(item, str)}
    return set()


@dataclass(frozen=True)
class AuthContext:
    user: str
    client_id: str = ""
    internal: bool = False


_mcp_auth_context: contextvars.ContextVar[AuthContext | None] = contextvars.ContextVar(
    "ocr_rag_mcp_auth_context", default=None
)


def require_mcp_admin() -> None:
    """Reject document-mutating MCP tools unless the caller is an admin."""
    context = _mcp_auth_context.get()
    allowed = bool(context and context.internal)
    if context and not allowed:
        try:
            from docchecker import auth

            allowed = auth.is_admin(context.user)
        except Exception as exc:  # Fail closed if the admin store is unavailable.
            log.warning("Could not verify MCP administrator %s: %s", context.user, exc)
            allowed = False
    if not allowed:
        from mcp.server.fastmcp.exceptions import ToolError

        raise ToolError("Administrator access required for document correction tools")


class ByomTokenValidator:
    """Validate and briefly cache BYOM access-token introspection results."""

    def __init__(self, cache_seconds: float = 30.0):
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, AuthContext]] = {}

    async def validate(self, token: str) -> AuthContext:
        # Opportunistically discard expired cache entries so repeated valid
        # sessions cannot leave an ever-growing token dictionary behind.
        now = time.time()
        if len(self._cache) > 256:
            self._cache = {
                key: value for key, value in self._cache.items() if value[0] > now
            }
        cached = self._cache.get(token)
        if cached and cached[0] > now:
            return cached[1]

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(
                    OAUTH_INTROSPECTION_URL,
                    data={"token": token},
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.error("BYOM token introspection failed: %s", exc)
            raise AuthorizationServerUnavailable from exc

        if not isinstance(result, dict) or result.get("active") is not True:
            raise InvalidToken("Inactive access token")

        claims = _jwt_claims_without_verification(token)
        subject = result.get("sub")
        if (
            not isinstance(subject, str)
            or not subject.strip()
            or claims.get("sub") != subject
        ):
            raise InvalidToken("Invalid token subject")
        issuer = claims.get("iss")
        if not isinstance(issuer, str) or issuer.rstrip("/") != OAUTH_ISSUER:
            raise InvalidToken("Wrong token issuer")
        if OAUTH_TOKEN_AUDIENCE and OAUTH_TOKEN_AUDIENCE not in _audiences(claims.get("aud")):
            raise InvalidToken("Wrong token audience")
        if claims.get("type") != "access":
            raise InvalidToken("Not an access token")

        issued_at = claims.get("iat")
        expires_at = claims.get("exp")
        if not isinstance(issued_at, (int, float)) or not isinstance(expires_at, (int, float)):
            raise InvalidToken("Missing token timestamps")
        if issued_at > now + 60 or expires_at <= now:
            raise InvalidToken("Expired or not-yet-valid token")

        scope = claims.get("scope")
        if not isinstance(scope, str) or (
            OAUTH_REQUIRED_SCOPE and OAUTH_REQUIRED_SCOPE not in scope.split()
        ):
            raise InvalidToken("Missing required scope")

        client_id = claims.get("client_id", "")
        context = AuthContext(
            user=subject.strip(),
            client_id=client_id if isinstance(client_id, str) else "",
        )
        cache_until = min(now + self.cache_seconds, float(expires_at))
        self._cache[token] = (cache_until, context)
        return context


validator = ByomTokenValidator()


def _request_base_url(scope: Scope) -> str:
    if MCP_RESOURCE_URL:
        return MCP_RESOURCE_URL
    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
    scheme = headers.get("x-forwarded-proto") or scope.get("scheme", "http")
    host = headers.get("x-forwarded-host") or headers.get("host", "localhost")
    return f"{scheme}://{host}".rstrip("/")


def _metadata_url(scope: Scope) -> str:
    resource = urlsplit(_request_base_url(scope))
    origin = f"{resource.scheme}://{resource.netloc}"
    return f"{origin}/.well-known/oauth-protected-resource"


def _challenge(scope: Scope) -> str:
    resource = _request_base_url(scope)
    pieces = [
        f'Bearer realm="{resource}"',
        f'resource_metadata="{_metadata_url(scope)}"',
    ]
    if OAUTH_REQUIRED_SCOPE:
        pieces.append(f'scope="{OAUTH_REQUIRED_SCOPE}"')
    return ", ".join(pieces)


async def _authenticate(scope: Scope) -> AuthContext:
    request = Request(scope)
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not separator
        or not token
        or any(char.isspace() for char in token)
    ):
        raise InvalidToken("Bearer token required")

    if secrets.compare_digest(token, INTERNAL_MCP_TOKEN):
        return AuthContext(
            user="ocr-rag-internal",
            client_id="ocr-rag-internal",
            internal=True,
        )
    return await validator.validate(token)


def protect_mcp_app(app: ASGIApp) -> ASGIApp:
    """Wrap a FastMCP ASGI app with OAuth discovery and bearer enforcement."""

    class OAuthMiddleware:
        def __init__(self, inner: ASGIApp):
            self.app = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            path = scope.get("path", "")
            if path in {
                "/.well-known/oauth-protected-resource",
                "/.well-known/oauth-protected-resource/mcp",
            }:
                response = JSONResponse(
                    {
                        "resource": _request_base_url(scope),
                        "authorization_servers": [OAUTH_ISSUER],
                        "scopes_supported": [OAUTH_REQUIRED_SCOPE]
                        if OAUTH_REQUIRED_SCOPE
                        else [],
                    },
                    headers={"Access-Control-Allow-Origin": "*"},
                )
                await response(scope, receive, send)
                return

            if scope.get("method") == "OPTIONS":
                response = Response(
                    status_code=204,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Accept, Authorization, Content-Type, Last-Event-ID, MCP-Protocol-Version, MCP-Session-Id",
                    },
                )
                await response(scope, receive, send)
                return

            if not OAUTH_ENABLED:
                context_token = _mcp_auth_context.set(
                    AuthContext(user="local-development", client_id="local-development", internal=True)
                )
                try:
                    await self.app(scope, receive, send)
                finally:
                    _mcp_auth_context.reset(context_token)
                return

            try:
                auth_context = await _authenticate(scope)
            except InvalidToken as exc:
                log.warning("Rejected MCP request: %s", exc)
                response = JSONResponse(
                    {"detail": "Authentication required. Provide a valid BYOM bearer token."},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": _challenge(scope),
                        "Access-Control-Allow-Origin": "*",
                    },
                )
                await response(scope, receive, send)
                return
            except AuthorizationServerUnavailable:
                response = JSONResponse(
                    {"detail": "Authorization server temporarily unavailable."},
                    status_code=503,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
                await response(scope, receive, send)
                return

            context_token = _mcp_auth_context.set(auth_context)
            try:
                await self.app(scope, receive, send)
            finally:
                _mcp_auth_context.reset(context_token)

    return OAuthMiddleware(app)


def run_protected_mcp(mcp: Any, *, host: str, port: int) -> None:
    """Run a FastMCP streamable-HTTP app behind the OAuth middleware."""
    import uvicorn

    app = protect_mcp_app(mcp.streamable_http_app())
    uvicorn.run(app, host=host, port=port, log_level="info")
