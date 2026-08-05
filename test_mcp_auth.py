"""Security regression tests for the OAuth-protected OCR-RAG MCP transport."""

from __future__ import annotations

import base64
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.server.fastmcp.exceptions import ToolError

import ocr_mcp_auth as auth


ISSUER = "https://oauth.example.test"
AUDIENCE = "https://byom.example.test"


def _token(**overrides) -> str:
    now = int(time.time())
    claims = {
        "sub": "alice@example.test",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "iat": now,
        "exp": now + 300,
        "type": "access",
        "scope": "openid imap offline_access",
        "client_id": "claude",
    }
    claims.update(overrides)
    for claim in overrides.get("_remove", ()):
        claims.pop(claim, None)
    claims.pop("_remove", None)

    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{encode({'alg': 'HS256', 'typ': 'JWT'})}.{encode(claims)}.signature"


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeAsyncClient:
    payload = {"active": True, "sub": "alice@example.test"}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, *args, **kwargs):
        return _FakeResponse(self.payload)


class ByomTokenValidationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = [
            patch.object(auth, "OAUTH_ISSUER", ISSUER),
            patch.object(auth, "OAUTH_TOKEN_AUDIENCE", AUDIENCE),
            patch.object(auth, "OAUTH_REQUIRED_SCOPE", "imap"),
            patch.object(auth.httpx, "AsyncClient", _FakeAsyncClient),
        ]
        for item in self.settings:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.settings):
            item.stop()

    async def test_valid_byom_access_token_returns_trusted_context(self):
        context = await auth.ByomTokenValidator(cache_seconds=0).validate(_token())
        self.assertEqual(context.user, "alice@example.test")
        self.assertEqual(context.client_id, "claude")

    async def test_rejects_inactive_token(self):
        with patch.object(_FakeAsyncClient, "payload", {"active": False}):
            with self.assertRaises(auth.InvalidToken):
                await auth.ByomTokenValidator(cache_seconds=0).validate(_token())

    async def test_rejects_invalid_security_claims(self):
        invalid_tokens = {
            "wrong issuer": _token(iss="https://attacker.example.test"),
            "invalid issuer type": _token(iss=None),
            "wrong audience": _token(aud="https://other.example.test"),
            "expired": _token(exp=int(time.time()) - 1),
            "future issued-at": _token(iat=int(time.time()) + 120),
            "refresh token": _token(type="refresh"),
            "missing scope": _token(_remove=("scope",)),
            "missing required scope": _token(scope="openid email profile"),
            "missing expiry": _token(_remove=("exp",)),
            "missing subject": _token(_remove=("sub",)),
        }
        for name, token in invalid_tokens.items():
            with self.subTest(name=name):
                with self.assertRaises(auth.InvalidToken):
                    await auth.ByomTokenValidator(cache_seconds=0).validate(token)


class MCPMiddlewareTests(unittest.TestCase):
    def setUp(self):
        inner = FastAPI()

        @inner.post("/mcp")
        async def endpoint():
            return {"ok": True}

        self.enabled = patch.object(auth, "OAUTH_ENABLED", True)
        self.issuer = patch.object(auth, "OAUTH_ISSUER", ISSUER)
        self.resource = patch.object(auth, "MCP_RESOURCE_URL", "https://ocr.example.test")
        self.enabled.start()
        self.issuer.start()
        self.resource.start()
        self.client = TestClient(auth.protect_mcp_app(inner))

    def tearDown(self):
        self.client.close()
        self.resource.stop()
        self.issuer.stop()
        self.enabled.stop()

    def test_missing_and_malformed_bearer_tokens_are_rejected(self):
        for value in (None, "Bearer", "Bearer ", "Basic abc", "Bearer abc def"):
            with self.subTest(value=value):
                headers = {"Authorization": value} if value is not None else {}
                response = self.client.post("/mcp", headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertIn("resource_metadata=", response.headers["www-authenticate"])
                self.assertIn('scope="imap"', response.headers["www-authenticate"])

    def test_internal_credential_allows_trusted_local_client(self):
        response = self.client.post("/mcp", headers=auth.internal_mcp_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    def test_validated_byom_token_reaches_mcp_app(self):
        with patch.object(auth.validator, "validate", AsyncMock(return_value=auth.AuthContext("alice@example.test"))):
            response = self.client.post("/mcp", headers={"Authorization": "Bearer public-token"})
        self.assertEqual(response.status_code, 200)

    def test_protected_resource_metadata_points_to_byom(self):
        response = self.client.get("/.well-known/oauth-protected-resource")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "resource": "https://ocr.example.test",
            "authorization_servers": [ISSUER],
            "scopes_supported": ["imap"],
        })

    def test_challenge_metadata_stays_at_origin_for_path_resource(self):
        with patch.object(auth, "MCP_RESOURCE_URL", "https://ocr.example.test/mcp"):
            response = self.client.post("/mcp")
        challenge = response.headers["www-authenticate"]
        self.assertIn('realm="https://ocr.example.test/mcp"', challenge)
        self.assertIn(
            'resource_metadata="https://ocr.example.test/.well-known/oauth-protected-resource"',
            challenge,
        )


class MCPAdministratorTests(unittest.TestCase):
    def _context(self, context: auth.AuthContext):
        class ContextManager:
            def __enter__(inner_self):
                inner_self.token = auth._mcp_auth_context.set(context)

            def __exit__(inner_self, exc_type, exc, tb):
                auth._mcp_auth_context.reset(inner_self.token)

        return ContextManager()

    def test_non_admin_byom_user_cannot_run_correction_tools(self):
        from docchecker import auth as user_auth

        with (
            self._context(auth.AuthContext(user="alice@example.test")),
            patch.object(user_auth, "is_admin", return_value=False) as is_admin,
            self.assertRaises(ToolError),
        ):
            auth.require_mcp_admin()
        is_admin.assert_called_once_with("alice@example.test")

    def test_admin_byom_user_can_run_correction_tools(self):
        from docchecker import auth as user_auth

        with (
            self._context(auth.AuthContext(user="admin@example.test")),
            patch.object(user_auth, "is_admin", return_value=True) as is_admin,
        ):
            auth.require_mcp_admin()
        is_admin.assert_called_once_with("admin@example.test")

    def test_trusted_internal_mcp_client_can_run_correction_tools(self):
        with self._context(auth.AuthContext(user="ocr-rag-internal", internal=True)):
            auth.require_mcp_admin()


class CorrectionToolGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_correction_tool_invokes_admin_gate_before_database(self):
        from mcp.server.fastmcp import FastMCP

        from corrections import register_correction_tools

        server = FastMCP("correction-gate-test")

        def database_must_not_open():
            raise AssertionError("database opened before admin authorization")

        register_correction_tools(server, database_must_not_open)
        with (
            patch.object(
                auth,
                "require_mcp_admin",
                side_effect=ToolError("Administrator access required"),
            ) as require_admin,
            self.assertRaises(ToolError),
        ):
            await server.call_tool(
                "suggest_reocr",
                {"doc_id": 1, "reason": "test"},
            )
        require_admin.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
