"""Configuration for the document-checker, embedded inside ocr-rag.

Reads env from ``docchecker/.env`` (checker-specific: Byom OAuth, session, model)
and ``<ocr-rag>/.env`` (for ANTHROPIC_API_KEY). Data lives under ocr-rag's data dir
in dedicated ``check_*`` files so ocr-rag's own ``docs.db`` (the company KB) stays
pristine; existing-reference retrieval uses the running company MCP on :8200.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DOCCHECKER_DIR = Path(__file__).resolve().parent           # /btrfs/ocr-rag/docchecker
BASE_DIR = DOCCHECKER_DIR.parent                           # /btrfs/ocr-rag

# checker vars win; ocr-rag's .env fills ANTHROPIC_API_KEY.
load_dotenv(DOCCHECKER_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.environ.get("CHECKER_DATA_DIR", str(BASE_DIR / "data")))
UPLOADS_DIR = DATA_DIR / "check_uploads"
ANNOTATED_DIR = DATA_DIR / "check_annotated"
EXPORTS_DIR = DATA_DIR / "check_exports"

CHECKER_DB = str(DATA_DIR / "checks.db")           # app/audit tables
DOCS_DB = str(DATA_DIR / "check_docs.db")          # ocr-rag schema, checker uploads only

# ocr-rag modules (ingest, file_extractors) are importable from BASE_DIR at runtime;
# pdf-annotator is vendored as docchecker.pdfannotator. No sys.path wiring needed.
OCR_RAG_DIR = BASE_DIR
PDF_ANNOTATOR_DIR = DOCCHECKER_DIR
OCR_RAG_PYTHON = os.environ.get("OCR_RAG_PYTHON", str(BASE_DIR / "venv" / "bin" / "python"))
OCR_RAG_MCP_SERVER = str(BASE_DIR / "mcp_server.py")
CHECKER_MCP_PORT = int(os.environ.get("CHECKER_MCP_PORT", "8200"))
COMPANY_MCP_PORT = int(os.environ.get("COMPANY_MCP_PORT", "8200"))
COMPANY_DOCS_DB = os.environ.get("COMPANY_DOCS_DB", "").strip() or None
# The company knowledge base = ocr-rag's own MCP (same process, :8200).
COMPANY_MCP_URL = os.environ.get("COMPANY_MCP_URL", "http://127.0.0.1:8200/mcp").strip() or None

# Web / session
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-insecure-secret-change-me")
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8201"))

# Auth (Byom OAuth2 / PKCE)
AUTH_MODE = os.environ.get("AUTH_MODE", "oidc").lower()
OIDC_ISSUER = os.environ.get("OIDC_ISSUER", "")
OIDC_CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "")
OIDC_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "")
OIDC_REDIRECT_URI = os.environ.get("OIDC_REDIRECT_URI", "")
OIDC_METADATA_URL = os.environ.get("OIDC_METADATA_URL", "").strip() or (
    f"{OIDC_ISSUER.rstrip('/')}/.well-known/openid-configuration" if OIDC_ISSUER else ""
)
OIDC_SCOPE = os.environ.get("OIDC_SCOPE", "openid email profile")

# Anthropic / models
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CHECKER_MODEL = os.environ.get("CHECKER_MODEL", "claude-opus-4-8")
CHECKER_FAST_MODEL = os.environ.get("CHECKER_FAST_MODEL", "claude-haiku-4-5")
CHECKER_LIVE_AGENT = os.environ.get("CHECKER_LIVE_AGENT", "1").strip() in ("1", "true", "yes")


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOADS_DIR, ANNOTATED_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
