#!/usr/bin/env bash
# ============================================================================
# OCR-RAG Production Deployment Script (Ubuntu)
# ============================================================================
# Deploys the MCP document server + heuristic extractor on Ubuntu.
#
# Usage:
#   # First time — full install:
#   sudo bash deploy.sh
#
#   # Update code only (after git push):
#   sudo bash deploy.sh --update
#
#   # Ingest a project folder:
#   sudo -u ocrrag bash deploy.sh --ingest /path/to/pdf/folder ProjectName
#
# What it sets up:
#   - System user: ocrrag
#   - App dir:     /opt/ocr-rag
#   - Data dir:    /var/lib/ocr-rag  (database + sidecars)
#   - Venv:        /opt/ocr-rag/venv
#   - Systemd:     ocr-rag-mcp.service (port 8200)
#   - Logs:        journalctl -u ocr-rag-mcp
# ============================================================================

set -euo pipefail

APP_DIR="/opt/ocr-rag"
DATA_DIR="/var/lib/ocr-rag"
VENV_DIR="${APP_DIR}/venv"
SERVICE_NAME="ocr-rag-mcp"
SERVICE_USER="ocrrag"
REPO_URL="https://github.com/monugeni/ocr-rag.git"
MCP_PORT=8200
WEB_PORT=8201
DB_NAME="docs.db"
UPLOADS_DIR="${DATA_DIR}/uploads"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Ingest mode (doesn't need root)
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--ingest" ]]; then
    PDF_DIR="${2:?Usage: deploy.sh --ingest /path/to/pdfs ProjectName}"
    PROJECT="${3:?Usage: deploy.sh --ingest /path/to/pdfs ProjectName}"
    DB_PATH="${DATA_DIR}/${DB_NAME}"

    if [[ ! -d "$PDF_DIR" ]]; then
        err "PDF directory not found: $PDF_DIR"
        exit 1
    fi

    log "Ingesting project '${PROJECT}' from ${PDF_DIR}"

    source "${VENV_DIR}/bin/activate"

    python3 - "$PDF_DIR" "$PROJECT" "$DB_PATH" <<'PYEOF'
import sys, os
sys.path.insert(0, "/opt/ocr-rag")

from pathlib import Path
from ingest import init_db, ingest_document, replay_corrections
from extractor import extract_pdf

pdf_dir = Path(sys.argv[1])
project = sys.argv[2]
db_path = sys.argv[3]

conn = init_db(db_path)

existing = set()
for row in conn.execute("SELECT filename FROM documents WHERE project = ?", (project,)):
    existing.add(row["filename"])

pdfs = sorted(pdf_dir.glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs, {len(existing)} already ingested")

ingested = 0
for pdf in pdfs:
    if pdf.name in existing:
        print(f"  Skip (exists): {pdf.name}")
        continue

    print(f"\n--- {pdf.name} ---")
    try:
        pages, sections = extract_pdf(str(pdf))
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

    if not pages:
        print(f"  No content, skipping")
        continue

    title = pdf.stem.replace('_', ' ').replace('-', ' ')
    doc_id = ingest_document(
        conn, pages, sections, project, title,
        filename=pdf.name, pdf_path=str(pdf.resolve()),
    )
    replay_corrections(conn, doc_id, str(pdf.resolve()))
    ingested += 1

stats = conn.execute(
    "SELECT COUNT(*) as docs, COALESCE(SUM(total_pages),0) as pages "
    "FROM documents WHERE project = ?", (project,)
).fetchone()
print(f"\nProject '{project}': {stats['docs']} docs, {stats['pages']} pages ({ingested} new)")
conn.close()
PYEOF

    log "Ingestion complete. Restart server to pick up changes:"
    log "  sudo systemctl restart ${SERVICE_NAME}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Must be root for install/update
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    err "Run as root: sudo bash deploy.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Update mode — pull latest code, reinstall deps, restart
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--update" ]]; then
    log "Updating code..."
    cd "$APP_DIR"
    sudo -u "$SERVICE_USER" git pull --ff-only
    log "Updating dependencies..."
    sudo -u "$SERVICE_USER" "${VENV_DIR}/bin/pip" install -q -r requirements.txt
    log "Restarting service..."
    systemctl restart "$SERVICE_NAME"
    systemctl --no-pager status "$SERVICE_NAME"
    log "Update complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Full install
# ---------------------------------------------------------------------------

log "=== OCR-RAG Production Deployment ==="

# --- System packages ---
log "Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git \
    ocrmypdf tesseract-ocr tesseract-ocr-eng \
    > /dev/null 2>&1

# --- System user ---
if ! id "$SERVICE_USER" &>/dev/null; then
    log "Creating system user: ${SERVICE_USER}"
    useradd --system --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$SERVICE_USER"
fi

# --- Clone/update repo ---
if [[ -d "${APP_DIR}/.git" ]]; then
    log "Updating existing repo..."
    cd "$APP_DIR"
    sudo -u "$SERVICE_USER" git pull --ff-only || {
        warn "Git pull failed, continuing with existing code"
    }
else
    log "Cloning repo..."
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    chown -R "${SERVICE_USER}:${SERVICE_USER}" "$APP_DIR"
fi

# --- Data directory ---
log "Setting up data directory: ${DATA_DIR}"
mkdir -p "$DATA_DIR"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "$DATA_DIR"

# --- Python venv ---
log "Setting up Python venv..."
if [[ ! -d "$VENV_DIR" ]]; then
    sudo -u "$SERVICE_USER" python3 -m venv "$VENV_DIR"
fi

log "Installing Python dependencies..."
sudo -u "$SERVICE_USER" "${VENV_DIR}/bin/pip" install -q --upgrade pip
sudo -u "$SERVICE_USER" "${VENV_DIR}/bin/pip" install -q -r "${APP_DIR}/requirements.txt"

# --- Initialize database ---
DB_PATH="${DATA_DIR}/${DB_NAME}"
if [[ ! -f "$DB_PATH" ]]; then
    log "Initializing database: ${DB_PATH}"
    sudo -u "$SERVICE_USER" "${VENV_DIR}/bin/python3" -c "
import sys; sys.path.insert(0, '${APP_DIR}')
from ingest import init_db
conn = init_db('${DB_PATH}')
conn.close()
print('Database initialized')
"
else
    log "Database exists: ${DB_PATH}"
    # Run schema migration (adds new tables if missing)
    sudo -u "$SERVICE_USER" "${VENV_DIR}/bin/python3" -c "
import sys; sys.path.insert(0, '${APP_DIR}')
from ingest import init_db
conn = init_db('${DB_PATH}')
conn.close()
print('Schema updated')
"
fi

# --- Environment file ---
ENV_FILE="${APP_DIR}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    log "Creating .env file..."
    cat > "$ENV_FILE" <<EOF
# Anthropic API key for LLM metadata extraction during ingestion
# ANTHROPIC_API_KEY=sk-ant-...

# MCP server port
MCP_PORT=${MCP_PORT}

# Web GUI port
WEB_PORT=${WEB_PORT}
EOF
    chown "${SERVICE_USER}:${SERVICE_USER}" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    warn "Edit ${ENV_FILE} to add your ANTHROPIC_API_KEY"
fi

# --- Uploads directory ---
log "Setting up uploads directory: ${UPLOADS_DIR}"
mkdir -p "$UPLOADS_DIR"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "$UPLOADS_DIR"

# --- Systemd service (Web GUI + MCP in one process) ---
log "Creating systemd service: ${SERVICE_NAME}"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=OCR-RAG Server (Web GUI + MCP)
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV_DIR}/bin/python3 ${APP_DIR}/web.py --db ${DB_PATH} --port ${WEB_PORT} --mcp-port ${MCP_PORT} --uploads-dir ${UPLOADS_DIR}
Restart=on-failure
RestartSec=5

NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=${DATA_DIR}
PrivateTmp=yes

StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

# Remove old separate web service if it exists
if [[ -f "/etc/systemd/system/ocr-rag-web.service" ]]; then
    systemctl stop ocr-rag-web 2>/dev/null || true
    systemctl disable ocr-rag-web 2>/dev/null || true
    rm -f "/etc/systemd/system/ocr-rag-web.service"
fi

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

# --- Wait and verify ---
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "Service is running!"
else
    err "Service failed to start. Check: journalctl -u ${SERVICE_NAME} -n 20 --no-pager"
    exit 1
fi

# --- Summary ---
echo ""
echo "============================================================"
log "Deployment complete!"
echo "============================================================"
echo ""
echo "  App:       ${APP_DIR}"
echo "  Data:      ${DATA_DIR}"
echo "  Database:  ${DB_PATH}"
echo "  Uploads:   ${UPLOADS_DIR}"
echo ""
echo "  Web GUI:     http://localhost:${WEB_PORT}"
echo "  MCP:         http://localhost:${MCP_PORT}/sse"
echo ""
echo "  Commands:"
echo "    sudo systemctl status ${SERVICE_NAME}"
echo "    sudo systemctl restart ${SERVICE_NAME}"
echo "    sudo journalctl -u ${SERVICE_NAME} -f"
echo "    sudo bash deploy.sh --update"
echo ""
echo "  Ingest PDFs (via web GUI or CLI):"
echo "    sudo -u ${SERVICE_USER} bash ${APP_DIR}/deploy.sh --ingest /path/to/pdfs MyProject"
echo ""
if [[ ! -f "${ENV_FILE}" ]] || ! grep -q "^ANTHROPIC_API_KEY=sk-" "${ENV_FILE}" 2>/dev/null; then
    warn "Set ANTHROPIC_API_KEY in ${ENV_FILE} for LLM metadata extraction"
fi
echo ""
