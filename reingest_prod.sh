#!/usr/bin/env bash
# ============================================================================
# Reingest all docs in the prod DB through the fast Poppler pipeline.
# ============================================================================
# Usage (on the prod server, as root):
#   sudo bash reingest_prod.sh                      # all docs, no LLM, no embeddings
#   sudo bash reingest_prod.sh --project EPL-251    # one folder (and its sub-folders)
#   sudo bash reingest_prod.sh --with-embeddings --ocr
#   sudo bash reingest_prod.sh --limit 5            # dry-run a few first
#
# Any flags after the script name are forwarded to reingest_fast.py.
# Run `sudo -u ocrrag /btrfs/ocr-rag/venv/bin/python /btrfs/ocr-rag/reingest_fast.py --help`
# for the full list.
#
# ANTHROPIC_API_KEY for --with-llm-metadata is read from /btrfs/ocr-rag/.env
# automatically (ingest.py calls load_dotenv()).  Override with:
#   sudo ANTHROPIC_API_KEY=sk-ant-... bash reingest_prod.sh --with-llm-metadata
#
# What this does:
#   1. Stops ocr-rag-mcp so the DB isn't written while the server reads it.
#   2. cd's to /btrfs/ocr-rag so load_dotenv() picks up .env.
#   3. Runs reingest_fast.py as the ocrrag service user (preserves file ownership);
#      reingest_fast.py itself snapshots the DB to .before-fast-reingest-<stamp>.bak.
#   4. Restarts the service on exit, even if reingest fails.
# ============================================================================

set -euo pipefail

APP_DIR="/btrfs/ocr-rag"
DATA_DIR="${APP_DIR}/data"
VENV_DIR="${APP_DIR}/venv"
SERVICE_NAME="ocr-rag-mcp"
SERVICE_USER="ocrrag"
DB_PATH="${DATA_DIR}/docs.db"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: must run as root (use sudo)." >&2
    exit 1
fi

for path in "$APP_DIR" "$DATA_DIR" "$VENV_DIR" "$DB_PATH" "${APP_DIR}/reingest_fast.py"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: expected path not found: $path" >&2
        echo "       Run 'sudo bash deploy.sh' (or '--update') first." >&2
        exit 1
    fi
done

if ! id "$SERVICE_USER" &>/dev/null; then
    echo "ERROR: service user '$SERVICE_USER' does not exist." >&2
    exit 1
fi

# Pre-flight: if --with-embeddings is requested, fail fast before touching the
# service if sentence-transformers isn't installed in the venv.
for arg in "$@"; do
    if [[ "$arg" == "--with-embeddings" ]]; then
        if ! sudo -u "$SERVICE_USER" "${VENV_DIR}/bin/python" -c "import sentence_transformers" 2>/dev/null; then
            echo "ERROR: --with-embeddings requested but sentence_transformers is not importable in ${VENV_DIR}." >&2
            echo "       Install with: sudo bash setup_embeddings.sh" >&2
            exit 1
        fi
        break
    fi
done

echo "[+] Stopping ${SERVICE_NAME}"
systemctl stop "$SERVICE_NAME" || true

restart_service() {
    local status=$?
    echo "[+] Restarting ${SERVICE_NAME}"
    if systemctl start "$SERVICE_NAME"; then
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo "[+] ${SERVICE_NAME} is running."
        else
            echo "WARNING: ${SERVICE_NAME} not active. Check: journalctl -u ${SERVICE_NAME} -n 50 --no-pager" >&2
        fi
    else
        echo "ERROR: failed to restart ${SERVICE_NAME}. Check: journalctl -u ${SERVICE_NAME} -n 50 --no-pager" >&2
    fi
    exit "$status"
}
trap restart_service EXIT

echo "[+] Reingesting through fast pipeline"
echo "    DB:   ${DB_PATH}"
echo "    Root: ${DATA_DIR}"
echo "    User: ${SERVICE_USER}"
echo "    Args: $*"

# cd into APP_DIR so load_dotenv() picks up /btrfs/ocr-rag/.env.
# Preserve ANTHROPIC_API_KEY in case the caller overrides it inline.
cd "$APP_DIR"
sudo -u "$SERVICE_USER" \
    --preserve-env=ANTHROPIC_API_KEY \
    "${VENV_DIR}/bin/python" "${APP_DIR}/reingest_fast.py" \
    --db "$DB_PATH" \
    --root "$DATA_DIR" \
    "$@"
