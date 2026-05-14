#!/bin/bash
# Setup and compute embeddings for all existing projects.
#
# Local dev:
#   bash setup_embeddings.sh [db_path]
#
# Prod (auto-detected via /btrfs/ocr-rag/venv):
#   sudo bash setup_embeddings.sh
#   sudo bash setup_embeddings.sh /btrfs/ocr-rag/data/docs.db
#
# On prod the script installs into the ocrrag venv, pulling the CPU-only torch
# wheel from pytorch.org (the default PyPI torch is the CUDA build, ~2GB of
# unused libs on a CPU-only host).
set -e

PROD_VENV="/btrfs/ocr-rag/venv"
PROD_APP="/btrfs/ocr-rag"
PROD_DB="/btrfs/ocr-rag/data/docs.db"
PROD_USER="ocrrag"

if [[ -x "${PROD_VENV}/bin/python" ]]; then
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "ERROR: prod venv detected; run as root (sudo)." >&2
        exit 1
    fi
    PYTHON="${PROD_VENV}/bin/python"
    PIP="${PROD_VENV}/bin/pip"
    RUNAS=(sudo -u "$PROD_USER")
    DB="${1:-$PROD_DB}"
    APP_DIR="$PROD_APP"
    cd "$APP_DIR"
    echo "Mode: prod (venv $PROD_VENV, user $PROD_USER)"
else
    PYTHON="python3"
    PIP="pip"
    RUNAS=()
    DB="${1:-docs.db}"
    APP_DIR="$(pwd)"
    echo "Mode: local"
fi

echo "=== Embedding Setup ==="
echo "Python:   $PYTHON"
echo "Database: $DB"
echo ""

if "${RUNAS[@]}" "$PYTHON" -c "import sentence_transformers" 2>/dev/null; then
    echo "sentence-transformers: already installed"
else
    echo "Installing torch (CPU-only wheel)..."
    "${RUNAS[@]}" "$PIP" install --index-url https://download.pytorch.org/whl/cpu torch
    echo "Installing sentence-transformers..."
    "${RUNAS[@]}" "$PIP" install sentence-transformers
fi

echo ""

"${RUNAS[@]}" env APP_DIR="$APP_DIR" DB="$DB" "$PYTHON" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["APP_DIR"])
from ingest import init_db, compute_embeddings_for_project

db = os.environ["DB"]
conn = init_db(db)
projects = conn.execute(
    'SELECT DISTINCT project FROM documents ORDER BY project'
).fetchall()

if not projects:
    print('No projects found in database.')
    sys.exit(0)

for row in projects:
    project = row['project']
    doc_count = conn.execute(
        'SELECT COUNT(*) FROM documents WHERE project = ?', (project,)
    ).fetchone()[0]
    page_count = conn.execute(
        'SELECT COUNT(*) FROM pages p '
        'JOIN documents d ON d.id = p.doc_id WHERE d.project = ?',
        (project,),
    ).fetchone()[0]
    print('\n' + '=' * 60)
    print(f'Project: {project}')
    print(f'Documents: {doc_count}, Pages: {page_count}')
    print('=' * 60)
    compute_embeddings_for_project(conn, project)

total = conn.execute('SELECT COUNT(*) FROM page_embeddings').fetchone()[0]
print('\n' + '=' * 60)
print(f'Done. Total embeddings: {total}')
print('=' * 60)
conn.close()
PYEOF
