#!/bin/bash
# Setup and compute embeddings for all existing projects
# Run: bash setup_embeddings.sh [db_path]

set -e

DB="${1:-docs.db}"

echo "=== Embedding Setup ==="
echo "Database: $DB"
echo ""

# Install sentence-transformers if not present
if python3 -c "import sentence_transformers" 2>/dev/null; then
    echo "sentence-transformers: already installed"
else
    echo "Installing sentence-transformers..."
    pip install sentence-transformers
fi

echo ""

# Compute embeddings for all projects
python3 -c "
import sqlite3, sys
sys.path.insert(0, '.')
from ingest import init_db, compute_embeddings_for_project

conn = init_db('$DB')
projects = conn.execute('SELECT DISTINCT project FROM documents ORDER BY project').fetchall()

if not projects:
    print('No projects found in database.')
    sys.exit(0)

for row in projects:
    project = row['project']
    doc_count = conn.execute('SELECT COUNT(*) FROM documents WHERE project = ?', (project,)).fetchone()[0]
    page_count = conn.execute('''
        SELECT COUNT(*) FROM pages p JOIN documents d ON d.id = p.doc_id WHERE d.project = ?
    ''', (project,)).fetchone()[0]
    print(f'\n{\"=\" * 60}')
    print(f'Project: {project}')
    print(f'Documents: {doc_count}, Pages: {page_count}')
    print(f'{\"=\" * 60}')
    compute_embeddings_for_project(conn, project)

# Summary
total = conn.execute('SELECT COUNT(*) FROM page_embeddings').fetchone()[0]
print(f'\n{\"=\" * 60}')
print(f'Done. Total embeddings: {total}')
print(f'{\"=\" * 60}')
conn.close()
"
