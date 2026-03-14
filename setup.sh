#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== OCR-RAG Setup ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi
echo "Python: $(python3 --version)"

# Check Marker
if command -v marker_single &>/dev/null; then
    echo "Marker: $(which marker_single)"
else
    echo "WARNING: marker_single not found."
    echo "  Install with: pip install marker-pdf"
    echo "  Marker is needed to convert PDFs before ingestion."
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check for Anthropic API key
echo ""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY: set"
else
    echo "WARNING: ANTHROPIC_API_KEY not set."
    echo "  Metadata extraction will be skipped during ingestion."
    echo "  Set it with: export ANTHROPIC_API_KEY=sk-..."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  # Ingest a project folder (PDFs must be processed with Marker first)"
echo "  python ingest.py --project /path/to/documents/ --db docs.db"
echo ""
echo "  # Start the MCP server"
echo "  python mcp_server.py --db docs.db"
