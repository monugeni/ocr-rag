FROM python:3.12-slim

# Same system deps deploy.sh installs on the prod box
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils ghostscript qpdf \
    ocrmypdf tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# ponytail: app lives at /btrfs/ocr-rag inside the container too — the DB and
# docchecker store absolute paths, so mirroring the prod path avoids any migration.
WORKDIR /btrfs/ocr-rag
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# docchecker defaults its worker python to <app>/venv/bin/python; no venv here
ENV OCR_RAG_PYTHON=/usr/local/bin/python3

EXPOSE 8200 8201
CMD ["python3", "web.py", "--db", "/btrfs/ocr-rag/data/docs.db", \
     "--port", "8201", "--mcp-port", "8200", \
     "--uploads-dir", "/btrfs/ocr-rag/data/uploads"]
