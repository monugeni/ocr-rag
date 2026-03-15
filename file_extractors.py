"""
Extract text from non-PDF file types (images, DOCX, XLSX)
and handle archive extraction (ZIP, TAR, GZ).

Every extractor returns (pages, sections) in the same format as
extractor.extract_pdf() so the rest of the ingestion pipeline
(ingest_document, metadata, embeddings) works unchanged.
"""

import gzip
import subprocess
import tarfile
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
DOCX_EXTENSIONS = {'.docx'}
XLSX_EXTENSIONS = {'.xlsx', '.xls'}
PDF_EXTENSIONS = {'.pdf'}
INGESTABLE_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | DOCX_EXTENSIONS | XLSX_EXTENSIONS
ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tgz'}  # .tar.gz handled separately


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------

def is_archive(filename: str) -> bool:
    lower = filename.lower()
    if lower.endswith('.tar.gz'):
        return True
    return Path(lower).suffix in ARCHIVE_EXTENSIONS | {'.gz'}


def extract_archive(archive_path: str, dest_dir: Path) -> list[str]:
    """Extract supported files from an archive into dest_dir (flattened).

    Returns list of extracted filenames.
    """
    lower = str(archive_path).lower()
    extracted = []

    if lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name or name.startswith('.'):
                    continue
                if Path(name).suffix.lower() in INGESTABLE_EXTENSIONS:
                    data = zf.read(info.filename)
                    (dest_dir / name).write_bytes(data)
                    extracted.append(name)

    elif lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz'):
        mode = 'r:gz' if (lower.endswith('.gz') or lower.endswith('.tgz')) else 'r'
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name or name.startswith('.'):
                    continue
                if Path(name).suffix.lower() in INGESTABLE_EXTENSIONS:
                    f = tf.extractfile(member)
                    if f:
                        (dest_dir / name).write_bytes(f.read())
                        extracted.append(name)

    elif lower.endswith('.gz'):
        name = Path(archive_path).stem  # remove .gz
        if Path(name).suffix.lower() in INGESTABLE_EXTENSIONS:
            with gzip.open(archive_path, 'rb') as gz:
                (dest_dir / name).write_bytes(gz.read())
            extracted.append(name)

    return extracted


# ---------------------------------------------------------------------------
# Image extraction (tesseract OCR)
# ---------------------------------------------------------------------------

def extract_image(image_path: str) -> tuple[list[dict], list[dict]]:
    """OCR an image file via tesseract -> (pages, sections)."""
    try:
        result = subprocess.run(
            ['tesseract', str(image_path), 'stdout', '-l', 'eng'],
            capture_output=True, text=True, timeout=120,
        )
        text = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        text = ''

    if not text:
        return [], []

    pages = [{'page_num': 1, 'content': text, 'breadcrumb': ''}]
    return pages, []


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def _table_to_markdown(table) -> str:
    """Convert a python-docx Table to a markdown table."""
    rows = []
    for row in table.rows:
        cells = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
        rows.append(cells)

    if not rows:
        return ''

    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append('')

    lines = []
    lines.append('| ' + ' | '.join(rows[0]) + ' |')
    lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
    for r in rows[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def extract_docx(docx_path: str) -> tuple[list[dict], list[dict]]:
    """Extract text + tables from DOCX -> (pages, sections).

    Each heading-delimited section becomes a page.
    Tables are rendered as markdown in document order.
    """
    from docx import Document

    doc = Document(str(docx_path))

    # Map element identity -> Paragraph/Table for ordered iteration
    para_map = {id(p._element): p for p in doc.paragraphs}
    table_map = {id(t._element): t for t in doc.tables}

    pages: list[dict] = []
    sections: list[dict] = []
    current_parts: list[str] = []
    current_heading = ''

    def flush():
        nonlocal current_parts
        text = '\n'.join(current_parts).strip()
        if text:
            pages.append({
                'page_num': len(pages) + 1,
                'content': text,
                'breadcrumb': current_heading,
            })
        current_parts = []

    for child in doc.element.body:
        cid = id(child)

        if cid in para_map:
            para = para_map[cid]
            style = para.style.name if para.style else ''
            text = para.text.strip()

            if style.startswith('Heading'):
                flush()
                raw_level = style.replace('Heading', '').strip()
                level = int(raw_level) if raw_level.isdigit() else 1
                current_heading = text
                sections.append({
                    'heading': text, 'level': level,
                    'page_num': len(pages) + 1,
                    'breadcrumb': text,
                })
                if text:
                    current_parts.append(text)
            elif text:
                current_parts.append(text)

        elif cid in table_map:
            md = _table_to_markdown(table_map[cid])
            if md:
                current_parts.append('\n' + md)

    flush()

    # No headings? Chunk into ~3000-char pages
    if not sections and pages:
        all_text = '\n'.join(p['content'] for p in pages)
        pages = []
        for i in range(0, max(len(all_text), 1), 3000):
            chunk = all_text[i:i + 3000].strip()
            if chunk:
                pages.append({
                    'page_num': len(pages) + 1,
                    'content': chunk,
                    'breadcrumb': '',
                })

    return pages, sections


# ---------------------------------------------------------------------------
# XLSX extraction
# ---------------------------------------------------------------------------

def extract_xlsx(xlsx_path: str) -> tuple[list[dict], list[dict]]:
    """Extract sheets from XLSX -> (pages, sections).

    Each sheet becomes a page with cell data rendered as a markdown table.
    """
    from openpyxl import load_workbook

    wb = load_workbook(str(xlsx_path), data_only=True, read_only=True)
    pages: list[dict] = []
    sections: list[dict] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        page_num = len(pages) + 1

        sections.append({
            'heading': sheet_name, 'level': 1,
            'page_num': page_num, 'breadcrumb': sheet_name,
        })

        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() if c is not None else '' for c in row]
            if any(cells):
                rows.append(cells)

        if not rows:
            pages.append({
                'page_num': page_num,
                'content': f'{sheet_name}\n[Empty sheet]',
                'breadcrumb': sheet_name,
            })
            continue

        ncols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < ncols:
                r.append('')

        lines = [sheet_name, '']
        lines.append('| ' + ' | '.join(rows[0]) + ' |')
        lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
        for r in rows[1:]:
            lines.append('| ' + ' | '.join(r) + ' |')

        pages.append({
            'page_num': page_num,
            'content': '\n'.join(lines),
            'breadcrumb': sheet_name,
        })

    wb.close()
    return pages, sections


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_file(file_path: str) -> tuple[list[dict], list[dict]]:
    """Route to the right extractor based on file extension.

    PDFs go through the full extractor.extract_pdf() pipeline.
    Other types use the simpler extractors above.
    """
    ext = Path(file_path).suffix.lower()

    if ext in PDF_EXTENSIONS:
        from extractor import extract_pdf
        return extract_pdf(file_path)
    if ext in IMAGE_EXTENSIONS:
        return extract_image(file_path)
    if ext in DOCX_EXTENSIONS:
        return extract_docx(file_path)
    if ext in XLSX_EXTENSIONS:
        return extract_xlsx(file_path)

    print(f"  Unsupported file type: {ext}")
    return [], []
