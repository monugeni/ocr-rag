#!/usr/bin/env python3
"""
Heuristic PDF Structure Extractor
===================================
Extracts page text, headings, section hierarchy, and tables from PDFs
using only pdfplumber + heuristics — no ML models required.

For born-digital PDFs: uses embedded font metadata (size, bold, name).
For scanned PDFs: falls back to ocrmypdf/tesseract, then re-extracts.

Heading detection signals (scored & combined):
  1. Decimal numbering depth (1. → H1, 2.1. → H2, 2.1.1. → H3)
  2. ALL CAPS short lines
  3. Font size relative to body text mode
  4. Bold font flag
  5. Vertical gap before line (section break spacing)
  6. Short standalone line (not a continuation)
  7. Colon termination ("SCOPE OF WORK:")
  8. Indentation (flush-left vs indented)

Running headers/footers are auto-detected and stripped.

Output matches the (pages, sections) format used by ingest.py so it
plugs directly into the existing ingestion pipeline.

LLM feedback loop: corrections written to a sidecar JSON are merged
on subsequent extractions, continuously improving heading detection.

Usage:
    python extractor.py input.pdf
    python extractor.py input.pdf --output output.json
    python extractor.py /folder/of/pdfs/ --output-dir ./extracted/
"""

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pdfplumber


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedLine:
    """A single text line with metadata."""
    page_num: int
    text: str
    font_size: float = 0.0
    is_bold: bool = False
    is_upper: bool = False
    word_count: int = 0
    top: float = 0.0
    bottom: float = 0.0
    x0: float = 0.0
    gap_above: float = 0.0  # vertical gap from previous line
    fonts: set = field(default_factory=set)


@dataclass
class Heading:
    """A detected heading."""
    text: str
    level: int
    page_num: int
    confidence: float = 0.0
    signals: list = field(default_factory=list)
    numbering: str = ""  # e.g. "2.1.3"


# ---------------------------------------------------------------------------
# Running header/footer detection
# ---------------------------------------------------------------------------

def detect_running_headers(page_lines: dict[int, list[ExtractedLine]],
                           min_pages: int = 3) -> set[str]:
    """Find text that repeats in the same position across multiple pages.

    Returns set of normalized text strings to exclude from heading detection.
    """
    # Collect first 6 and last 3 lines per page
    header_candidates = Counter()
    footer_candidates = Counter()

    for page_num in sorted(page_lines.keys()):
        lines = page_lines[page_num]
        if not lines:
            continue
        for line in lines[:6]:
            normalized = _normalize_for_comparison(line.text)
            if normalized and len(normalized) > 3:
                header_candidates[normalized] += 1
        for line in lines[-3:]:
            normalized = _normalize_for_comparison(line.text)
            if normalized and len(normalized) > 3:
                footer_candidates[normalized] += 1

    total_pages = len(page_lines)
    threshold = max(min_pages, total_pages * 0.4)

    running = set()
    for text, count in header_candidates.items():
        if count >= threshold:
            running.add(text)
    for text, count in footer_candidates.items():
        if count >= threshold:
            running.add(text)

    return running


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for running header comparison (replace digits, collapse space)."""
    text = re.sub(r'\d+', '#', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


# ---------------------------------------------------------------------------
# Page number detection
# ---------------------------------------------------------------------------

PAGE_NUM_RE = re.compile(
    r'(?:page|sheet|pg\.?|p\.?)?\s*(\d{1,4})\s*(?:of|/)\s*(\d{1,4})',
    re.IGNORECASE
)
SIMPLE_PAGE_RE = re.compile(r'^\s*(\d{1,4})\s*$')


def detect_page_number(lines: list[ExtractedLine]) -> Optional[int]:
    """Try to find a page number from the last few lines of a page."""
    for line in reversed(lines[-3:]):
        m = PAGE_NUM_RE.search(line.text)
        if m:
            return int(m.group(1))
        m = SIMPLE_PAGE_RE.match(line.text.strip())
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Numbering pattern detection
# ---------------------------------------------------------------------------

DECIMAL_NUM_RE = re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)')
ALPHA_NUM_RE = re.compile(r'^([a-z])\.\s+(.+)', re.IGNORECASE)
ROMAN_NUM_RE = re.compile(r'^(i{1,3}|iv|vi{0,3}|ix|xi{0,3}|xiv|xv)\.\s+(.+)',
                          re.IGNORECASE)


def parse_decimal_numbering(text: str) -> tuple[str, int, str]:
    """Parse decimal numbering like '2.1.3. Heading text'.

    Returns (number_str, depth, remaining_text) or ('', 0, text).
    """
    m = DECIMAL_NUM_RE.match(text.strip())
    if m:
        num_str = m.group(1)
        depth = len(num_str.split('.'))
        return num_str, depth, m.group(2).strip()
    return '', 0, text


# ---------------------------------------------------------------------------
# Font size clustering (like marker's KMeans but simpler)
# ---------------------------------------------------------------------------

def cluster_font_sizes(all_sizes: list[float], max_levels: int = 4) -> list[float]:
    """Cluster font sizes into heading levels.

    Returns sorted list of size thresholds (largest first).
    Sizes above threshold[0] = H1, above threshold[1] = H2, etc.
    """
    if not all_sizes:
        return []

    # Find the mode (most common size = body text)
    size_counts = Counter(round(s, 1) for s in all_sizes)
    body_size = size_counts.most_common(1)[0][0]

    # Sizes larger than body text are heading candidates
    unique_larger = sorted(set(s for s in size_counts if s > body_size), reverse=True)

    if not unique_larger:
        return []

    # Take up to max_levels distinct sizes
    return unique_larger[:max_levels]


# ---------------------------------------------------------------------------
# Core line extraction from pdfplumber
# ---------------------------------------------------------------------------

def extract_lines_from_page(page, page_num: int) -> list[ExtractedLine]:
    """Extract text lines with font metadata from a pdfplumber page."""
    words = page.extract_words(
        extra_attrs=['fontname', 'size', 'top', 'bottom'],
        keep_blank_chars=False
    )

    if not words:
        return []

    # Group words into lines by vertical position
    lines_by_top = defaultdict(list)
    for w in words:
        line_key = round(w['top'], 1)
        lines_by_top[line_key].append(w)

    result = []
    prev_bottom = 0.0

    for top in sorted(lines_by_top.keys()):
        ws = lines_by_top[top]
        text = ' '.join(w['text'] for w in ws)

        if not text.strip():
            continue

        sizes = [w['size'] for w in ws if w['size'] > 0]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        fonts = set(w['fontname'] for w in ws)
        is_bold = any('Bold' in f or 'bold' in f or 'BOLD' in f for f in fonts)
        is_upper = text == text.upper() and len(text.strip()) > 3 and any(c.isalpha() for c in text)

        gap = top - prev_bottom if prev_bottom > 0 else 0.0

        result.append(ExtractedLine(
            page_num=page_num,
            text=text,
            font_size=round(avg_size, 1),
            is_bold=is_bold,
            is_upper=is_upper,
            word_count=len(ws),
            top=top,
            bottom=max(w['bottom'] for w in ws),
            x0=min(w['x0'] for w in ws),
            gap_above=gap,
            fonts=fonts,
        ))
        prev_bottom = result[-1].bottom

    return result


# ---------------------------------------------------------------------------
# Born-digital detection
# ---------------------------------------------------------------------------

def is_born_digital(pdf_path: str, sample_pages: int = 3) -> bool:
    """Check if a PDF has embedded text (born-digital) or is scanned."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            chars_found = 0
            pages_checked = 0
            for page in pdf.pages[:sample_pages]:
                text = page.extract_text() or ''
                chars_found += len(text.strip())
                pages_checked += 1

            # If we found reasonable text, it's born-digital
            return chars_found > (50 * pages_checked)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OCR fallback for scanned PDFs
# ---------------------------------------------------------------------------

def ocr_pdf(pdf_path: str) -> str:
    """Run ocrmypdf to add text layer, return path to OCR'd PDF.

    Falls back to original if ocrmypdf is not available.
    """
    try:
        out_path = tempfile.mktemp(suffix='.pdf')
        # --force-ocr: OCR every page even if text exists (scanned PDFs
        # sometimes have garbage invisible text layers)
        result = subprocess.run(
            ['ocrmypdf', '--force-ocr', '-l', 'eng',
             pdf_path, out_path],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            return out_path
        print(f"  ocrmypdf attempt 1 failed: {result.stderr.strip()[:200]}")
        # Fallback: just add OCR where missing
        result = subprocess.run(
            ['ocrmypdf', '--skip-text', '-l', 'eng', pdf_path, out_path],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print(f"  WARNING: ocrmypdf not available, using original PDF")
    return pdf_path


# ---------------------------------------------------------------------------
# Heading scoring
# ---------------------------------------------------------------------------

def score_heading(line: ExtractedLine, body_size: float,
                  heading_sizes: list[float], median_gap: float,
                  left_margin: float, running_headers: set[str]) -> Optional[Heading]:
    """Score a line for heading-ness using multiple signals.

    Requires at least one PRIMARY signal (numbering, bold, caps, bigger font)
    to be present. Secondary signals (gap, short, colon, flush) only boost.

    Returns a Heading if confidence is above threshold, else None.
    """
    # Skip running headers
    normalized = _normalize_for_comparison(line.text)
    if normalized in running_headers:
        return None

    # Skip very short lines (OCR noise, single chars)
    if len(line.text.strip()) < 4:
        return None

    # Skip very long lines (body text)
    if line.word_count > 12:
        return None

    # Skip page numbers
    if SIMPLE_PAGE_RE.match(line.text.strip()):
        return None
    if PAGE_NUM_RE.search(line.text):
        return None

    signals = []
    confidence = 0.0
    level = None
    has_primary = False  # Must have at least one primary signal

    # --- Signal 1: Decimal numbering (strongest, PRIMARY) ---
    num_str, num_depth, remaining = parse_decimal_numbering(line.text)
    if num_depth > 0:
        remaining_words = len(remaining.split()) if remaining else 0
        # Must have some text after the number, and be short
        if remaining and remaining_words <= 10:
            confidence += 0.5
            level = min(num_depth, 4)
            signals.append(f'numbered_d{num_depth}')
            has_primary = True
        elif not remaining:
            # Bare number like "3.3." with no text — skip
            return None

    # --- Signal 2: ALL CAPS (PRIMARY) ---
    if line.is_upper and line.word_count <= 10:
        confidence += 0.25
        signals.append('upper')
        has_primary = True

    # --- Signal 3: Font size larger than body (PRIMARY) ---
    if body_size > 0 and line.font_size > body_size * 1.1:
        size_ratio = line.font_size / body_size
        bonus = min(0.35, (size_ratio - 1.0) * 0.6)
        confidence += bonus
        signals.append(f'bigger_font({line.font_size:.1f}>{body_size:.1f})')
        has_primary = True

        # Font size can determine level if no numbering
        if level is None and heading_sizes:
            for i, threshold in enumerate(heading_sizes):
                if line.font_size >= threshold * 0.95:
                    level = i + 1
                    break

    # --- Signal 4: Bold (PRIMARY) ---
    if line.is_bold:
        confidence += 0.2
        signals.append('bold')
        has_primary = True

    # --- Secondary signals (only boost, never enough alone) ---

    # --- Signal 5: Vertical gap above ---
    if median_gap > 0 and line.gap_above > median_gap * 2.0:
        gap_ratio = line.gap_above / median_gap
        bonus = min(0.15, (gap_ratio - 1.5) * 0.08)
        confidence += bonus
        signals.append(f'gap({line.gap_above:.1f})')

    # --- Signal 6: Short line ---
    if line.word_count <= 6:
        confidence += 0.1
        signals.append('short')

    # --- Signal 7: Colon termination ---
    if line.text.rstrip().endswith(':'):
        confidence += 0.1
        signals.append('colon')

    # --- Signal 8: Flush left (at margin) ---
    if left_margin > 0 and abs(line.x0 - left_margin) < 5:
        confidence += 0.05
        signals.append('flush_left')

    # MUST have at least one primary signal
    if not has_primary:
        return None

    # Minimum confidence threshold
    if confidence < 0.3:
        return None

    # Default level assignment if no numbering and no font size gave us one
    if level is None:
        if line.is_upper and confidence >= 0.5:
            level = 1
        elif line.is_bold and confidence >= 0.4:
            level = 2
        else:
            level = 3

    return Heading(
        text=line.text.strip(),
        level=level,
        page_num=line.page_num,
        confidence=confidence,
        signals=signals,
        numbering=num_str,
    )


# ---------------------------------------------------------------------------
# LLM feedback integration
# ---------------------------------------------------------------------------

CORRECTIONS_SUFFIX = '_corrections.json'


def load_corrections(pdf_path: str) -> dict:
    """Load LLM corrections sidecar file if it exists.

    Version 1 fields (heading corrections):
        heading_overrides, new_headings, removed_headings

    Version 2 fields (full correction suite):
        + document_splits, document_merges, document_titles, document_types,
          skipped_pages, page_reclassifications, breadcrumb_overrides,
          running_headers_add, running_headers_remove, ocr_fixes,
          metadata_overrides, cross_references, quality_flags
    """
    corrections_path = Path(pdf_path).with_suffix('') / CORRECTIONS_SUFFIX
    if not corrections_path.exists():
        # Also check next to the PDF
        corrections_path = Path(str(pdf_path) + CORRECTIONS_SUFFIX)
    if not corrections_path.exists():
        corrections_path = Path(pdf_path).parent / (Path(pdf_path).stem + CORRECTIONS_SUFFIX)

    if corrections_path.exists():
        try:
            with open(corrections_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def apply_corrections(headings: list[Heading], corrections: dict) -> list[Heading]:
    """Apply LLM corrections to detected headings.

    Handles heading_overrides, new_headings, and removed_headings from the
    sidecar JSON. Other correction types (OCR fixes, running headers, etc.)
    are applied at different pipeline stages.
    """
    if not corrections:
        return headings

    overrides = corrections.get('heading_overrides', {})
    removed_prefixes = {
        (r['page_num'], r['text_prefix'])
        for r in corrections.get('removed_headings', [])
    }

    result = []
    for h in headings:
        # Check if removed
        is_removed = False
        for page_num, prefix in removed_prefixes:
            if h.page_num == page_num and h.text.startswith(prefix):
                is_removed = True
                break
        if is_removed:
            continue

        # Check for overrides
        key = f"{h.page_num}:{h.text[:50]}"
        if key in overrides:
            override = overrides[key]
            if not override.get('is_heading', True):
                continue
            if 'level' in override:
                h.level = override['level']
            if 'corrected_text' in override:
                h.text = override['corrected_text']
            h.signals.append('llm_corrected')

        result.append(h)

    # Add new headings from LLM
    for new_h in corrections.get('new_headings', []):
        result.append(Heading(
            text=new_h['text'],
            level=new_h['level'],
            page_num=new_h['page_num'],
            confidence=0.9,
            signals=['llm_added'],
        ))

    # Re-sort by page number
    result.sort(key=lambda h: (h.page_num, h.confidence), reverse=False)
    return result


def apply_running_header_corrections(running_headers: set[str],
                                     corrections: dict) -> set[str]:
    """Apply running header add/remove corrections.

    running_headers_add: additional patterns to strip
    running_headers_remove: patterns that should NOT be stripped
    """
    if not corrections:
        return running_headers

    for text in corrections.get('running_headers_add', []):
        running_headers.add(_normalize_for_comparison(text))

    for text in corrections.get('running_headers_remove', []):
        running_headers.discard(_normalize_for_comparison(text))

    return running_headers


def apply_ocr_fixes(content: str, page_num: int,
                    corrections: dict) -> str:
    """Apply OCR text fixes from sidecar to page content."""
    if not corrections:
        return content
    for fix in corrections.get('ocr_fixes', []):
        if fix['page_num'] == page_num:
            content = content.replace(fix['old_text'], fix['new_text'])
    return content


def get_skipped_pages(corrections: dict) -> set[int]:
    """Get set of page numbers to skip from corrections."""
    if not corrections:
        return set()
    return set(corrections.get('skipped_pages', []))


def save_corrections(pdf_path: str, corrections: dict):
    """Save corrections sidecar file."""
    corrections_path = Path(pdf_path).parent / (Path(pdf_path).stem + CORRECTIONS_SUFFIX)
    with open(corrections_path, 'w') as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Breadcrumb building
# ---------------------------------------------------------------------------

def build_breadcrumbs(headings: list[Heading]) -> dict[int, str]:
    """Build a heading stack and assign breadcrumbs per page.

    Returns {page_num: breadcrumb_string}
    """
    breadcrumbs = {}
    stack = []  # [(level, text)]

    # Sort headings by page number
    sorted_headings = sorted(headings, key=lambda h: h.page_num)

    # Build a map of page -> list of headings on that page
    page_headings = defaultdict(list)
    for h in sorted_headings:
        page_headings[h.page_num].append(h)

    # Track current breadcrumb across all pages
    current_bc = ''

    # We need to know all pages to assign breadcrumbs
    if not sorted_headings:
        return breadcrumbs

    min_page = sorted_headings[0].page_num
    max_page = sorted_headings[-1].page_num

    for page_num in range(min_page, max_page + 1):
        if page_num in page_headings:
            for h in page_headings[page_num]:
                # Pop stack to same or higher level
                while stack and stack[-1][0] >= h.level:
                    stack.pop()
                stack.append((h.level, h.text))

        current_bc = ' > '.join(text for _, text in stack)
        breadcrumbs[page_num] = current_bc

    return breadcrumbs


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def extract_tables_from_page(page) -> list[str]:
    """Extract tables from a pdfplumber page as markdown."""
    tables = page.extract_tables()
    result = []

    for table in tables:
        if not table or len(table) < 2:
            continue
        # Filter false positives
        ncols = max(len(r) for r in table)
        if ncols < 2:
            continue

        clean = []
        for row in table:
            clean.append([
                str(c).replace('\n', ' ').strip() if c else ''
                for c in row
            ])

        # Normalize column count
        for r in clean:
            while len(r) < ncols:
                r.append('')

        lines = []
        lines.append('| ' + ' | '.join(clean[0]) + ' |')
        lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
        for r in clean[1:]:
            lines.append('| ' + ' | '.join(r) + ' |')

        result.append('\n'.join(lines))

    return result


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline stages (composable)
# ---------------------------------------------------------------------------

def stage_ocr(pdf_path: str) -> str:
    """Stage 1: Ensure PDF has text layer. OCR if scanned.

    Returns path to a PDF with extractable text (may be the original
    if born-digital, or a temp OCR'd copy if scanned).
    """
    if is_born_digital(pdf_path):
        return pdf_path
    print(f"  Scanned PDF detected, running OCR...")
    return ocr_pdf(pdf_path)


def stage_extract_text(pdf_path: str, corrections: dict = None
                       ) -> tuple[dict[int, list[ExtractedLine]], dict]:
    """Stage 2: Extract text lines + font metadata from every page.

    Returns (page_lines, stats) where:
      page_lines = {page_num: [ExtractedLine, ...]}
      stats = {'body_size', 'heading_sizes', 'median_gap', 'left_margin'}
    """
    all_page_lines: dict[int, list[ExtractedLine]] = {}
    all_sizes = []
    all_gaps = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            lines = extract_lines_from_page(page, page_num)
            all_page_lines[page_num] = lines

            for line in lines:
                if line.font_size > 0:
                    all_sizes.append(line.font_size)
                if line.gap_above > 0:
                    all_gaps.append(line.gap_above)

    # Compute document-wide statistics
    size_counts = Counter(round(s, 1) for s in all_sizes)
    body_size = size_counts.most_common(1)[0][0] if size_counts else 0.0
    heading_sizes = cluster_font_sizes(all_sizes)
    median_gap = sorted(all_gaps)[len(all_gaps) // 2] if all_gaps else 0.0

    all_x0 = []
    for lines in all_page_lines.values():
        for line in lines:
            all_x0.append(round(line.x0, 0))
    x0_counts = Counter(all_x0)
    left_margin = x0_counts.most_common(1)[0][0] if x0_counts else 0.0

    running_headers = detect_running_headers(all_page_lines)

    # Apply running header corrections from sidecar
    if corrections:
        running_headers = apply_running_header_corrections(
            running_headers, corrections)

    stats = {
        'body_size': body_size,
        'heading_sizes': heading_sizes,
        'median_gap': median_gap,
        'left_margin': left_margin,
        'running_headers': running_headers,
    }

    print(f"  Body font: {body_size}pt | heading sizes: {heading_sizes}")
    print(f"  Median gap: {median_gap:.1f} | left margin: {left_margin:.0f} | running headers: {len(running_headers)}")

    return all_page_lines, stats


def stage_detect_headings(page_lines: dict[int, list[ExtractedLine]],
                          stats: dict,
                          pdf_path: str = None,
                          apply_llm_corrections: bool = True
                          ) -> list[Heading]:
    """Stage 3: Detect headings using heuristic scoring.

    Optionally applies LLM corrections from sidecar file.
    """
    headings = []
    for page_num in sorted(page_lines.keys()):
        for line in page_lines[page_num]:
            heading = score_heading(
                line,
                stats['body_size'],
                stats['heading_sizes'],
                stats['median_gap'],
                stats['left_margin'],
                stats['running_headers'],
            )
            if heading:
                headings.append(heading)

    if apply_llm_corrections and pdf_path:
        corrections = load_corrections(pdf_path)
        if corrections:
            before = len(headings)
            headings = apply_corrections(headings, corrections)
            print(f"  LLM corrections applied: {before} -> {len(headings)} headings")

    print(f"  Headings detected: {len(headings)}")
    return headings


def stage_build_pages(pdf_path: str,
                      page_lines: dict[int, list[ExtractedLine]],
                      headings: list[Heading],
                      stats: dict,
                      corrections: dict = None
                      ) -> tuple[list[dict], list[dict]]:
    """Stage 4: Build pages and sections output with breadcrumbs + tables.

    Returns (pages, sections) matching ingest.py's expected format.
    """
    running_headers = stats['running_headers']
    breadcrumbs = build_breadcrumbs(headings)
    skipped = get_skipped_pages(corrections) if corrections else set()

    # Build sections list
    sections = []
    heading_stack = []
    for h in sorted(headings, key=lambda x: x.page_num):
        while heading_stack and heading_stack[-1][0] >= h.level:
            heading_stack.pop()
        heading_stack.append((h.level, h.text))
        bc = ' > '.join(text for _, text in heading_stack)
        sections.append({
            'heading': h.text,
            'level': h.level,
            'page_num': h.page_num,
            'breadcrumb': bc,
        })

    # Build pages with content
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # Skip pages marked in corrections
            if page_num in skipped:
                continue

            lines = page_lines.get(page_num, [])

            content_parts = []
            for line in lines:
                normalized = _normalize_for_comparison(line.text)
                if normalized in running_headers:
                    continue
                if SIMPLE_PAGE_RE.match(line.text.strip()):
                    continue
                if PAGE_NUM_RE.search(line.text) and line.word_count <= 5:
                    continue
                content_parts.append(line.text)

            # Add tables
            tables = extract_tables_from_page(page)
            for table_md in tables:
                content_parts.append('\n' + table_md)

            content = '\n'.join(content_parts).strip()

            # Apply OCR fixes from corrections
            if corrections:
                content = apply_ocr_fixes(content, page_num, corrections)

            if content:
                pages.append({
                    'page_num': page_num,
                    'content': content,
                    'breadcrumb': breadcrumbs.get(page_num, ''),
                })

    return pages, sections


def stage_detect_subdocuments(pages: list[dict], sections: list[dict],
                              min_doc_pages: int = 4,
                              corrections: dict = None
                              ) -> list[tuple[list[dict], list[dict], str]]:
    """Stage 5: Detect and split sub-documents if the PDF contains multiple.

    Also incorporates document_splits from sidecar corrections.

    Returns list of (sub_pages, sub_sections, title_hint).
    Single-document PDFs return a list with one entry.
    """
    boundaries = detect_document_boundaries(pages, sections, min_doc_pages)

    # Add splits from corrections sidecar
    if corrections:
        for split in corrections.get('document_splits', []):
            at_page = split['at_page']
            if at_page not in boundaries:
                boundaries.append(at_page)
        boundaries = sorted(set(boundaries))

    if boundaries:
        subdocs = split_into_subdocuments(pages, sections, boundaries)
        print(f"  Sub-documents detected: {len(subdocs)}")
        for i, (sp, ss, title) in enumerate(subdocs):
            page_range = f"p{sp[0]['page_num']}-{sp[-1]['page_num']}"
            print(f"    [{i+1}] {title[:60]} ({page_range}, {len(sp)} pages)")
        return subdocs
    else:
        first_heading = sections[0]['heading'] if sections else ''
        return [(pages, sections, first_heading)]


# ---------------------------------------------------------------------------
# Full pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str, apply_llm_corrections: bool = True
                ) -> tuple[list[dict], list[dict]]:
    """Run the full extraction pipeline on a single PDF.

    Stages:
      1. OCR if scanned
      2. Extract text + font metadata via pdfplumber (+ running header corrections)
      3. Detect headings via heuristic scoring (+ LLM heading corrections)
      4. Build pages with content, breadcrumbs, tables (+ OCR fixes, skipped pages)
      5. Detect sub-document boundaries (+ sidecar splits)

    Returns (pages, sections) compatible with ingest.py.
    """
    pdf_path = str(pdf_path)

    # Load corrections once, pass through pipeline
    corrections = load_corrections(pdf_path) if apply_llm_corrections else {}

    # Stage 1: OCR
    actual_path = stage_ocr(pdf_path)

    # Stage 2: Extract text (applies running header corrections)
    page_lines, stats = stage_extract_text(actual_path, corrections)
    if not page_lines:
        return [], []

    # Stage 3: Detect headings (applies heading corrections)
    headings = stage_detect_headings(
        page_lines, stats, pdf_path, apply_llm_corrections
    )

    # Stage 4: Build pages + sections (applies OCR fixes, skips pages)
    pages, sections = stage_build_pages(
        actual_path, page_lines, headings, stats, corrections)

    # Clean up OCR temp file
    if actual_path != pdf_path:
        try:
            Path(actual_path).unlink()
        except OSError:
            pass

    return pages, sections


# ---------------------------------------------------------------------------
# Document boundary detection (for split PDFs containing multiple documents)
# ---------------------------------------------------------------------------

def detect_document_boundaries(pages: list[dict],
                               sections: list[dict],
                               min_doc_pages: int = 4) -> list[int]:
    """Detect document boundaries within a single PDF.

    Only flags boundaries where a top-level heading appears AND creates
    sub-documents of at least min_doc_pages pages. This avoids splitting
    a 6-page doc into 6 single-page "documents".

    Returns list of page numbers where new documents begin.
    """
    if not pages or not sections or len(pages) < min_doc_pages * 2:
        return []

    first_page = pages[0]['page_num']
    last_page = pages[-1]['page_num']
    min_level = min(s['level'] for s in sections) if sections else 1
    candidates = []

    for s in sections:
        if s['level'] != min_level or s['page_num'] <= first_page:
            continue
        # Numbered sections (1., 2., etc.) are parts of the same document,
        # NOT document boundaries. Only non-numbered top-level headings
        # (like a new title page or annexure start) signal a new document.
        heading_text = s['heading'].strip()
        if DECIMAL_NUM_RE.match(heading_text):
            continue
        candidates.append(s['page_num'])

    if not candidates:
        return []

    # Filter: each sub-document must have at least min_doc_pages pages
    boundaries = []
    prev_start = first_page
    for boundary in sorted(set(candidates)):
        if boundary - prev_start >= min_doc_pages:
            boundaries.append(boundary)
            prev_start = boundary

    # Also check the last segment has enough pages
    if boundaries and (last_page - boundaries[-1] + 1) < min_doc_pages:
        boundaries.pop()

    return boundaries


def split_into_subdocuments(
    pages: list[dict], sections: list[dict], boundaries: list[int]
) -> list[tuple[list[dict], list[dict], str]]:
    """Split pages and sections into sub-documents at boundary pages.

    Returns list of (sub_pages, sub_sections, title_hint) tuples.
    """
    if not boundaries:
        first_heading = sections[0]['heading'] if sections else ''
        return [(pages, sections, first_heading)]

    breaks = [pages[0]['page_num']] + boundaries + [pages[-1]['page_num'] + 1]

    subdocs = []
    for i in range(len(breaks) - 1):
        start, end = breaks[i], breaks[i + 1]

        sub_pages = [p for p in pages if start <= p['page_num'] < end]
        sub_sections = [s for s in sections if start <= s['page_num'] < end]

        if not sub_pages:
            continue

        # Rebuild breadcrumbs relative to this sub-document
        if sub_sections:
            top_heading = sub_sections[0]['heading']
            for p in sub_pages:
                bc = p.get('breadcrumb', '')
                if top_heading in bc:
                    idx = bc.index(top_heading)
                    p['breadcrumb'] = bc[idx:]
        else:
            top_heading = ''

        subdocs.append((sub_pages, sub_sections, top_heading))

    return subdocs


# ---------------------------------------------------------------------------
# JSON output for LLM feedback
# ---------------------------------------------------------------------------

def extraction_to_json(pages: list[dict], sections: list[dict],
                       pdf_path: str) -> dict:
    """Package extraction results as JSON for review/feedback."""
    return {
        'source': str(pdf_path),
        'total_pages': len(pages),
        'total_sections': len(sections),
        'sections': sections,
        'pages': [{
            'page_num': p['page_num'],
            'breadcrumb': p['breadcrumb'],
            'content_preview': p['content'][:200] + '...' if len(p['content']) > 200 else p['content'],
            'content_length': len(p['content']),
        } for p in pages],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def process_single(pdf_path: str, output_path: str = None, verbose: bool = False):
    """Process a single PDF file."""
    pdf_path = str(Path(pdf_path).resolve())
    print(f"\nExtracting: {Path(pdf_path).name}")

    pages, sections = extract_pdf(pdf_path)

    if not pages:
        print(f"  No content extracted")
        return

    # Classify pages
    types = Counter()
    for p in pages:
        n = len(p['content'].strip())
        if n < 60:
            types['drawing'] += 1
        elif p['page_num'] <= 2 and n < 500:
            types['cover'] += 1
        else:
            types['text'] += 1

    print(f"  Pages: {len(pages)} ({', '.join(f'{v} {k}' for k, v in sorted(types.items()))})")
    print(f"  Sections: {len(sections)}")

    if verbose:
        print(f"\n  Section hierarchy:")
        for s in sections:
            indent = '    ' * s['level']
            print(f"    {indent}{s['heading']} (p{s['page_num']})")

    # Check for sub-documents
    boundaries = detect_document_boundaries(pages, sections)
    if boundaries:
        subdocs = split_into_subdocuments(pages, sections, boundaries)
        print(f"  Sub-documents detected: {len(subdocs)}")
        for i, (sp, ss, title) in enumerate(subdocs):
            page_range = f"p{sp[0]['page_num']}-{sp[-1]['page_num']}"
            print(f"    [{i+1}] {title[:60]} ({page_range}, {len(sp)} pages, {len(ss)} sections)")

    # Output
    if output_path:
        result = extraction_to_json(pages, sections, pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Output: {output_path}")

    return pages, sections


def process_folder(folder_path: str, output_dir: str = None, verbose: bool = False):
    """Process all PDFs in a folder."""
    folder = Path(folder_path)
    pdfs = sorted(folder.glob('*.pdf'))
    print(f"\nFound {len(pdfs)} PDFs in {folder}")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for pdf in pdfs:
        out_path = None
        if output_dir:
            out_path = str(Path(output_dir) / (pdf.stem + '.json'))

        result = process_single(str(pdf), out_path, verbose)
        if result:
            results[pdf.name] = result

    print(f"\nProcessed {len(results)}/{len(pdfs)} PDFs successfully")
    return results


def main():
    import argparse

    p = argparse.ArgumentParser(
        description='Heuristic PDF structure extractor (no ML required)'
    )
    p.add_argument('input', help='PDF file or folder of PDFs')
    p.add_argument('--output', '-o', help='Output JSON file (single PDF mode)')
    p.add_argument('--output-dir', help='Output directory (folder mode)')
    p.add_argument('--verbose', '-v', action='store_true', help='Show section hierarchy')

    args = p.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        process_folder(str(input_path), args.output_dir, args.verbose)
    elif input_path.is_file():
        process_single(str(input_path), args.output, args.verbose)
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
