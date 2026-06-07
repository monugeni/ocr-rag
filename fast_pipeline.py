#!/usr/bin/env python3
"""
Fast CPU-first PDF structure pipeline.

This is intentionally separate from extractor.py / ingest.py.  It is built for
large engineering PDF packs on small CPU machines:

  1. Split first when requested, using the existing heuristic splitter.
  2. Extract quickly with Poppler (pdftohtml -xml + pdftotext --layout).
  3. Reconstruct document-local headings and breadcrumbs from layout signals.
  4. Emit paragraph/clause chunks with local breadcrumbs and diagnostics.

The output is diagnostic-first: headings.json, chunks.jsonl, pages.json,
sections.json, preview.md, and report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Regexes and constants
# ---------------------------------------------------------------------------

PAGE_NUMBER_RE = re.compile(
    r"^(?:[-\s]*(?:page|pg\.?|sheet)?\s*)?"
    r"(?:\d{1,5}|[ivxlcdm]{1,12})"
    r"(?:\s*(?:of|/)\s*(?:\d{1,5}|[ivxlcdm]{1,12}))?"
    r"[-\s]*$",
    re.IGNORECASE,
)
DECIMAL_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,8})([.)])?\s+(.+)$")
COMPACT_DECIMAL_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){1,8})[.)]?\s*([A-Z][A-Z0-9 /&(),-]{2,})$")
SECTION_RE = re.compile(
    r"^(section|part|chapter|volume|book)\s+(\d+(?:\.\d+)*|[IVXLCM]+|[A-Z])"
    r"(?:\s*[-:.)]\s*|\s+)(.+)?$",
    re.IGNORECASE,
)
APPENDIX_RE = re.compile(
    r"^(appendix|annex(?:ure)?|attachment|enclosure|exhibit|schedule)\s*"
    r"([A-Za-z0-9IVXLCM.-]+)?(?:\s*[-:.)]\s*|\s+)?(.+)?$",
    re.IGNORECASE,
)
ALPHA_RE = re.compile(r"^([A-Z])(?:[.)])\s+(.+)$")
ALPHA_DECIMAL_RE = re.compile(r"^([A-Z])\s*\.\s*(\d+(?:\.\d+)*)\s+(.+)$")
ROMAN_RE = re.compile(r"^([IVX]{1,8}|iv|ix|v?i{1,3}|x{1,3})(?:[.)])\s+(.+)$", re.IGNORECASE)
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOC_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d{1,5}\s*$")
TOC_LEADER_RE = re.compile(r"(?:\.{3,}|[-–—_]{5,})\s*\d{1,5}\s*$")
TOC_ROW_RE = re.compile(
    r"^\s*(?P<num>(?:(?:section|part|chapter|appendix|annex(?:ure)?|attachment|schedule)\s+"
    r"[A-Za-z0-9IVXLCM.-]+|\d+(?:\.\d+)*|[IVXLCM]+|[A-Z])?)"
    r"\s*(?P<title>.*?)"
    r"(?:(?:\.{3,}|[-–—_]{5,})|\s{3,})\s*(?P<page>\d{1,5})\s*$",
    re.IGNORECASE,
)
DOC_TITLE_HINT_RE = re.compile(
    r"\b(advisory|specification|procedure|standard|code|manual|datasheet|data sheet|"
    r"calculation|philosophy|scope of work|technical requirement|"
    r"inspection and test plan|method statement|drawing list)\b",
    re.IGNORECASE,
)
STRONG_DOC_TITLE_HINT_RE = re.compile(
    r"\b(advisory|specification|procedure|standard|code|manual|datasheet|data sheet|"
    r"calculation|philosophy|scope of work|technical requirement|"
    r"inspection and test plan|method statement|drawing list)\b",
    re.IGNORECASE,
)
CONTEXT_TITLE_HINT_RE = re.compile(
    r"\b(piping|pipeline|pipe|valve|control valve|solenoid valve|damper|"
    r"analy[sz]er|instrument|electrical|civil|road|structural|mechanical|"
    r"welding|insulation|painting|refractory|scaffolding|fire|safety|"
    r"equipment|boiler|turnaround|maintenance|inspection|vendor|approved makes)\b",
    re.IGNORECASE,
)
TABLE_WORDS = {
    "description", "qty", "quantity", "uom", "unit", "rate", "amount",
    "remarks", "revision", "date", "prepared", "checked", "approved",
    "item", "tag", "line", "size", "schedule", "material",
    "activity", "activities", "ppe", "used", "use",
}
VALUE_WORDS = {
    "yes", "no", "na", "n/a", "applicable", "not applicable", "limited",
    "domestic", "international", "inr", "usd", "true", "false",
}
HEADING_KEYWORDS = {
    "scope", "objective", "objectives", "definition", "definitions",
    "responsibility", "responsibilities", "requirement", "requirements",
    "description", "introduction", "general", "record", "records",
    "revision", "revisions", "activity", "activities", "procedure",
    "procedures", "method", "methodology", "reference", "references",
    "abbreviation", "abbreviations", "term", "terms", "condition",
    "conditions", "quality", "inspection", "testing", "list", "lists",
    "unit", "units", "code", "codes", "standard", "standards",
    "purpose", "applicable", "base", "metal", "consumable",
    "consumables", "shielding", "purging", "process", "procedure",
    "qualification", "edge", "preparation", "alignment", "alingnment",
    "spacing", "weather", "technique", "workmanship", "root", "pass",
    "joint", "completion", "dissimilar", "heat", "treatment", "preheating",
    "repairs", "repair", "documentation", "documents",
    "weight", "weights", "span", "spans",
}
FORM_LABEL_RE = re.compile(
    r"^(?:prepared|approved|checked|reviewed|date|rev(?:ision)?|doc(?:ument)?\s*no|"
    r"ref(?:erence)?|subject|signature|name|designation)\b",
    re.IGNORECASE,
)
DRAWING_FIELD_RE = re.compile(
    r"\b(?:drawing|drg|dwg|sheet|scale|rev(?:ision)?|client|project|title|"
    r"checked|approved|designed|drawn|issued)\b",
    re.IGNORECASE,
)
DRAWING_CORE_FIELD_RE = re.compile(
    r"\b(?:drawing|drg|dwg|sheet|sht\.?|scale|ref\.?\s*drg|drn|chkd|appd)\b",
    re.IGNORECASE,
)
DRAWING_VIEW_RE = re.compile(
    r"\b(?:view|elevation|section|detail|plan|typ\.?|thk\.?|nts|ref\.?\s*drg|"
    r"drn|chkd|appd|sht\.?|rev\.?|scale)\b",
    re.IGNORECASE,
)
METADATA_TITLE_RE = re.compile(
    r"\b(?:report|document|doc(?:ument)?|package|specification)\s+name\s*:\s*(.+?)(?:\s+Generated\s+By\b|\s+Report\s+ID\b|$)",
    re.IGNORECASE,
)
SUBJECT_TITLE_RE = re.compile(
    r"\bsubject\s*:\s*(?:tender\s+for\s+)?(.+?)(?:\s+You\s+are\s+invited\b|\s+Tender\s+Type\b|$)",
    re.IGNORECASE,
)
LOCAL_CAPTION_RE = re.compile(
    r"^(?:table|figure|fig\.?|drawing|sketch|chart|annex(?:ure)?|appendix|"
    r"attachment|schedule)\s*[-.:]?\s*[A-Za-z0-9IVXLCM.-]*(?:\s+.+)?$",
    re.IGNORECASE,
)
SUBDOCUMENT_ANCHOR_RE = re.compile(
    r"\b(?:technical\s+advisory|advisory|procedure|specification|standard|"
    r"method\s+statement|status\s+table|scope\s+of\s+work)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FontSpec:
    font_id: str
    size: float
    family: str = ""
    color: str = ""


@dataclass
class TextNode:
    page_num: int
    top: float
    left: float
    width: float
    height: float
    font_size: float
    font_family: str
    text: str
    raw_text: str = ""
    bold: bool = False

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2


@dataclass
class PageInfo:
    page_num: int
    width: float
    height: float


@dataclass
class LogicalLine:
    page_num: int
    line_index: int
    top: float
    left: float
    width: float
    height: float
    text: str
    font_size: float
    bold: bool
    centered: bool
    word_count: int
    gap_above: float = 0.0
    node_count: int = 1
    row_density: int = 1
    repeated_key: str = ""
    is_running: bool = False
    is_table_like: bool = False
    role: str = "body"
    local_context: str = ""
    role_reasons: list[str] = field(default_factory=list)

    @property
    def right(self) -> float:
        return self.left + self.width


@dataclass
class TocEntry:
    page_num: int
    text: str
    level: int
    number: str = ""
    source_page: int = 0
    confidence: float = 0.0


@dataclass
class HeadingEvent:
    id: int
    page_num: int
    line_index: int
    top: float
    left: float
    text: str
    level: int
    score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    numbering: str = ""
    scheme: str = ""
    breadcrumb: str = ""
    toc_match: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParagraphChunk:
    chunk_id: int
    page_start: int
    page_end: int
    breadcrumb: str
    text: str
    chunk_type: str = "text"
    heading_id: Optional[int] = None
    confidence: float = 0.0


@dataclass
class DocumentContext:
    title: str
    path_parts: list[str] = field(default_factory=list)
    source: str = "inferred"

    @property
    def breadcrumb_root(self) -> str:
        parts = [self.title] if self.title else []
        for part in self.path_parts:
            clean = normalize_space(part)
            if clean and comparable_title(clean) not in {comparable_title(p) for p in parts}:
                parts.append(clean)
        return " > ".join(parts)


@dataclass
class DocumentArtifacts:
    source_pdf: str
    document_title: str
    document_context: DocumentContext
    page_count: int
    body_font_size: float
    headings: list[HeadingEvent]
    toc_entries: list[TocEntry]
    chunks: list[ParagraphChunk]
    pages: list[dict]
    sections: list[dict]
    warnings: list[str]
    extractor: dict


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_key(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"\d+", "<n>", text)
    text = re.sub(r"[_\-–—:;,.]+", " ", text)
    return normalize_space(text)


def comparable_title(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"^(?:\d+(?:\.\d+)*|[a-z]|[ivxlcm]+)[.)]?\s+", "", text)
    text = re.sub(r"^(?:section|part|chapter|appendix|annex(?:ure)?|attachment)\s+[a-z0-9ivxlcm.-]+\s*", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return normalize_space(text)


def is_all_caps_short(text: str, max_words: int = 14) -> bool:
    alpha = [c for c in text if c.isalpha()]
    return bool(alpha) and text == text.upper() and len(text.split()) <= max_words


def is_centered(left: float, width: float, page_width: float) -> bool:
    if page_width <= 0:
        return False
    return abs((left + width / 2) - page_width / 2) <= page_width * 0.08


def is_page_number(text: str) -> bool:
    return bool(PAGE_NUMBER_RE.match(normalize_space(text)))


def is_separator_line(text: str) -> bool:
    text = normalize_space(text)
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 12 and re.fullmatch(r"[_=\-—–.]+", compact):
        return True
    if len(compact) >= 20:
        marks = sum(1 for char in compact if char in "_=-—–.")
        return marks / max(len(compact), 1) >= 0.75
    return False


def has_long_blank_run(text: str) -> bool:
    return bool(re.search(r"[_]{8,}", normalize_space(text)))


def is_value_like(text: str) -> bool:
    t = normalize_space(text)
    lower = t.lower()
    if lower in VALUE_WORDS:
        return True
    if re.fullmatch(r"[\d:/.\- ]+", t):
        return True
    if re.fullmatch(r"[A-Z0-9]{5,}", t) and not re.search(r"[AEIOU]", t):
        return True
    if len(t.split()) <= 2 and re.fullmatch(r"[A-Z]{1,4}-?\d+[A-Z0-9-]*", t):
        return True
    return False


def looks_like_bank_or_contact_field(text: str) -> bool:
    text = normalize_space(text)
    return bool(re.search(
        r"\b(?:micr|ifsc|account\s+no|a/c|current\s+account|name\s+of\s+the\s+bank|"
        r"name\s+of\s+the\s+branch|address\s+of\s+the\s+bank|branch\s+code|swift|iban|gstin|pan\s+no)\b",
        text,
        re.IGNORECASE,
    ))


def looks_like_datasheet_value_line(text: str) -> bool:
    text = normalize_space(text)
    if re.match(r"^\d+(?:\.\d+)+\s+FOR\b", text, re.IGNORECASE):
        return True
    if (
        is_all_caps_short(text, 16)
        and re.search(r"\b(?:REFER|SKETCH|SHEET|EXISTING|CASE|CASES)\b", text, re.IGNORECASE)
        and re.search(r"\b\d+\s*$", text)
    ):
        return True
    return False


def looks_like_clause_table_header(text: str) -> bool:
    text = normalize_space(text)
    return bool(
        (
            re.match(r"^S\.?\s*No\.?\b", text, re.IGNORECASE)
            and re.search(r"\b(?:clause|title|description|remarks|status)\b", text, re.IGNORECASE)
        )
        or (
            re.match(r"^Sr\.?\s+", text, re.IGNORECASE)
            and re.search(r"\b(?:description|standard|remarks|equivalent)\b", text, re.IGNORECASE)
        )
    )


def looks_like_part_standard_reference(text: str) -> bool:
    text = normalize_space(text)
    return bool(
        re.match(r"^PART\s+\d+\s+", text, re.IGNORECASE)
        and re.search(
            r"\b(?:assessment|method|determination|preparation|field|test|classification|measurement|sampling)\b",
            text,
            re.IGNORECASE,
        )
        and not is_all_caps_short(text, 12)
    )


def looks_like_standard_table_row(text: str) -> bool:
    text = normalize_space(text)
    return bool(
        re.match(r"^\d+(?:\.\d+)*\s+", text)
        and re.search(r"\b(?:SSPC|NACE|ISO|ASTM|AMPP|Sa\s*\d|SP-\d)\b", text, re.IGNORECASE)
        and len(text.split()) >= 5
    )


def is_compact_code_token(token: str) -> bool:
    token = token.strip("()[],:;\"'")
    if not token:
        return False
    upper = token.upper().replace(" ", "")
    if upper in {"CS", "SS", "CR", "MO", "NI", "CU", "TI", "NB", "ER", "ENI", "ERTI"}:
        return True
    if re.fullmatch(r"(?:E|ER|ENI|ERTI|B)?[A-Z]{0,5}\d+[A-Z0-9.-]*", upper):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?[A-Z][A-Z0-9.-]*", upper):
        return True
    if re.fullmatch(r"[A-Z]{1,6}\d+[A-Z0-9.-]*", upper):
        return True
    return False


def looks_like_table_data_row(text: str) -> bool:
    """Reject compact numbered rows made mostly from engineering codes/values."""
    text = normalize_space(text)
    m = re.match(r"^\d{1,4}(?!\.\d)(?:[.)])?\s+(.+)$", text)
    if not m:
        return False
    rest = m.group(1)
    words = [word.strip("()[],:;\"'") for word in rest.split() if word.strip("()[],:;\"'")]
    if not words or len(words) > 14:
        return False
    lower_words = {word.lower() for word in words}
    if lower_words & HEADING_KEYWORDS:
        return False
    code_tokens = 0
    for word in words:
        parts = [part for part in re.split(r"[/&,+]", word) if part]
        if any(is_compact_code_token(part) for part in parts):
            code_tokens += 1
    material_hits = sum(
        1
        for word in words
        if re.fullmatch(
            r"(?:CS|SS\d{2,4}[A-Z]?|DUPLEX|MONEL|INCOLLOY|TITANIUM|TI|CAST|IRON|SUPER)",
            word.upper(),
        )
    )
    slash_code_row = "/" in rest and code_tokens >= 2
    dense_code_row = code_tokens + material_hits >= 3 and len(words) <= 10
    return slash_code_row or dense_code_row


def looks_like_numeric_table_value_row(text: str) -> bool:
    text = normalize_space(text)
    if not re.match(r"^\d{1,4}\s+\S+", text):
        return False
    tokens = text.split()
    if len(tokens) < 3 or len(tokens) > 10:
        return False
    alpha_chars = sum(1 for char in text if char.isalpha())
    numericish = sum(
        1
        for token in tokens
        if re.fullmatch(r"[\d./¼½¾\"'“”]+", token)
    )
    return alpha_chars <= 3 and numericish >= max(2, len(tokens) - 1)


def looks_like_sentence(text: str) -> bool:
    text = normalize_space(text)
    if len(text) > 180:
        return True
    if len(text.split()) > 22:
        return True
    return bool(re.search(r"[a-z].*,\s+[a-z]", text)) and not text.rstrip().endswith(":")


def looks_like_table_line(text: str) -> bool:
    lower = normalize_space(text).lower()
    if looks_like_table_data_row(text):
        return True
    if "|" in text and text.count("|") >= 2:
        return True
    if re.match(r"^(?:s[il]|sr)\s*\.?\s*no\b", lower):
        return True
    if re.match(r"^table\s*[-:]?\s*\d", lower):
        return True
    if re.search(r"\b(?:bm|a|m)\s*[-–—]?\s*ppe\b", lower):
        return True
    tokens = [token.strip("()[],:.;") for token in lower.split()]
    if not tokens:
        return False
    table_hits = sum(1 for token in tokens if token in TABLE_WORDS)
    coded = sum(1 for token in tokens if re.fullmatch(r"[a-z]{0,4}\d+[a-z0-9-]*", token))
    return table_hits >= 3 or (lower.startswith("description") and table_hits >= 2) or coded >= 4


def looks_like_form_label(text: str) -> bool:
    text = normalize_space(normalize_space(text).strip(":"))
    if FORM_LABEL_RE.match(text) and len(text.split()) <= 6:
        return True
    return text.lower() in {"prepared", "approved", "checked", "reviewed", "status codes"}


def looks_like_form_value_label(text: str) -> bool:
    text = normalize_space(text)
    if ":" not in text:
        return False
    label = normalize_space(text.split(":", 1)[0])
    if len(label) < 3 or len(label.split()) > 6:
        return False
    alpha = [char for char in label if char.isalpha()]
    if not alpha:
        return False
    if label.upper() != label:
        return False
    return bool(re.search(r"\b(?:pipe|pipes|fitting|fittings|flange|flanges|material|materials|metal|welding|root|filler|gas|gases|preheat|heat|treatment|hardness|code|method|property|joint|groove|process|pass|composition|others)\b", label, re.IGNORECASE))


def looks_like_welding_chart_header(text: str) -> bool:
    text = normalize_space(text)
    if not is_all_caps_short(text, 10):
        return False
    terms = re.findall(
        r"\b(?:BASE|WELD|WELDING|METAL|MATERIAL|MATERIALS|GROOVE|JOINT|JOINTS|"
        r"ROOT|FILLER|PASS|TREATMENT|PROCESS|GAS|COMPOSITION)\b",
        text,
        re.IGNORECASE,
    )
    return len({term.upper() for term in terms}) >= 3


def looks_like_inspection_table_header(text: str) -> bool:
    text = normalize_space(text)
    if not is_all_caps_short(text, 12):
        return False
    terms = re.findall(
        r"\b(?:STAGE|ACTIVITY|CHARACTERISTICS|RECORD|AGENCY|EXTENT|CHECK|"
        r"SUB|VENDOR|TPI|INSPECTION|TEST|DOCUMENT|FORMAT|ACCEPTANCE)\b",
        text,
        re.IGNORECASE,
    )
    return len({term.upper() for term in terms}) >= 3


def looks_like_standard_header_noise(text: str) -> bool:
    text = normalize_space(text)
    lower = text.lower()
    if "issued as standard" in lower:
        return True
    if "revised" in lower and "standard" in lower and re.search(r"\d{5,}", lower):
        return True
    if re.search(r"\bpage\s*\d+\s*of\s*\d+\b", lower):
        return True
    if "engineers standard specification" in lower:
        return True
    if re.search(r"\b\w?\s*ngineers\s+standard\s+specification\b", lower):
        return True
    if "encinefrs standard specification" in lower:
        return True
    if "standard specification" in lower and re.search(r"\b(?:rev|no\.?|limited|engineers)\b", lower):
        return True
    if re.search(r"\b(?:standard|specification)\s+no\.?\b", lower):
        return True
    return False


def looks_like_symbol_or_code_fragment(text: str) -> bool:
    text = normalize_space(text)
    if not text:
        return False
    alpha = sum(1 for char in text if char.isalpha())
    alnum = sum(1 for char in text if char.isalnum())
    if len(text) <= 10 and alpha <= 1 and re.search(r"[-=()[\]{}|]", text):
        return True
    if len(text) <= 16 and alnum <= 5 and re.search(r"[-=()[\]{}|]", text):
        return True
    return False


def looks_like_local_caption(text: str) -> bool:
    text = normalize_space(text)
    if not text or len(text.split()) > 22:
        return False
    if LOCAL_CAPTION_RE.match(text):
        return True
    return bool(re.match(r"^(?:note|notes)\s*[:.-]\s+", text, re.IGNORECASE))


def looks_like_drawing_field(text: str) -> bool:
    text = normalize_space(text)
    if not text or len(text.split()) > 12:
        return False
    if DRAWING_FIELD_RE.search(text):
        return True
    return bool(re.search(r"\b(?:A0|A1|A2|A3|A4)\b", text))


def looks_like_drawing_view_text(text: str) -> bool:
    text = normalize_space(text)
    if not text or len(text.split()) > 16:
        return False
    return bool(DRAWING_VIEW_RE.search(text))


def looks_like_drawing_sheet_page(
    line_count: int,
    drawing_hits: int,
    drawing_core_hits: int,
    drawing_view_hits: int,
    symbol_rows: int,
    prose_rows: int,
    caps_rows: int,
    table_rows: int,
) -> bool:
    if line_count < 12:
        return False
    drawing_signal = drawing_hits + drawing_core_hits + drawing_view_hits
    graphics_signal = symbol_rows + table_rows
    dense_sheet = (
        drawing_signal >= 6
        and symbol_rows >= 10
        and prose_rows <= 6
        and (caps_rows >= 18 or table_rows >= 5)
    )
    compact_drawing_sheet = (
        drawing_signal >= 6
        and symbol_rows >= 10
        and prose_rows <= 2
        and caps_rows >= 10
        and line_count <= 45
    )
    callout_drawing_sheet = (
        drawing_signal >= 10
        and (symbol_rows + table_rows) >= 8
        and caps_rows >= 12
        and prose_rows <= 2
    )
    view_heavy_sheet = (
        drawing_view_hits >= 8
        and graphics_signal >= 14
        and prose_rows <= 8
        and caps_rows >= 18
    )
    return dense_sheet or compact_drawing_sheet or callout_drawing_sheet or view_heavy_sheet


def looks_like_dense_table_sheet_page(
    line_count: int,
    table_rows: int,
    symbol_rows: int,
    prose_rows: int,
    caps_rows: int,
) -> bool:
    if line_count < 40 or prose_rows > 2:
        return False
    dense_rows = table_rows + symbol_rows
    return (
        table_rows >= max(18, int(line_count * 0.25))
        or (dense_rows >= int(line_count * 0.35) and caps_rows >= 8)
    )


def looks_like_mojibake(text: str) -> bool:
    text = normalize_space(text)
    if not text:
        return False
    marker_hits = len(re.findall(r"(?:keÀ|ef|Dee|Òe|³e|HeÀ|meÀ|ì|ð|þ|®|Æ|ﬁ|ﬂ|@@|@g)", text))
    latin_ext_hits = sum(1 for char in text if "\u00c0" <= char <= "\u024f")
    ligature_hits = sum(1 for char in text if "\ufb00" <= char <= "\ufb06")
    at_hits = text.count("@")
    return marker_hits >= 2 or latin_ext_hits >= 8 or ligature_hits >= 2 or at_hits >= 3


def has_clause_sentence_shape(text: str) -> bool:
    text = normalize_space(text).rstrip(":")
    words = text.split()
    if len(words) >= 12:
        return True
    if re.search(r"\b(?:shall|should|must|will|may|is|are|be|being|been|was|were|has|have|had|can|could)\b", text, re.IGNORECASE):
        return True
    if len(words) >= 8 and re.search(r"\b(?:the|and|or|of|for|in|with|to|by|as|but|not|from|during|under)\b", text, re.IGNORECASE):
        return True
    if re.search(r"[a-z].*,\s+[a-z]", text):
        return True
    return False


def is_compact_heading_phrase(text: str) -> bool:
    text = normalize_space(text).rstrip(":")
    if not text:
        return False
    words = text.split()
    if len(words) > 9:
        return False
    if has_clause_sentence_shape(text):
        return False
    if is_all_caps_short(text, 12):
        return True
    alpha_words = [w.strip("()[],:;.\"'") for w in words if re.search(r"[A-Za-z]", w)]
    if not alpha_words:
        return False
    titleish = sum(1 for word in alpha_words if word[:1].isupper() or word.isupper())
    return titleish / max(len(alpha_words), 1) >= 0.65


def is_numbered_body_clause(scheme: str, numbered_level: int, numbered_text: str, text: str) -> bool:
    if scheme != "decimal":
        return False
    clean = normalize_space(numbered_text)
    if not clean:
        return False
    title = re.sub(r"^\d{1,3}(?:\.\d{1,3}){0,8}[.)]?\s*", "", clean)
    first_word = title.split()[0].strip("()[],:;.\"'").lower() if title.split() else ""
    if first_word in HEADING_KEYWORDS and len(title.split()) <= 14:
        return False
    if is_all_caps_short(title, 14):
        return False
    if is_compact_heading_phrase(title):
        return False
    if text.rstrip().endswith(":") and len(title.split()) <= 10 and not has_clause_sentence_shape(title):
        return False
    if numbered_level >= 2 and has_clause_sentence_shape(title):
        return True
    if numbered_level == 1 and has_clause_sentence_shape(title) and not is_all_caps_short(title, 12):
        return True
    return False


def clause_context_from_line(text: str) -> str:
    scheme, number, _level, numbered_text = parse_numbering(text)
    if scheme != "decimal" or not number:
        return ""
    label = numbered_text if numbered_text.startswith(number) else normalize_space(f"{number} {numbered_text}")
    return f"Clause {label[:180]}"


def role_rank(role: str) -> int:
    ranks = {
        "body": 0,
        "caption": 1,
        "table": 2,
        "form": 3,
        "drawing": 4,
        "toc": 5,
        "running": 6,
        "page_number": 7,
    }
    return ranks.get(role, 0)


def parse_numbering(text: str) -> tuple[str, str, int, str]:
    """Return (scheme, number, level, title) for heading-like numbering."""
    stripped = normalize_space(text).strip("# ").strip()
    if (
        has_long_blank_run(stripped)
        or looks_like_bank_or_contact_field(stripped)
        or looks_like_datasheet_value_line(stripped)
        or looks_like_clause_table_header(stripped)
        or looks_like_standard_table_row(stripped)
    ):
        return "", "", 0, stripped
    if looks_like_part_standard_reference(stripped):
        return "", "", 0, stripped
    md = MD_HEADING_RE.match(stripped)
    if md:
        return "markdown", "", min(len(md.group(1)), 6), normalize_space(md.group(2))

    m = DECIMAL_RE.match(stripped)
    if m:
        number, delimiter, title = m.groups()
        # Avoid years and doc numbers such as "2024 Project Data".
        if delimiter is None and "." not in number and len(number) > 2:
            return "", "", 0, stripped
        title = normalize_space(title)
        if title and not is_value_like(title):
            parts = number.split(".")
            level = len(parts)
            if len(parts) > 1 and parts[-1] == "0":
                level -= 1
            return "decimal", number, min(max(level, 1), 6), f"{number} {title}"

    m = COMPACT_DECIMAL_RE.match(stripped)
    if m:
        number, title = m.groups()
        title = normalize_space(title)
        parts = number.split(".")
        level = len(parts)
        if len(parts) > 1 and parts[-1] == "0":
            level -= 1
        return "decimal", number, min(max(level, 1), 6), f"{number} {title}"

    m = SECTION_RE.match(stripped)
    if m:
        label, number, title = m.groups()
        full = normalize_space(" ".join(part for part in (label.upper(), number, title or "") if part))
        return "section", f"{label} {number}", 1, full

    m = APPENDIX_RE.match(stripped)
    if m:
        label, number, title = m.groups()
        title_text = normalize_space(title or "")
        if title_text and title_text[:1].islower():
            return "", "", 0, stripped
        if (
            title_text
            and len(title_text.split()) > 10
            and not is_all_caps_short(title_text, 16)
            and not DOC_TITLE_HINT_RE.search(title_text)
        ):
            return "", "", 0, stripped
        full = normalize_space(" ".join(part for part in (label.upper(), number or "", title or "") if part))
        return "appendix", normalize_space(f"{label} {number or ''}"), 1, full

    m = ALPHA_DECIMAL_RE.match(stripped)
    if m:
        letter, number, title = m.groups()
        title = normalize_space(title)
        level = min(1 + len(number.split(".")), 6)
        return "alpha_decimal", f"{letter.upper()}.{number}", level, f"{letter.upper()}.{number} {title}"

    m = ALPHA_RE.match(stripped)
    if m and len(m.group(2).split()) <= 14:
        return "alpha", m.group(1), 2, f"{m.group(1)}. {normalize_space(m.group(2))}"

    m = ROMAN_RE.match(stripped)
    if m and len(m.group(2).split()) <= 14:
        return "roman", m.group(1), 2, f"{m.group(1)}. {normalize_space(m.group(2))}"

    return "", "", 0, stripped


def fuzzy_ratio(a: str, b: str) -> float:
    aa = comparable_title(a)
    bb = comparable_title(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    if aa in bb or bb in aa:
        return 0.92
    return SequenceMatcher(None, aa, bb).ratio()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Poppler extraction
# ---------------------------------------------------------------------------

def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def split_layout_text_pages(layout_text: str) -> dict[int, str]:
    pages = layout_text.split("\f") if layout_text else []
    return {
        idx + 1: page.rstrip()
        for idx, page in enumerate(pages)
        if page.strip()
    }


def run_pdftotext(pdf_path: Path) -> tuple[dict[int, str], str]:
    if not command_exists("pdftotext"):
        return {}, "pdftotext_not_found"
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}, f"pdftotext_failed: {result.stderr.strip()[:300]}"
    return split_layout_text_pages(result.stdout), "pdftotext_layout"


def run_pdftohtml_xml(pdf_path: Path, work_dir: Path) -> tuple[Optional[Path], str]:
    if not command_exists("pdftohtml"):
        return None, "pdftohtml_not_found"
    prefix = work_dir / pdf_path.stem
    result = subprocess.run(
        ["pdftohtml", "-xml", "-hidden", "-i", "-nodrm", str(pdf_path), str(prefix)],
        capture_output=True,
        text=True,
        check=False,
    )
    candidates = [
        prefix.with_suffix(".xml"),
        work_dir / f"{pdf_path.stem}.xml",
        work_dir / f"{pdf_path.stem}s.xml",
    ]
    xml_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if result.returncode != 0 and not xml_path:
        return None, f"pdftohtml_failed: {result.stderr.strip()[:300]}"
    if not xml_path:
        xmls = sorted(work_dir.glob("*.xml"))
        xml_path = xmls[0] if xmls else None
    return xml_path, "pdftohtml_xml" if xml_path else "pdftohtml_no_xml"


def run_ocrmypdf_skip_text(pdf_path: Path, work_dir: Path, jobs: int = 4) -> tuple[Path, str]:
    """Add an OCR text layer to scanned pages, preserving existing text pages."""
    if not command_exists("ocrmypdf"):
        return pdf_path, "ocrmypdf_not_found"
    out_path = work_dir / f"{pdf_path.stem}.ocr.pdf"
    result = subprocess.run(
        [
            "ocrmypdf",
            "--skip-text",
            "--invalidate-digital-signatures",
            "--jobs",
            str(max(1, jobs)),
            "-l",
            "eng",
            str(pdf_path),
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not out_path.exists():
        detail = (result.stderr or result.stdout or "").strip()[:300]
        return pdf_path, f"ocrmypdf_failed: {detail}"
    return out_path, "ocrmypdf_skip_text"


def load_xml_root(xml_path: Path) -> ET.Element:
    try:
        return ET.parse(xml_path).getroot()
    except ET.ParseError:
        try:
            from lxml import etree as LET  # type: ignore
        except ImportError:
            raise
        parser = LET.XMLParser(recover=True)
        return LET.parse(str(xml_path), parser).getroot()


def parse_pdftohtml_xml(xml_path: Path) -> tuple[list[PageInfo], list[TextNode]]:
    root = load_xml_root(xml_path)
    pages: list[PageInfo] = []
    nodes: list[TextNode] = []
    known_fonts: dict[str, FontSpec] = {}

    for page in root.findall("page"):
        page_num = int(page.attrib.get("number", len(pages) + 1))
        width = float(page.attrib.get("width", 0.0))
        height = float(page.attrib.get("height", 0.0))
        pages.append(PageInfo(page_num=page_num, width=width, height=height))

        fonts = dict(known_fonts)
        for font in page.findall("fontspec"):
            spec = FontSpec(
                font_id=font.attrib.get("id", ""),
                size=float(font.attrib.get("size", 0.0)),
                family=font.attrib.get("family", ""),
                color=font.attrib.get("color", ""),
            )
            if spec.font_id:
                fonts[spec.font_id] = spec
                known_fonts[spec.font_id] = spec

        for text_elem in page.findall("text"):
            text = normalize_space("".join(text_elem.itertext()))
            if not text:
                continue
            font_id = text_elem.attrib.get("font", "")
            font = fonts.get(font_id, FontSpec(font_id, 0.0))
            nodes.append(
                TextNode(
                    page_num=page_num,
                    top=float(text_elem.attrib.get("top", 0.0)),
                    left=float(text_elem.attrib.get("left", 0.0)),
                    width=float(text_elem.attrib.get("width", 0.0)),
                    height=float(text_elem.attrib.get("height", 0.0)),
                    font_size=font.size,
                    font_family=font.family,
                    raw_text="".join(text_elem.itertext()),
                    text=text,
                    bold=any(child.tag.lower() == "b" for child in text_elem.iter()),
                )
            )

    return pages, nodes


def extract_with_fitz_fallback(pdf_path: Path) -> tuple[list[PageInfo], list[TextNode], dict[int, str], str]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return [], [], {}, "fitz_not_available"

    doc = fitz.open(str(pdf_path))
    pages: list[PageInfo] = []
    nodes: list[TextNode] = []
    layout_pages: dict[int, str] = {}
    for idx, page in enumerate(doc):
        page_num = idx + 1
        rect = page.rect
        pages.append(PageInfo(page_num=page_num, width=rect.width, height=rect.height))
        layout_pages[page_num] = page.get_text("text") or ""
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception:
            continue
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = normalize_space(span.get("text", ""))
                    if not text:
                        continue
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    font = span.get("font", "")
                    nodes.append(
                        TextNode(
                            page_num=page_num,
                            top=float(bbox[1]),
                            left=float(bbox[0]),
                            width=float(bbox[2] - bbox[0]),
                            height=float(bbox[3] - bbox[1]),
                            font_size=float(span.get("size", 0.0)),
                            font_family=font,
                            text=text,
                            raw_text=text,
                            bold="bold" in font.lower(),
                        )
                    )
    doc.close()
    return pages, nodes, layout_pages, "pymupdf_fallback"


# ---------------------------------------------------------------------------
# Logical line construction and document profile
# ---------------------------------------------------------------------------

def group_nodes_by_page(nodes: Iterable[TextNode]) -> dict[int, list[TextNode]]:
    page_map: dict[int, list[TextNode]] = defaultdict(list)
    for node in nodes:
        page_map[node.page_num].append(node)
    return page_map


# A superscript/subscript token sitting alone is one of: a raised "degree" sign
# (poppler decodes the ° glyph as a small raised o/0, often glued to its unit, e.g.
# "oC"/"0C"), a numeric exponent ("13" in 1×10^13), or an ordinal/footnote mark.
_DEGREE_SUP_RE = re.compile(r"^[oO0º°˚]\s*([CFcf])?$")
_NUM_SUP_RE = re.compile(r"^[+\-]?\d+$")


def merge_super_subscripts(nodes: list[TextNode]) -> list[TextNode]:
    """Fold superscript/subscript fragments back into their base token before
    rows are assembled, so they aren't scattered onto their own line.

    pdftohtml emits a superscript/subscript as a smaller-font node sitting
    slightly above/below and horizontally adjacent to its base. Left alone, the
    row grouper drops it on a separate line — turning ``1×10^13 ohm-cm`` into a
    floating ``13`` over ``1× 10 ohm-cm`` and ``27°C`` into ``27`` + ``oC``.
    Exponents are rejoined as caret notation (``1× 10^13``); the raised-o degree
    sign becomes ``°`` (``27°C``); ordinals/marks are appended inline.

    Returns a new node list (inputs are not mutated)."""
    if len(nodes) < 2:
        return nodes
    work = [replace(n) for n in nodes]
    sizes = [n.font_size for n in work if n.font_size > 0 and n.text.strip()]
    if not sizes:
        return nodes
    body = statistics.median(sizes)
    if body <= 0:
        return nodes

    # Pages can mix body sizes (e.g. a 15pt report with a 12pt sub-section), so a
    # fragment's "smallness" and the base's "largeness" are decided RELATIVE to
    # each candidate base, not against a single global size. Any node is a
    # potential base; we only try to attach nodes below the page median.
    smalls = [n for n in work if n.text.strip() and 0 < n.font_size < body * 0.95]
    if not smalls:
        return nodes

    # Phase 1 — match each fragment to a base using PRISTINE geometry, so a base
    # that already absorbed one fragment can't widen and over-capture the next.
    matches: list[tuple[TextNode, TextNode, str]] = []  # (base, small, rendered)
    consumed: set[int] = set()
    for sm in sorted(smalls, key=lambda n: (n.top, n.left)):
        sm_cy = sm.top + sm.height / 2.0
        best, best_key, best_is_super = None, None, True
        for b in work:
            if b is sm or not b.text.strip():
                continue
            # base must be clearly larger than the fragment
            if sm.font_size > b.font_size * 0.85:
                continue
            # fragment must be raised (superscript) or lowered (subscript) vs the
            # base — not merely a smaller word sitting on the same baseline.
            is_super = sm.bottom <= b.bottom - b.height * 0.12
            is_sub = sm.top >= b.top + b.height * 0.12
            if not (is_super or is_sub):
                continue
            # vertical band overlap and horizontal adjacency (sup/subs sit at the
            # base's right edge, sometimes slightly overlapping it).
            if sm_cy < b.top - b.height * 0.6 or sm_cy > b.bottom + b.height * 0.6:
                continue
            if b.left > sm.left + 2:
                continue
            gap = sm.left - b.right
            if gap < -max(b.width, 1.0) * 0.6 or gap > 10:
                continue
            key = abs(gap)
            if best_key is None or key < best_key:
                # direction by centre: fragment above the base's mid-line is a
                # superscript, below it a subscript.
                best, best_key = b, key
                best_is_super = sm_cy <= (b.top + b.bottom) / 2.0
        if best is None:
            continue
        token = sm.text.strip()
        m = _DEGREE_SUP_RE.match(token)
        if m:
            rendered = "°" + (m.group(1).upper() if m.group(1) else "")
        elif _NUM_SUP_RE.match(token):
            rendered = ("^" if best_is_super else "_") + token
        else:
            rendered = token  # ordinals / footnote marks: inline, no marker
        matches.append((best, sm, rendered))
        consumed.add(id(sm))

    if not consumed:
        return nodes
    # Phase 2 — apply merges left-to-right per base, then widen the base box.
    for base, sm, rendered in matches:
        base.text = base.text.rstrip() + rendered
        if base.raw_text:
            base.raw_text = base.raw_text.rstrip() + rendered
        base.width = max(base.right, sm.right) - base.left
    return [n for n in work if id(n) not in consumed]


def build_logical_lines_for_page(page: PageInfo, nodes: list[TextNode]) -> list[LogicalLine]:
    if not nodes:
        return []

    nodes = merge_super_subscripts(nodes)
    sorted_nodes = sorted(nodes, key=lambda item: (item.top, item.left))
    rows: list[list[TextNode]] = []
    for node in sorted_nodes:
        if not rows:
            rows.append([node])
            continue
        row_top = statistics.median(item.top for item in rows[-1])
        tolerance = max(2.5, min(5.0, node.height * 0.35 if node.height else 3.0))
        if abs(node.top - row_top) <= tolerance:
            rows[-1].append(node)
        else:
            rows.append([node])

    row_density_lookup = Counter(round(node.top / 6) for node in nodes)
    lines: list[LogicalLine] = []
    prev_bottom = 0.0
    for row in rows:
        pieces = sorted(row, key=lambda item: item.left)
        text = normalize_space(" ".join(piece.text for piece in pieces))
        if not text:
            continue
        left = min(piece.left for piece in pieces)
        right = max(piece.right for piece in pieces)
        top = min(piece.top for piece in pieces)
        bottom = max(piece.bottom for piece in pieces)
        tallest = max(pieces, key=lambda item: (item.font_size, item.height))
        gap = top - prev_bottom if prev_bottom else 0.0
        table_like = (
            looks_like_table_line(text)
            or len(pieces) >= 5
            or (len(pieces) >= 4 and not is_centered(left, right - left, page.width))
        )
        line = LogicalLine(
            page_num=page.page_num,
            line_index=len(lines),
            top=top,
            left=left,
            width=right - left,
            height=bottom - top,
            text=text,
            font_size=tallest.font_size,
            bold=any(piece.bold for piece in pieces),
            centered=is_centered(left, right - left, page.width),
            word_count=len(text.split()),
            gap_above=gap,
            node_count=len(pieces),
            row_density=row_density_lookup.get(round(top / 6), len(pieces)),
            repeated_key=normalize_key(text),
            is_table_like=table_like,
        )
        lines.append(line)
        prev_bottom = bottom

    return lines


def build_logical_lines(pages: list[PageInfo], nodes: list[TextNode]) -> dict[int, list[LogicalLine]]:
    node_map = group_nodes_by_page(nodes)
    page_lines = {
        page.page_num: build_logical_lines_for_page(page, node_map.get(page.page_num, []))
        for page in pages
    }
    return page_lines


def detect_running_lines(page_lines: dict[int, list[LogicalLine]], pages: list[PageInfo]) -> set[str]:
    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()
    page_heights = {page.page_num: page.height for page in pages}

    for page_num, lines in page_lines.items():
        if not lines:
            continue
        height = page_heights.get(page_num, 0.0)
        for line in lines:
            key = line.repeated_key
            if len(key) < 6 or is_page_number(line.text):
                continue
            if height and line.top <= height * 0.14:
                top_counts[key] += 1
            elif height and line.top >= height * 0.86:
                bottom_counts[key] += 1

    total_pages = max(len(pages), 1)
    threshold = max(2, int(total_pages * 0.30))
    running = {key for key, count in top_counts.items() if count >= threshold}
    running.update({key for key, count in bottom_counts.items() if count >= threshold})
    return running


def estimate_body_font_size(lines: Iterable[LogicalLine]) -> float:
    weighted: list[float] = []
    for line in lines:
        if line.font_size <= 0:
            continue
        if line.is_running or line.is_table_like or is_page_number(line.text):
            continue
        if len(line.text) < 3:
            continue
        weight = min(max(len(line.text), 1), 120)
        weighted.extend([round(line.font_size, 1)] * weight)
    if not weighted:
        return 12.0
    return Counter(weighted).most_common(1)[0][0]


def detect_left_margin(lines: Iterable[LogicalLine]) -> float:
    lefts = [
        round(line.left / 5) * 5
        for line in lines
        if line.word_count >= 3 and not line.centered and not line.is_running
    ]
    if not lefts:
        return 0.0
    ordered = sorted(lefts)
    return float(ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.10)))])


def document_title_from_lines(page_lines: dict[int, list[LogicalLine]], body_size: float) -> str:
    first_pages = [page_lines.get(page_num, []) for page_num in sorted(page_lines)[:3]]
    candidates: list[tuple[float, str]] = []
    for lines in first_pages:
        if is_toc_page(lines):
            continue
        for line in lines[:18]:
            text = normalize_space(line.text)
            if not text or line.is_running or is_page_number(text) or is_separator_line(text) or is_value_like(text):
                continue
            if looks_like_mojibake(text) or looks_like_standard_header_noise(text) or looks_like_symbol_or_code_fragment(text):
                continue
            if line.word_count > 18 or looks_like_table_line(text):
                continue
            score = 0.0
            if line.centered:
                score += 2.0
            if line.font_size >= body_size + 2:
                score += line.font_size - body_size
            if line.bold:
                score += 1.0
            if is_all_caps_short(text, 16):
                score += 1.0
            if DOC_TITLE_HINT_RE.search(text):
                score += 3.0
            if score >= 2.0:
                candidates.append((score, text))
    if candidates:
        candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return candidates[0][1][:180]
    return ""


def _title_candidate_score(line: LogicalLine, body_size: float, page_num: int) -> float:
    text = normalize_space(line.text)
    if not text or line.is_running or is_page_number(text) or is_value_like(text):
        return -10.0
    if looks_like_mojibake(text) or looks_like_standard_header_noise(text) or looks_like_symbol_or_code_fragment(text):
        return -10.0
    if looks_like_form_label(text) or looks_like_table_line(text):
        return -8.0
    if line.word_count > 24:
        return -5.0
    score = 0.0
    if page_num <= 2:
        score += 1.0
    if line.centered:
        score += 1.5
    if line.bold:
        score += 1.0
    if line.font_size >= body_size + 2:
        score += min(4.0, line.font_size - body_size)
    if is_all_caps_short(text, 18):
        score += 1.2
    if DOC_TITLE_HINT_RE.search(text):
        score += 4.0
    if CONTEXT_TITLE_HINT_RE.search(text):
        score += 1.5
    if re.search(r"\b(?:annex(?:ure)?|appendix|attachment|schedule|section|part)\b", text, re.IGNORECASE):
        score += 2.0
    return score


def infer_document_context(
    pdf_path: Path,
    page_lines: dict[int, list[LogicalLine]],
    headings: list[HeadingEvent],
    body_size: float,
) -> DocumentContext:
    """Infer stable context that should prefix every chunk breadcrumb."""
    def complete_title(title: str) -> str:
        title_key = normalize_space(title)
        if not title_key:
            return title
        for lines in page_lines.values():
            for idx, line in enumerate(lines):
                if normalize_space(line.text) != title_key:
                    continue
                parts = [title_key]
                prev = line
                for nxt in lines[idx + 1:idx + 5]:
                    next_text = normalize_space(nxt.text)
                    if (
                        not next_text
                        or nxt.is_running
                        or is_page_number(next_text)
                        or is_separator_line(next_text)
                        or looks_like_table_line(next_text)
                        or re.match(r"^\d{1,3}(?:\.\d+)?[.)]?\s+", next_text)
                    ):
                        break
                    current = normalize_space(" ".join(parts))
                    continuation_trigger = (
                        current.upper().endswith((" FOR", " OF", " AND", " TO", " IN", " BY", " WITH"))
                        or (prev.centered and nxt.centered and abs(prev.font_size - nxt.font_size) <= 2)
                        or (is_all_caps_short(current, 22) and is_all_caps_short(next_text, 18))
                    )
                    if continuation_trigger and len(next_text.split()) <= 12:
                        parts.append(next_text)
                        prev = nxt
                        continue
                    break
                return normalize_space(" ".join(parts))[:220]
        return title

    filename_title = pdf_path.stem.replace("_", " ").replace("-", " ")
    filename_title = re.sub(r"\bpart\d+\b", " ", filename_title, flags=re.IGNORECASE)
    filename_title = re.sub(r"\bp\d+\s*\d+\b", " ", filename_title, flags=re.IGNORECASE)
    filename_title = normalize_space(filename_title)
    line_title = document_title_from_lines(page_lines, body_size)
    metadata_title = ""
    subject_title = ""
    for page_num in sorted(page_lines)[:3]:
        for line in page_lines[page_num][:20]:
            m = METADATA_TITLE_RE.search(line.text)
            if m:
                metadata_title = normalize_space(m.group(1))
                break
            m = SUBJECT_TITLE_RE.search(line.text)
            if m:
                subject_title = normalize_space(m.group(1))
                break
        if metadata_title:
            break
        if subject_title:
            break

    def title_is_weak(title: str) -> bool:
        text = normalize_space(title)
        if not text:
            return True
        if looks_like_mojibake(text) or looks_like_standard_header_noise(text) or looks_like_symbol_or_code_fragment(text):
            return True
        if is_value_like(text) or looks_like_sentence(text):
            return True
        if text[:1].islower() and not STRONG_DOC_TITLE_HINT_RE.search(text):
            return True
        if text.endswith(".") and text[:1].islower():
            return True
        if re.match(r"^\d{1,3}(?:\.\d{1,3})*[.)]?\s+", text) and re.search(
            r"\b(?:details of|shall|should|must|may|navigate|to be followed|given below)\b",
            text,
            re.IGNORECASE,
        ):
            return True
        if re.fullmatch(r"(?:INR|USD|EUR)?\s*[\d., ]+(?:INR|USD|EUR)?", text, re.IGNORECASE):
            return True
        if len(text.split()) > 16 and not STRONG_DOC_TITLE_HINT_RE.search(text):
            return True
        return False

    first_heading_title = ""
    for heading in sorted(headings, key=lambda h: (h.page_num, h.top, h.left)):
        if heading.page_num > 2:
            break
        text = normalize_space(heading.text)
        scheme, _number, _level, _title = parse_numbering(text)
        if scheme in {"decimal", "section"}:
            continue
        if looks_like_mojibake(text) or looks_like_standard_header_noise(text) or looks_like_symbol_or_code_fragment(text):
            continue
        if text and not title_is_weak(text) and (is_all_caps_short(text, 18) or DOC_TITLE_HINT_RE.search(text)):
            first_heading_title = text
            break

    def choose_title(candidate: str, source: str = "inferred") -> DocumentContext:
        clean = normalize_space(candidate)
        if title_is_weak(clean) and first_heading_title:
            return DocumentContext(title=first_heading_title[:220], path_parts=[], source="heading_fallback")
        if title_is_weak(clean) and filename_title:
            return DocumentContext(title=filename_title[:220], path_parts=[], source="filename_fallback")
        return DocumentContext(title=clean[:220], path_parts=[], source=source)

    if metadata_title and not title_is_weak(metadata_title):
        return choose_title(metadata_title, "metadata")
    if subject_title and not title_is_weak(subject_title):
        return choose_title(subject_title, "subject")
    completed_line_title = complete_title(line_title) if line_title else ""
    if completed_line_title and STRONG_DOC_TITLE_HINT_RE.search(completed_line_title):
        return choose_title(completed_line_title)
    if filename_title and STRONG_DOC_TITLE_HINT_RE.search(filename_title):
        return DocumentContext(title=filename_title[:220], path_parts=[], source="filename")

    candidates: list[tuple[float, int, str]] = []
    for page_num in sorted(page_lines)[:3]:
        if is_toc_page(page_lines[page_num]):
            continue
        for line in page_lines[page_num][:30]:
            text = normalize_space(line.text)
            if looks_like_mojibake(text) or looks_like_standard_header_noise(text) or looks_like_symbol_or_code_fragment(text):
                continue
            score = _title_candidate_score(line, body_size, page_num)
            if looks_like_sentence(text) and not DOC_TITLE_HINT_RE.search(text):
                continue
            if CONTEXT_TITLE_HINT_RE.search(text) and not (line.bold or line.centered or is_all_caps_short(text, 18)):
                continue
            if not (DOC_TITLE_HINT_RE.search(text) or is_all_caps_short(text, 18)):
                continue
            if score >= 3.0:
                candidates.append((score, page_num, text))

    if candidates:
        candidates.sort(key=lambda item: (item[0], -item[1], len(item[2])), reverse=True)
        title = complete_title(candidates[0][2][:220])
        if line_title and DOC_TITLE_HINT_RE.search(line_title) and not DOC_TITLE_HINT_RE.search(title):
            title = complete_title(line_title)
    else:
        title = complete_title(line_title) if line_title else filename_title

    # The root should be stable document identity only. Chapter/section context
    # is supplied by the live heading stack so it cannot be duplicated or made
    # stale when the paragraph moves into another chapter.
    return choose_title(title)


# ---------------------------------------------------------------------------
# TOC parsing
# ---------------------------------------------------------------------------

def is_toc_page(lines: list[LogicalLine]) -> bool:
    if not lines:
        return False
    first_text = "\n".join(line.text for line in lines[:12]).lower()
    dot_rows = sum(1 for line in lines if TOC_LEADER_RE.search(line.text))
    numbered_rows = sum(1 for line in lines if TOC_ROW_RE.match(line.text))
    page_ref_rows = sum(
        1 for line in lines
        if re.search(r"(?:^|\s)(?:[ivxlcdm]+|\d{1,4})\s*$", line.text, re.IGNORECASE)
        and len(line.text.split()) >= 3
        and not looks_like_sentence(line.text)
    )
    index_list_rows = sum(
        1 for line in lines
        if re.match(r"^\s*\d{1,3}[.)]?\s+\S+", line.text)
        and len(line.text.split()) >= 2
        and len(line.text.split()) <= 14
        and not looks_like_sentence(line.text)
    )
    index_header = bool(
        re.search(r"(^|\n)\s*(?:index|contents)\s*($|\n)", first_text)
        or ("description" in first_text and "page no" in first_text)
        or ("clause" in first_text and "page" in first_text)
    )
    return (
        "table of contents" in first_text
        or re.search(r"(^|\n)\s*contents\s*($|\n)", first_text)
        or dot_rows >= 5
        or numbered_rows >= 8
        or (index_header and page_ref_rows >= 4)
        or (index_header and index_list_rows >= 8)
    )


def toc_level_from_number(number: str, left: float, left_margin: float) -> int:
    n = re.sub(r"\s*\.\s*", ".", normalize_space(number))
    if not n:
        return max(1, min(6, int((left - left_margin) // 24) + 1 if left_margin else 1))
    alpha_decimal = re.match(r"^[A-Z]\.(\d+(?:\.\d+)*)$", n, re.IGNORECASE)
    if alpha_decimal:
        return min(6, 1 + len(alpha_decimal.group(1).split(".")))
    scheme, parsed_number, level, _title = parse_numbering(f"{n} dummy")
    if scheme in {"decimal", "markdown"}:
        return max(1, level)
    if scheme in {"section", "appendix"}:
        return 1
    return max(1, min(6, int((left - left_margin) // 24) + 1 if left_margin else 2))


def normalize_toc_number_title(number: str, title: str) -> tuple[str, str]:
    number = normalize_space(number)
    title = normalize_space(title)
    if re.fullmatch(r"[A-Z]", number, re.IGNORECASE):
        m = re.match(r"^\.\s*(\d+(?:\.\d+)*)\s+(.+)$", title)
        if m:
            number = f"{number}.{m.group(1)}"
            title = normalize_space(m.group(2))
    return re.sub(r"\s*\.\s*", ".", number), title


def parse_toc_entries(page_lines: dict[int, list[LogicalLine]], left_margin: float) -> list[TocEntry]:
    entries: list[TocEntry] = []
    for source_page, lines in sorted(page_lines.items()):
        if not is_toc_page(lines):
            continue
        for line in lines:
            text = normalize_space(line.text)
            m = TOC_ROW_RE.match(text)
            if not m:
                continue
            title = normalize_space(m.group("title"))
            number = normalize_space(m.group("num"))
            number, title = normalize_toc_number_title(number, title)
            page_num = int(m.group("page"))
            if not title or len(title) < 3:
                continue
            if title.lower() in {"page", "pages"}:
                continue
            full_text = normalize_space(f"{number} {title}") if number else title
            entries.append(
                TocEntry(
                    page_num=page_num,
                    text=full_text[:220],
                    level=toc_level_from_number(number, line.left, left_margin),
                    number=number,
                    source_page=source_page,
                    confidence=0.85,
                )
            )
    # Deduplicate exact same TOC rows.
    seen = set()
    deduped: list[TocEntry] = []
    for entry in entries:
        key = (entry.page_num, comparable_title(entry.text), entry.level)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def match_toc(line_text: str, page_num: int, toc_entries: list[TocEntry]) -> Optional[TocEntry]:
    if not toc_entries:
        return None
    best: tuple[float, TocEntry] | None = None
    line_cmp = comparable_title(line_text)
    if len(line_cmp) < 4:
        return None
    line_scheme, line_number, _line_level, _line_title = parse_numbering(line_text)
    line_number_norm = re.sub(r"\s*\.\s*", ".", normalize_space(line_number)).lower()
    for entry in toc_entries:
        entry_cmp = comparable_title(entry.text)
        if not line_cmp or not entry_cmp:
            continue
        entry_number_norm = toc_entry_number(entry)
        if line_number_norm and entry_number_norm and line_number_norm != entry_number_norm:
            continue
        if len(line_cmp.split()) <= 1 and line_cmp != entry_cmp:
            continue
        if len(entry_cmp.split()) <= 2 and line_cmp != entry_cmp:
            continue
        # A body sentence that happens to contain one TOC keyword should not
        # inherit TOC authority.  Prefer exact/substring matches; require a
        # much stronger fuzzy score when lengths differ substantially.
        line_words = set(line_cmp.split())
        entry_words = set(entry_cmp.split())
        if len(entry_words) <= 2 and len(line_words) > len(entry_words) + 2 and not line_cmp.startswith(entry_cmp):
            continue
        overlap = len(line_words & entry_words) / max(len(entry_words), 1)
        if overlap < 0.6 and entry_cmp not in line_cmp and line_cmp not in entry_cmp:
            continue
        page_penalty = 0.0 if abs(entry.page_num - page_num) <= 1 else min(abs(entry.page_num - page_num) * 0.03, 0.25)
        ratio = fuzzy_ratio(line_text, entry.text) - page_penalty
        if abs(len(line_cmp.split()) - len(entry_cmp.split())) >= 3 and ratio < 0.88:
            continue
        if best is None or ratio > best[0]:
            best = (ratio, entry)
    if best and best[0] >= 0.72:
        return best[1]
    return None


# ---------------------------------------------------------------------------
# Local role classification
# ---------------------------------------------------------------------------

def classify_line_roles(
    page_lines: dict[int, list[LogicalLine]],
    pages: list[PageInfo],
    body_size: float,
    left_margin: float,
    toc_entries: list[TocEntry],
) -> None:
    """Classify each line by local structure, not by whole-document type."""
    page_by_num = {page.page_num: page for page in pages}
    toc_pages = {page_num for page_num, lines in page_lines.items() if is_toc_page(lines)}

    for page_num, lines in page_lines.items():
        page = page_by_num.get(page_num)
        page_height = page.height if page else 0.0
        page_width = page.width if page else 0.0
        drawing_hits = 0
        drawing_core_hits = 0
        drawing_view_hits = 0
        lower_zone_hits = 0
        symbol_rows = 0
        prose_rows = 0
        caps_rows = 0
        table_rows = 0
        structured_section_rows = 0
        for line in lines:
            if looks_like_drawing_field(line.text):
                drawing_hits += 1
                if page_height and line.top >= page_height * 0.58:
                    lower_zone_hits += 1
            if DRAWING_CORE_FIELD_RE.search(line.text):
                drawing_core_hits += 1
            if looks_like_drawing_view_text(line.text):
                drawing_view_hits += 1
            if re.fullmatch(r"[Xx<>=`'\" .:+\-/0-9A-Z]{3,}", normalize_space(line.text)):
                symbol_rows += 1
            if line.word_count >= 9 and re.search(r"\b(?:shall|should|procedure|manual|operation|requirement)\b", line.text, re.IGNORECASE):
                prose_rows += 1
            if is_all_caps_short(normalize_space(line.text), 16):
                caps_rows += 1
            if line.is_table_like:
                table_rows += 1
            scheme, _number, _level, heading_text = parse_numbering(line.text)
            if scheme == "decimal" and (
                is_all_caps_short(heading_text, 10)
                or DOC_TITLE_HINT_RE.search(heading_text)
                or len(heading_text.split()) <= 6
            ):
                structured_section_rows += 1
        appendix_index_rows = sum(
            1 for line in lines
            if APPENDIX_RE.match(normalize_space(line.text))
        )
        drawing_layout_page = (
            (
                drawing_core_hits >= 2 and drawing_hits >= 4 and (lower_zone_hits >= 2 or drawing_view_hits >= 3 or len(lines) <= 45)
            )
            or (
                drawing_view_hits >= 5 and (symbol_rows >= 4 or drawing_core_hits >= 2)
            )
        ) and prose_rows <= 6 and structured_section_rows == 0
        drawing_sheet_page = looks_like_drawing_sheet_page(
            len(lines),
            drawing_hits,
            drawing_core_hits,
            drawing_view_hits,
            symbol_rows,
            prose_rows,
            caps_rows,
            table_rows,
        )
        dense_table_sheet_page = looks_like_dense_table_sheet_page(
            len(lines),
            table_rows,
            symbol_rows,
            prose_rows,
            caps_rows,
        )
        drawing_like_page = drawing_layout_page or drawing_sheet_page or dense_table_sheet_page

        active_caption = ""
        for line in lines:
            text = normalize_space(line.text)
            reasons: list[str] = []
            role = "body"
            line_scheme, _line_number, _line_level, line_heading_text = parse_numbering(text)
            line_heading_words = [
                token.strip("().,:;").lower()
                for token in line_heading_text.split()
            ]
            structural_heading_line = bool(
                (
                    line_scheme in {"section", "appendix", "alpha_decimal"}
                    and not drawing_layout_page
                    and not drawing_sheet_page
                )
                or (
                    dense_table_sheet_page
                    and not drawing_layout_page
                    and not drawing_sheet_page
                    and
                    line_scheme == "decimal"
                    and (
                        DOC_TITLE_HINT_RE.search(line_heading_text)
                        or any(word in HEADING_KEYWORDS for word in line_heading_words)
                    )
                )
            )

            if line.is_running:
                role = "running"
                reasons.append("repeated_header_footer")
            elif is_page_number(text):
                role = "page_number"
                reasons.append("page_number")
            elif page_num in toc_pages or TOC_ROW_RE.match(text):
                role = "toc"
                reasons.append("toc")
            elif (
                appendix_index_rows >= 3
                and APPENDIX_RE.match(text)
                and not (
                    line.centered
                    or line.font_size >= body_size + 3.0
                    or (line.bold and line.word_count <= 4)
                )
            ):
                role = "toc"
                reasons.append("appendix_index")
            else:
                in_title_block = bool(
                    drawing_like_page
                    and (
                        (page_height and line.top >= page_height * 0.58)
                        or (page_width and line.left >= page_width * 0.45)
                    )
                )
                if looks_like_local_caption(text) and not looks_like_sentence(text):
                    role = "caption"
                    reasons.append("local_caption")
                    active_caption = text[:180]
                elif in_title_block and (looks_like_drawing_field(text) or line.is_table_like or line.node_count >= 3):
                    role = "drawing"
                    reasons.append("drawing_title_block")
                elif drawing_like_page and not structural_heading_line and (
                    looks_like_drawing_view_text(text)
                    or looks_like_drawing_field(text)
                    or line.is_table_like
                    or symbol_rows >= 4
                ):
                    role = "drawing"
                    reasons.append("drawing_page")
                elif looks_like_form_label(text):
                    role = "form"
                    reasons.append("form_label")
                elif looks_like_form_value_label(text):
                    role = "form"
                    reasons.append("form_value_label")
                elif line.is_table_like or (line.row_density >= 5 and not line.centered):
                    role = "table"
                    reasons.append("table_or_dense_row")

            if role_rank(role) < role_rank("toc") and active_caption and role in {"body", "table", "form"}:
                line.local_context = active_caption
            if role == "body" and not line.local_context:
                scheme, _number, numbered_level, numbered_text = parse_numbering(text)
                if is_numbered_body_clause(scheme, numbered_level, numbered_text, text):
                    line.local_context = clause_context_from_line(text)
                    reasons.append("numbered_clause_context")
            if role == "drawing":
                drawing_title = text
                if looks_like_drawing_field(text) and ":" in text:
                    drawing_title = normalize_space(text.split(":", 1)[-1])
                if drawing_title and not looks_like_form_label(drawing_title):
                    line.local_context = normalize_space(f"Drawing/title block: {drawing_title}")[:180]
            elif role == "caption":
                line.local_context = text[:180]

            line.role = role
            line.role_reasons = reasons


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def score_heading_line(
    line: LogicalLine,
    body_size: float,
    left_margin: float,
    repeated_counts: Counter[str],
    toc_entries: list[TocEntry],
) -> tuple[float, list[str], str, str, int, str, Optional[TocEntry]]:
    text = normalize_space(line.text)
    reasons: list[str] = []
    score = 0.0
    scheme, number, numbered_level, numbered_text = parse_numbering(text)
    toc_match_entry = match_toc(text, line.page_num, toc_entries)
    font_delta = line.font_size - body_size

    if not text or len(text) < 3:
        return -10.0, ["too_short"], "", "", 0, text, None
    if is_separator_line(text):
        return -10.0, ["separator_line"], "", "", 0, text, None
    if is_page_number(text):
        return -10.0, ["page_number"], "", "", 0, text, None
    if looks_like_table_data_row(text) or looks_like_numeric_table_value_row(text):
        return -7.0, ["table_value_row"], "", "", 0, text, None
    if line.role == "page_number":
        return -10.0, ["page_number"], "", "", 0, text, None
    if line.is_running:
        return -9.0, ["running_header_footer"], "", "", 0, text, None
    if line.role == "running":
        return -9.0, ["running_header_footer"], "", "", 0, text, None
    if line.role == "toc":
        return -8.0, ["toc_row"], "", "", 0, text, None
    if is_value_like(text):
        return -7.0, ["value_like"], "", "", 0, text, None
    if has_long_blank_run(text):
        return -7.0, ["blank_form_line"], "", "", 0, text, None
    if looks_like_bank_or_contact_field(text):
        return -7.0, ["bank_or_contact_field"], "", "", 0, text, None
    if looks_like_datasheet_value_line(text):
        return -7.0, ["datasheet_value_line"], "", "", 0, text, None
    if looks_like_clause_table_header(text):
        return -7.0, ["clause_table_header"], "", "", 0, text, None
    if looks_like_standard_table_row(text):
        return -7.0, ["standard_table_row"], "", "", 0, text, None
    if looks_like_part_standard_reference(text):
        return -6.0, ["part_standard_reference"], "", "", 0, text, None
    if looks_like_form_value_label(text):
        return -6.5, ["form_value_label"], "", "", 0, text, None
    if looks_like_symbol_or_code_fragment(text):
        return -6.5, ["symbol_or_code_fragment"], "", "", 0, text, None
    if not scheme and looks_like_standard_header_noise(text):
        return -6.5, ["standard_header_noise"], "", "", 0, text, None
    if re.match(r"^\d{1,3}\s*=\s+", text):
        return -6.0, ["equation_definition"], "", "", 0, text, None
    if not scheme and (looks_like_welding_chart_header(text) or looks_like_inspection_table_header(text)):
        reason = "welding_chart_header" if looks_like_welding_chart_header(text) else "inspection_table_header"
        return -6.0, [reason], "", "", 0, text, None
    if line.role == "drawing":
        return -7.0, ["drawing_title_block"], "", "", 0, text, None
    if line.role == "form" and not (
        scheme in {"section", "appendix", "markdown"} and (line.centered or line.bold or font_delta >= 2.0)
    ):
        return -6.5, ["form_region"], "", "", 0, text, None
    if line.role == "table" and not (
        scheme in {"section", "appendix", "markdown"} and (line.centered or line.bold or font_delta >= 2.0)
    ):
        return -6.5, ["table_region"], "", "", 0, text, None
    if line.role == "caption" and re.match(r"^(?:table|figure|fig\.?|drawing|sketch|plate|chart)\b", text, re.IGNORECASE):
        return -5.5, ["local_caption"], "", "", 0, text, None
    if looks_like_mojibake(text):
        return -5.0, ["mojibake_text"], "", "", 0, text, None
    if (
        scheme == "appendix"
        and re.match(r"^annexure\s+\d+\s*\.\s*\d+", text, re.IGNORECASE)
        and line.word_count > 5
    ):
        return -5.0, ["appendix_index_row"], "", "", 0, text, None
    if is_numbered_body_clause(scheme, numbered_level, numbered_text, text):
        return -4.5, ["numbered_body_clause"], "", "", 0, text, None
    if not scheme and has_clause_sentence_shape(text) and not (
        is_all_caps_short(text, 14)
        or text.rstrip().endswith(":")
        or DOC_TITLE_HINT_RE.search(text)
    ):
        return -4.5, ["sentence_body_line"], "", "", 0, text, None
    if looks_like_drawing_view_text(text) and line.word_count <= 6 and not re.search(r"\b(?:chapter|section)\b", text, re.IGNORECASE):
        return -5.0, ["drawing_annotation"], "", "", 0, text, None
    if re.fullmatch(
        r"\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*\s+"
        r"(?:THK\.?|TYP\.?|DETAIL|ELEVATION|VIEW|PLAN|R\d+|WEB\s+PLATE).{0,50}",
        text,
        re.IGNORECASE,
    ):
        return -5.0, ["drawing_dimension_annotation"], "", "", 0, text, None
    if re.fullmatch(r"[`'\"A-Z0-9 ]{1,12}", text) and text.count("`") + text.count("'") >= 2:
        return -5.0, ["symbol_annotation"], "", "", 0, text, None
    if looks_like_form_label(text):
        return -6.0, ["form_label"], "", "", 0, text, None
    if line.is_table_like:
        # Multi-column rows are usually table content, not document hierarchy.
        # Allow exceptionally styled/centered captions to continue scoring.
        if not (line.centered and (line.font_size - body_size) >= 2.0):
            return -6.0, ["table_row"], "", "", 0, text, None
        score -= 4.0
        reasons.append("table_like")
    if line.row_density >= 5 and not line.centered:
        score -= 3.0
        reasons.append("dense_row")

    if font_delta >= 6:
        score += 4.0
        reasons.append("very_large_font")
    elif font_delta >= 3:
        score += 2.5
        reasons.append("large_font")
    elif font_delta >= 1.2:
        score += 1.2
        reasons.append("slightly_large_font")

    gap_signal = line.gap_above >= max(8.0, line.height * 1.4)
    centered_title_shape = bool(
        line.centered
        and len(text) <= 180
        and (
            is_all_caps_short(text, 14)
            or line.bold
            or font_delta >= 1.2
        )
    )

    if line.bold and len(text) <= 180:
        score += 1.3
        reasons.append("bold")
    if centered_title_shape:
        score += 1.2
        reasons.append("centered")
    if gap_signal and line.word_count <= 22:
        score += 1.0
        reasons.append("gap_above")
    if line.left <= left_margin + 28 and line.word_count <= 22:
        score += 0.5
        reasons.append("left_aligned")

    font_delta = line.font_size - body_size
    has_structural_style = (
        line.bold
        or centered_title_shape
        or gap_signal
        or font_delta >= 0.8
        or bool(toc_match_entry)
    )
    has_list_heading_style = (
        line.bold
        or gap_signal
        or font_delta >= 0.8
        or bool(toc_match_entry)
    )

    if scheme:
        plain_integer = (
            scheme == "decimal"
            and "." not in number
            and re.match(r"^\d{1,3}\s+\S+", text)
            and not re.match(r"^\d{1,3}[.)]\s+\S+", text)
        )
        integer_list_item = (
            scheme == "decimal"
            and "." not in number
            and re.match(r"^\d{1,3}[.)]?\s+\S+", text)
        )
        numbered_words = numbered_text.split()
        numbered_has_table_terms = any(token.strip("().,:;").lower() in TABLE_WORDS for token in numbered_words)
        numbered_has_heading_terms = any(token.strip("().,:;").lower() in HEADING_KEYWORDS for token in numbered_words)
        alpha_decimal_value = (
            scheme == "alpha_decimal"
            and not toc_match_entry
            and len(numbered_words) <= 6
            and not numbered_has_heading_terms
            and not DOC_TITLE_HINT_RE.search(numbered_text)
        )
        if alpha_decimal_value:
            return -5.5, ["alpha_decimal_value"], "", "", 0, text, None
        clause_modal = bool(re.search(r"\b(?:shall|should|must|will|may|is|are|be|being|been)\b", numbered_text, re.IGNORECASE))
        trailing_colon_heading = bool(text.rstrip().endswith(":") and (is_all_caps_short(numbered_text, 14) or not clause_modal))
        decimal_has_layout_signal = (
            line.bold
            or gap_signal
            or (font_delta >= 1.2 if plain_integer and not toc_match_entry else font_delta >= 0.8)
            or bool(toc_match_entry)
            or centered_title_shape
            or trailing_colon_heading
            or is_all_caps_short(numbered_text, 12)
            or (numbered_level <= 2 and DOC_TITLE_HINT_RE.search(numbered_text))
        )
        integer_item_is_probably_list = (
            integer_list_item
            and not toc_match_entry
            and not DOC_TITLE_HINT_RE.search(numbered_text)
            and not text.rstrip().endswith(":")
            and not is_all_caps_short(numbered_text)
            and not numbered_has_heading_terms
        )
        weak_plain_integer_list = (
            plain_integer
            and not toc_match_entry
            and not numbered_has_heading_terms
            and not DOC_TITLE_HINT_RE.search(numbered_text)
            and not is_all_caps_short(numbered_text)
            and not text.rstrip().endswith(":")
            and not line.bold
            and not gap_signal
            and font_delta < 1.2
        )
        short_plain_integer_value = (
            plain_integer
            and not toc_match_entry
            and len(numbered_words) <= 8
            and not numbered_has_heading_terms
            and not DOC_TITLE_HINT_RE.search(numbered_text)
            and not text.rstrip().endswith(":")
        )
        punctuated_integer_value = (
            scheme == "decimal"
            and "." not in number
            and re.match(r"^\d{1,3}[.)]\s+\d", text)
            and not toc_match_entry
            and not numbered_has_heading_terms
            and not DOC_TITLE_HINT_RE.search(numbered_text)
        )
        if short_plain_integer_value or punctuated_integer_value:
            reason = "punctuated_integer_value" if punctuated_integer_value else "short_plain_integer_value"
            return -5.5, [reason], "", "", 0, text, None
        numbered_body_clause = (
            not toc_match_entry
            and is_numbered_body_clause(scheme, numbered_level, numbered_text, text)
        )
        if (
            scheme in {"decimal", "alpha_decimal"}
            and len(numbered_words) <= 24
            and not looks_like_sentence(numbered_text)
            and decimal_has_layout_signal
            and not (plain_integer and not has_structural_style and not DOC_TITLE_HINT_RE.search(numbered_text))
            and not integer_item_is_probably_list
            and not weak_plain_integer_list
            and not short_plain_integer_value
            and not numbered_body_clause
            and not numbered_has_table_terms
        ):
            score += 4.0
            reasons.append(f"numbered_{numbered_level}")
        elif scheme == "section":
            score += 4.2
            reasons.append(scheme)
        elif scheme == "appendix":
            appendix_index_row = bool(
                re.match(
                    r"^(?:APPENDIX|ANNEX(?:URE)?|ATTACHMENT|ENCLOSURE|EXHIBIT|SCHEDULE)\s+"
                    r"\S+\s+\d+\s+",
                    numbered_text,
                    re.IGNORECASE,
                )
            )
            if not appendix_index_row or (line.centered and font_delta >= 3.0 and line.word_count <= 8):
                score += 4.2
                reasons.append(scheme)
        elif (
            scheme in {"alpha", "roman"}
            and has_list_heading_style
            and not looks_like_sentence(numbered_text)
            and (
                line.bold
                or text.rstrip().endswith(":")
                or is_all_caps_short(numbered_text, 10)
            )
            and not has_clause_sentence_shape(numbered_text)
        ):
            score += 2.0
            reasons.append(scheme)
        elif scheme == "markdown":
            score += 4.5
            reasons.append("markdown_heading")

    if is_all_caps_short(text) and line.word_count >= 2:
        score += 1.8
        reasons.append("all_caps_short")
    if text.endswith(":") and line.word_count <= 16:
        score += 1.0
        reasons.append("trailing_colon")
    if (
        not scheme
        and text.endswith(":")
        and line.word_count <= 8
        and is_compact_heading_phrase(text.rstrip(":"))
        and not has_clause_sentence_shape(text.rstrip(":"))
    ):
        score += 2.5
        reasons.append("compact_colon_heading")
    if toc_match_entry:
        score += 3.0
        reasons.append("toc_match")

    if repeated_counts[text] >= 3 and not line.centered:
        score -= 2.0
        reasons.append("repeated_text")
    if looks_like_sentence(text) and not scheme and not toc_match_entry:
        score -= 2.5
        reasons.append("sentence_like")
    if line.word_count <= 3 and not line.centered and not scheme and not text.endswith(":"):
        score -= 2.0
        reasons.append("short_unshaped")
    if line.word_count > 26:
        score -= 4.0
        reasons.append("too_many_words")

    heading_text = numbered_text if scheme in {"decimal", "alpha_decimal", "section", "appendix", "alpha", "roman", "markdown"} else text.rstrip(":")
    return score, reasons, scheme, number, numbered_level, heading_text, toc_match_entry


def merge_heading_lines(candidates: list[dict], pages_by_num: dict[int, PageInfo]) -> list[dict]:
    """Merge multi-line heading candidates on the same page."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: (item["line"].page_num, item["line"].top, item["line"].left))
    groups: list[list[dict]] = []
    for cand in candidates:
        line = cand["line"]
        if not groups:
            groups.append([cand])
            continue
        prev = groups[-1][-1]
        prev_line = prev["line"]
        same_page = line.page_num == prev_line.page_num
        close = line.top - prev_line.top <= (28 if (line.centered and prev_line.centered) else 12)
        similar_style = abs(line.font_size - prev_line.font_size) <= 2.0 and abs(line.left - prev_line.left) <= 45
        both_centered = line.centered and prev_line.centered
        if same_page and close and (both_centered or similar_style):
            # Do not merge two separate decimal clauses.
            if cand["scheme"] == "decimal" and prev["scheme"] == "decimal":
                groups.append([cand])
            else:
                groups[-1].append(cand)
        else:
            groups.append([cand])

    merged: list[dict] = []
    for group in groups:
        line = group[0]["line"]
        text = normalize_space(" ".join(item["heading_text"] for item in group))
        score = round(sum(item["score"] for item in group), 2)
        reasons = sorted({reason for item in group for reason in item["reasons"]})
        scheme, number, numbered_level, parsed_text = parse_numbering(text)
        toc_match_entry = next((item["toc_match"] for item in group if item["toc_match"]), None)
        merged.append({
            "line": line,
            "text": parsed_text if scheme else text.rstrip(":"),
            "score": score,
            "reasons": reasons,
            "scheme": scheme or group[0]["scheme"],
            "number": number or group[0]["number"],
            "numbered_level": numbered_level or group[0]["numbered_level"],
            "toc_match": toc_match_entry,
        })
    return merged


def extend_heading_candidates_with_continuations(
    candidates: list[dict],
    page_lines: dict[int, list[LogicalLine]],
) -> list[dict]:
    """Attach one short wrapped line to heading candidates when layout makes it clear."""
    if not candidates:
        return []
    candidate_lines = {(cand["line"].page_num, cand["line"].line_index) for cand in candidates}
    extended: list[dict] = []
    for cand in candidates:
        line: LogicalLine = cand["line"]
        if cand["scheme"] not in {"decimal", "section", "appendix"} or cand["heading_text"].rstrip().endswith(":"):
            extended.append(cand)
            continue
        lines = page_lines.get(line.page_num, [])
        next_line = next((item for item in lines if item.line_index == line.line_index + 1), None)
        if not next_line or (next_line.page_num, next_line.line_index) in candidate_lines:
            extended.append(cand)
            continue
        next_text = normalize_space(next_line.text)
        if (
            not next_text
            or next_line.is_running
            or next_line.role in {"table", "toc", "drawing", "form", "page_number"}
            or is_page_number(next_text)
            or looks_like_table_line(next_text)
            or re.match(r"^(?:\d{1,3}(?:\.\d+)?|[a-zA-Z])[.)]\s+", next_text)
            or next_line.word_count > 10
            or next_line.gap_above > max(12.0, line.height * 1.8)
        ):
            extended.append(cand)
            continue
        heading_text = normalize_space(cand["heading_text"])
        continuation_trigger = (
            heading_text.lower().endswith((" of", " for", " and", " or", " to", " in", " with", " by", " as", " per"))
            or next_text[:1].islower()
            or heading_text.count('"') % 2 == 1
            or heading_text.count("“") > heading_text.count("”")
            or heading_text.count("(") > heading_text.count(")")
        )
        aligned = next_line.centered == line.centered or abs(next_line.left - line.left) <= 60
        if continuation_trigger and aligned:
            updated = dict(cand)
            updated["heading_text"] = normalize_space(f"{heading_text} {next_text}")
            updated["score"] = cand["score"] + 0.4
            updated["reasons"] = sorted(set(cand["reasons"] + ["wrapped_heading"]))
            extended.append(updated)
        else:
            extended.append(cand)
    return extended


def assign_heading_level(cand: dict, body_size: float, size_ranks: dict[float, int]) -> int:
    line: LogicalLine = cand["line"]
    toc_match_entry: Optional[TocEntry] = cand["toc_match"]
    if toc_match_entry:
        return min(max(toc_match_entry.level, 1), 6)
    if cand["scheme"] == "markdown":
        return min(max(cand["numbered_level"], 1), 6)
    if cand["scheme"] in {"decimal", "section", "appendix", "alpha_decimal"}:
        return min(max(cand["numbered_level"] or 1, 1), 6)
    if cand["scheme"] in {"alpha", "roman"}:
        return 2
    if line.centered and line.font_size >= body_size + 4:
        return 1
    if line.centered and is_all_caps_short(cand["text"], 18):
        return 2
    if line.font_size > body_size + 0.5:
        return min(size_ranks.get(round(line.font_size, 1), 3), 6)
    if is_all_caps_short(cand["text"]):
        return 2
    return 3


def validate_heading_sequence(headings: list[HeadingEvent]) -> list[str]:
    warnings: list[str] = []
    decimal_seen: set[str] = set()
    prev_top_num: Optional[int] = None
    for heading in headings:
        if heading.scheme != "decimal" or not heading.numbering:
            continue
        parts = heading.numbering.split(".")
        is_zero_top = len(parts) > 1 and parts[-1] == "0"
        if len(parts) > 1 and not is_zero_top:
            parent = ".".join(parts[:-1])
            if parent not in decimal_seen:
                msg = f"p{heading.page_num}: decimal heading '{heading.numbering}' has no detected parent '{parent}'"
                heading.warnings.append(msg)
                warnings.append(msg)
        else:
            try:
                top = int(parts[0])
                if prev_top_num is not None and top > prev_top_num + 2:
                    msg = f"p{heading.page_num}: top-level decimal jump {prev_top_num} -> {top}"
                    heading.warnings.append(msg)
                    warnings.append(msg)
                prev_top_num = top
            except ValueError:
                pass
        decimal_seen.add(heading.numbering)
        if is_zero_top:
            decimal_seen.add(".".join(parts[:-1]))
    return warnings


def detect_headings(
    page_lines: dict[int, list[LogicalLine]],
    pages: list[PageInfo],
    body_size: float,
    left_margin: float,
    toc_entries: list[TocEntry],
) -> tuple[list[HeadingEvent], list[str]]:
    all_lines = [line for lines in page_lines.values() for line in lines]
    repeated_counts = Counter(line.text for line in all_lines if len(line.text) >= 5)
    font_sizes = sorted(
        {round(line.font_size, 1) for line in all_lines if line.font_size > body_size + 0.5},
        reverse=True,
    )
    size_ranks = {size: idx + 1 for idx, size in enumerate(font_sizes[:6])}
    pages_by_num = {page.page_num: page for page in pages}
    toc_pages = {page_num for page_num, lines in page_lines.items() if is_toc_page(lines)}

    raw_candidates: list[dict] = []
    for line in all_lines:
        # TOC rows are authoritative input for hierarchy reconstruction, but
        # they are not occurrences of the headings themselves.
        if line.page_num in toc_pages:
            continue
        score, reasons, scheme, number, numbered_level, heading_text, toc_match_entry = score_heading_line(
            line, body_size, left_margin, repeated_counts, toc_entries
        )
        if score < 3.0:
            continue
        if not scheme and not toc_match_entry and score < 4.0:
            continue
        if (
            scheme == "decimal"
            and not toc_match_entry
            and reasons == [f"numbered_{numbered_level}"]
        ):
            continue
        raw_candidates.append({
            "line": line,
            "score": score,
            "reasons": reasons,
            "scheme": scheme,
            "number": number,
            "numbered_level": numbered_level,
            "heading_text": heading_text,
            "toc_match": toc_match_entry,
        })

    raw_candidates = extend_heading_candidates_with_continuations(raw_candidates, page_lines)
    merged = merge_heading_lines(raw_candidates, pages_by_num)
    headings: list[HeadingEvent] = []
    for cand in merged:
        line: LogicalLine = cand["line"]
        level = assign_heading_level(cand, body_size, size_ranks)
        confidence = max(0.05, min(0.99, cand["score"] / 9.0))
        headings.append(
            HeadingEvent(
                id=len(headings) + 1,
                page_num=line.page_num,
                line_index=line.line_index,
                top=line.top,
                left=line.left,
                text=cand["text"][:260],
                level=level,
                score=round(cand["score"], 2),
                confidence=round(confidence, 3),
                reasons=cand["reasons"],
                numbering=cand["number"],
                scheme=cand["scheme"],
                toc_match=cand["toc_match"].text if cand["toc_match"] else None,
            )
        )

    # Deduplicate repeated same heading on adjacent pages, usually running headers.
    deduped: list[HeadingEvent] = []
    seen_recent: dict[str, HeadingEvent] = {}
    for heading in sorted(headings, key=lambda h: (h.page_num, h.top, h.left)):
        key = comparable_title(heading.text)
        prev = seen_recent.get(key)
        if prev and not heading.scheme and not heading.toc_match:
            continue
        if prev and heading.page_num - prev.page_num <= 2 and heading.level == prev.level:
            if "toc_match" not in heading.reasons and heading.score <= prev.score + 1:
                continue
        deduped.append(heading)
        seen_recent[key] = heading

    attach_breadcrumbs(deduped)
    warnings = validate_heading_sequence(deduped)
    return deduped, warnings


def attach_breadcrumbs(headings: list[HeadingEvent], root: str = "") -> None:
    stack: list[tuple[int, str]] = []
    for heading in sorted(headings, key=lambda h: (h.page_num, h.top, h.left)):
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.text))
        parts = [root] if root else []
        parts.extend(text for _level, text in stack)
        heading.breadcrumb = " > ".join(text for text in parts if text)


def filter_document_title_headings(headings: list[HeadingEvent], document_title: str) -> list[HeadingEvent]:
    if not document_title:
        return headings
    doc_key = comparable_title(document_title)
    doc_words = set(doc_key.split())

    def is_cover_title_fragment(heading: HeadingEvent) -> bool:
        if heading.scheme or heading.toc_match:
            return False
        key = comparable_title(heading.text)
        if not key:
            return False
        words = key.split()
        first_word = words[0] if words else ""
        title_word_overlap = sum(1 for word in words if word in doc_words)
        if key == doc_key:
            return True
        if len(key) >= 20 and doc_key.startswith(key):
            return True
        if looks_like_standard_header_noise(heading.text):
            return True
        if (
            heading.page_num <= 5
            and len(words) >= 5
            and doc_words
            and title_word_overlap / max(len(words), 1) >= 0.55
        ):
            return True
        if (
            heading.page_num <= 2
            and 1 <= len(words) <= 6
            and all(word in doc_words for word in words)
        ):
            return True
        if (
            heading.page_num <= 2
            and 1 <= len(words) <= 8
            and title_word_overlap >= max(1, len(words) // 2)
        ):
            return True
        if (
            heading.page_num <= 2
            and first_word not in HEADING_KEYWORDS
            and not DOC_TITLE_HINT_RE.search(heading.text)
            and not STRONG_DOC_TITLE_HINT_RE.search(heading.text)
        ):
            return True
        if (
            heading.page_num > 2
            and 2 <= len(words) <= 8
            and is_all_caps_short(heading.text, 10)
            and all(word in doc_words for word in words)
        ):
            return True
        return False

    filtered = [
        heading for heading in headings
        if not is_cover_title_fragment(heading)
    ]
    for idx, heading in enumerate(filtered, start=1):
        heading.id = idx
    attach_breadcrumbs(filtered)
    return filtered


def filter_unstructured_headings_after_numbering(headings: list[HeadingEvent]) -> list[HeadingEvent]:
    filtered: list[HeadingEvent] = []
    numbered_context = False
    for heading in sorted(headings, key=lambda h: (h.page_num, h.top, h.left)):
        if heading.scheme in {"decimal", "alpha_decimal", "section", "appendix"}:
            numbered_context = True
            filtered.append(heading)
            continue
        if not numbered_context or heading.toc_match or heading.scheme:
            filtered.append(heading)
            continue
        text = normalize_space(heading.text)
        words = comparable_title(text).split()
        first_word = words[0] if words else ""
        has_heading_term = (
            first_word in HEADING_KEYWORDS
            or any(word in HEADING_KEYWORDS for word in words[:3])
            or DOC_TITLE_HINT_RE.search(text)
            or STRONG_DOC_TITLE_HINT_RE.search(text)
        )
        if has_heading_term and not looks_like_standard_header_noise(text):
            filtered.append(heading)
    for idx, heading in enumerate(filtered, start=1):
        heading.id = idx
    attach_breadcrumbs(filtered)
    return filtered


def filter_obvious_non_heading_events(headings: list[HeadingEvent]) -> list[HeadingEvent]:
    filtered = [
        heading for heading in headings
        if not (
            has_long_blank_run(heading.text)
            or looks_like_bank_or_contact_field(heading.text)
            or looks_like_datasheet_value_line(heading.text)
            or looks_like_clause_table_header(heading.text)
            or looks_like_part_standard_reference(heading.text)
            or looks_like_standard_table_row(heading.text)
            or looks_like_symbol_or_code_fragment(heading.text)
        )
    ]
    for idx, heading in enumerate(filtered, start=1):
        heading.id = idx
    attach_breadcrumbs(filtered)
    return filtered


def toc_entry_number(entry: TocEntry) -> str:
    return re.sub(r"\s*\.\s*", ".", normalize_space(entry.number)).lower()


def nest_decimal_headings_under_preamble(headings: list[HeadingEvent], root: str = "") -> None:
    """Keep numbered clauses under a preceding attachment/spec title when present."""
    ordered = sorted(headings, key=lambda h: (h.page_num, h.top, h.left))
    first_decimal_idx = next((idx for idx, h in enumerate(ordered) if h.scheme == "decimal"), None)
    if first_decimal_idx is None or first_decimal_idx == 0:
        return
    preamble = None
    for heading in reversed(ordered[:first_decimal_idx]):
        if heading.level == 1 and heading.scheme in {"appendix", "section"}:
            preamble = heading
            break
        if heading.level == 1 and re.search(r"\b(?:attachment|annex|appendix|specification|procedure|standard)\b", heading.text, re.IGNORECASE):
            preamble = heading
            break
    if not preamble:
        return
    for heading in ordered[first_decimal_idx:]:
        if heading.scheme == "decimal":
            heading.level = min(heading.level + 1, 6)
    attach_breadcrumbs(ordered, root)


def anchor_mixed_document_decimal_headings(headings: list[HeadingEvent], root: str = "") -> None:
    """Nest decimal clauses under nearby unnumbered subdocument titles in stitched packs."""
    ordered = sorted(headings, key=lambda h: (h.page_num, h.top, h.left))
    active_anchor: Optional[HeadingEvent] = None

    def is_anchor(heading: HeadingEvent) -> bool:
        text = normalize_space(heading.text)
        if heading.scheme in {"appendix", "section"}:
            return True
        if heading.scheme == "decimal" or heading.level > 2:
            return False
        return bool(SUBDOCUMENT_ANCHOR_RE.search(text))

    for heading in ordered:
        if heading.scheme != "decimal":
            if (
                active_anchor
                and active_anchor.text.upper().startswith("ATTACHMENT")
                and heading.page_num > active_anchor.page_num
                and SUBDOCUMENT_ANCHOR_RE.search(normalize_space(heading.text))
            ):
                heading.level = 1
            if is_anchor(heading):
                active_anchor = heading
            continue
        if not active_anchor:
            continue
        if heading.page_num - active_anchor.page_num > 3:
            active_anchor = None
            continue
        decimal_depth = max(heading.level, 1)
        if heading.numbering:
            parts = heading.numbering.split(".")
            decimal_depth = len(parts)
            if len(parts) > 1 and parts[-1] == "0":
                decimal_depth -= 1
            decimal_depth = max(decimal_depth, 1)
        heading.level = min(max(active_anchor.level + decimal_depth, active_anchor.level + 1), 6)
        if "mixed_doc_anchor" not in heading.reasons:
            heading.reasons.append("mixed_doc_anchor")

    attach_breadcrumbs(ordered, root)


# ---------------------------------------------------------------------------
# Chunk and page construction
# ---------------------------------------------------------------------------

def heading_lookup(headings: list[HeadingEvent]) -> dict[tuple[int, int], HeadingEvent]:
    return {(h.page_num, h.line_index): h for h in headings}


def split_long_text(text: str, chunk_size: int = 240, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_chunks_and_pages(
    page_lines: dict[int, list[LogicalLine]],
    headings: list[HeadingEvent],
    document_context: DocumentContext,
) -> tuple[list[ParagraphChunk], list[dict]]:
    hlookup = heading_lookup(headings)
    stack: list[tuple[int, str, HeadingEvent]] = []
    chunks: list[ParagraphChunk] = []
    pages: list[dict] = []
    paragraph: list[LogicalLine] = []
    root = document_context.breadcrumb_root

    def current_breadcrumb() -> str:
        parts = [root] if root else []
        parts.extend(text for _level, text, _heading in stack)
        return " > ".join(part for part in parts if part)

    def with_local_context(breadcrumb: str, local_context: str) -> str:
        local_context = normalize_space(local_context)
        if not local_context:
            return breadcrumb
        if comparable_title(local_context) and comparable_title(local_context) in comparable_title(breadcrumb):
            return breadcrumb
        parts = [breadcrumb] if breadcrumb else []
        parts.append(local_context)
        return " > ".join(part for part in parts if part)

    def current_confidence() -> float:
        if not stack:
            return 0.0
        return min(item[2].confidence for item in stack)

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = "\n".join(line.text for line in paragraph).strip()
        page_start = paragraph[0].page_num
        page_end = paragraph[-1].page_num
        local_contexts = [line.local_context for line in paragraph if line.local_context]
        local_context = local_contexts[-1] if local_contexts else ""
        breadcrumb = with_local_context(current_breadcrumb(), local_context)
        confidence = current_confidence()
        roles = [line.role for line in paragraph if line.role not in {"body", "caption"}]
        chunk_type = Counter(roles).most_common(1)[0][0] if roles else "text"
        for piece in split_long_text(text):
            chunks.append(
                ParagraphChunk(
                    chunk_id=len(chunks) + 1,
                    page_start=page_start,
                    page_end=page_end,
                    breadcrumb=breadcrumb,
                    text=piece,
                    chunk_type=chunk_type,
                    heading_id=stack[-1][2].id if stack else None,
                    confidence=round(confidence, 3),
                )
            )
        paragraph = []

    for page_num in sorted(page_lines):
        page_content: list[str] = []
        page_breadcrumb = current_breadcrumb()
        active_local_context = ""
        for line in page_lines[page_num]:
            if line.is_running or is_page_number(line.text):
                continue
            heading = hlookup.get((line.page_num, line.line_index))
            if heading:
                flush_paragraph()
                active_local_context = ""
                while stack and stack[-1][0] >= heading.level:
                    stack.pop()
                stack.append((heading.level, heading.text, heading))
                chunks.append(
                    ParagraphChunk(
                        chunk_id=len(chunks) + 1,
                        page_start=heading.page_num,
                        page_end=heading.page_num,
                        breadcrumb=current_breadcrumb(),
                        text=heading.text,
                        chunk_type="heading",
                        heading_id=heading.id,
                        confidence=heading.confidence,
                    )
                )
                page_breadcrumb = current_breadcrumb()
                page_content.append(line.text)
                continue

            if line.role == "caption" and line.local_context:
                flush_paragraph()
                active_local_context = line.local_context
            elif line.role == "drawing" and line.local_context:
                active_local_context = line.local_context
            elif active_local_context and not line.local_context and line.role in {"body", "table", "form"}:
                line.local_context = active_local_context

            if paragraph and line.local_context.startswith("Clause "):
                flush_paragraph()

            # Infer paragraph boundaries from layout gaps and table/heading shifts.
            if paragraph and line.gap_above >= max(10.0, line.height * 1.8):
                flush_paragraph()
            paragraph.append(line)
            page_content.append(line.text)
            page_breadcrumb = current_breadcrumb()

        flush_paragraph()
        pages.append({
            "page_num": page_num,
            "content": "\n".join(page_content).strip(),
            "breadcrumb": page_breadcrumb,
        })

    return chunks, pages


def build_sections(headings: list[HeadingEvent], total_pages: int, document_context: DocumentContext) -> list[dict]:
    sections = []
    for idx, heading in enumerate(sorted(headings, key=lambda h: (h.page_num, h.top, h.left))):
        sections.append({
            "heading": heading.text,
            "level": heading.level,
            "page_num": heading.page_num,
            "page_start": heading.page_num,
            "breadcrumb": heading.breadcrumb,
            "seq": idx,
            "confidence": heading.confidence,
            "score": heading.score,
            "reasons": heading.reasons,
            "warnings": heading.warnings,
            "document_context": asdict(document_context),
        })
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section["page_end"] = max(section["page_start"], sections[i + 1]["page_start"] - 1)
        else:
            section["page_end"] = max(section["page_start"], total_pages)
    return sections


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def mark_running_lines(page_lines: dict[int, list[LogicalLine]], pages: list[PageInfo]) -> set[str]:
    running = detect_running_lines(page_lines, pages)
    for lines in page_lines.values():
        for line in lines:
            line.is_running = line.repeated_key in running
    return running


def run_fast_pipeline(
    pdf_path: Path,
    output_dir: Path,
    keep_work: bool = False,
    ocr: bool = False,
    ocr_jobs: int = 4,
) -> DocumentArtifacts:
    source_pdf = pdf_path.resolve()
    pdf_path = source_pdf
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    with tempfile.TemporaryDirectory(prefix="fast-pdf-pipeline-") as tmp:
        work_dir = Path(tmp)
        ocr_status = "not_requested"
        if ocr:
            pdf_path, ocr_status = run_ocrmypdf_skip_text(pdf_path, work_dir, jobs=ocr_jobs)
            if ocr_status != "ocrmypdf_skip_text":
                warnings.append(ocr_status)

        layout_pages, text_extractor = run_pdftotext(pdf_path)
        xml_path, xml_extractor = run_pdftohtml_xml(pdf_path, work_dir)

        if xml_path:
            pages, nodes = parse_pdftohtml_xml(xml_path)
            extractor_name = xml_extractor
        else:
            pages, nodes, fallback_layout, extractor_name = extract_with_fitz_fallback(pdf_path)
            if not layout_pages:
                layout_pages = fallback_layout
            warnings.append(xml_extractor)

        if not pages:
            raise RuntimeError(f"No pages extracted from {pdf_path}")

        page_lines = build_logical_lines(pages, nodes)
        running_lines = mark_running_lines(page_lines, pages)
        all_lines = [line for lines in page_lines.values() for line in lines]
        body_size = estimate_body_font_size(all_lines)
        left_margin = detect_left_margin(all_lines)
        toc_entries = parse_toc_entries(page_lines, left_margin)
        classify_line_roles(page_lines, pages, body_size, left_margin, toc_entries)
        provisional_title = document_title_from_lines(page_lines, body_size) or source_pdf.stem.replace("_", " ").replace("-", " ")
        headings, heading_warnings = detect_headings(page_lines, pages, body_size, left_margin, toc_entries)
        headings = filter_document_title_headings(headings, provisional_title)
        document_context = infer_document_context(source_pdf, page_lines, headings, body_size)
        headings = filter_document_title_headings(headings, document_context.title)
        headings = filter_unstructured_headings_after_numbering(headings)
        headings = filter_obvious_non_heading_events(headings)
        attach_breadcrumbs(headings, document_context.breadcrumb_root)
        nest_decimal_headings_under_preamble(headings, document_context.breadcrumb_root)
        anchor_mixed_document_decimal_headings(headings, document_context.breadcrumb_root)
        warnings.extend(heading_warnings)
        chunks, page_records = build_chunks_and_pages(page_lines, headings, document_context)
        sections = build_sections(headings, len(pages), document_context)

        extractor = {
            "ocr": ocr_status,
            "layout_text": text_extractor,
            "layout_page_count": len(layout_pages),
            "geometry": extractor_name,
            "running_line_count": len(running_lines),
            "line_role_counts": dict(Counter(line.role for line in all_lines)),
        }

        artifacts = DocumentArtifacts(
            source_pdf=str(source_pdf),
            document_title=document_context.title,
            document_context=document_context,
            page_count=len(pages),
            body_font_size=body_size,
            headings=headings,
            toc_entries=toc_entries,
            chunks=chunks,
            pages=page_records,
            sections=sections,
            warnings=warnings,
            extractor=extractor,
        )

        write_artifacts(artifacts, output_dir)

        if keep_work:
            work_copy = output_dir / "_work"
            work_copy.mkdir(exist_ok=True)
            if xml_path and xml_path.exists():
                shutil.copy2(xml_path, work_copy / xml_path.name)

        return artifacts


def write_artifacts(artifacts: DocumentArtifacts, output_dir: Path) -> None:
    write_json(output_dir / "headings.json", [asdict(item) for item in artifacts.headings])
    write_json(output_dir / "toc_entries.json", [asdict(item) for item in artifacts.toc_entries])
    write_json(output_dir / "sections.json", artifacts.sections)
    write_json(output_dir / "pages.json", artifacts.pages)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in artifacts.chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    report = {
        "source_pdf": artifacts.source_pdf,
        "document_title": artifacts.document_title,
        "document_context": asdict(artifacts.document_context),
        "page_count": artifacts.page_count,
        "body_font_size": artifacts.body_font_size,
        "heading_count": len(artifacts.headings),
        "toc_entry_count": len(artifacts.toc_entries),
        "chunk_count": len(artifacts.chunks),
        "warnings": artifacts.warnings,
        "extractor": artifacts.extractor,
    }
    write_json(output_dir / "report.json", report)
    (output_dir / "preview.md").write_text(render_preview(artifacts), encoding="utf-8")


def render_preview(artifacts: DocumentArtifacts) -> str:
    lines = [
        f"# {artifacts.document_title}",
        "",
        f"Context: `{artifacts.document_context.breadcrumb_root}`",
        "",
        f"Source: `{artifacts.source_pdf}`",
        f"Pages: {artifacts.page_count}",
        f"Headings: {len(artifacts.headings)}",
        f"Chunks: {len(artifacts.chunks)}",
        "",
        "## Headings",
        "",
    ]
    for heading in artifacts.headings:
        indent = "  " * max(heading.level - 1, 0)
        warn = " ⚠" if heading.warnings else ""
        lines.append(
            f"- p{heading.page_num} L{heading.level} "
            f"({heading.confidence:.2f}) {indent}{heading.text}{warn}"
        )
    if not artifacts.headings:
        lines.append("- No headings detected")
    lines.extend(["", "## Chunk Samples", ""])
    for chunk in artifacts.chunks[:30]:
        sample = normalize_space(chunk.text)[:260]
        lines.append(f"### C{chunk.chunk_id} p{chunk.page_start}-{chunk.page_end} [{chunk.chunk_type}]")
        lines.append("")
        lines.append(f"_Breadcrumb: {chunk.breadcrumb or '(none)'}_")
        lines.append("")
        lines.append(sample)
        lines.append("")
    if artifacts.warnings:
        lines.extend(["## Warnings", ""])
        for warning in artifacts.warnings[:100]:
            lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"


def page_count(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore
    except ImportError:
        return 0
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def run_with_optional_split(args) -> None:
    pdf_path = Path(args.pdf).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.split:
        artifacts = run_fast_pipeline(
            pdf_path,
            output_dir,
            keep_work=args.keep_work,
            ocr=args.ocr,
            ocr_jobs=args.ocr_jobs,
        )
        print_summary(artifacts, output_dir)
        return

    try:
        from splitter import PDFSplitter
    except ImportError as exc:
        raise RuntimeError("Split mode requires splitter.py dependencies") from exc

    split_dir = output_dir / "_split_parts"
    split_dir.mkdir(exist_ok=True)
    splitter = PDFSplitter(
        pdf_path=str(pdf_path),
        output_dir=str(split_dir),
        threshold=args.split_threshold,
        min_doc_pages=args.min_doc_pages,
    )
    report = splitter.run()
    segments = report.get("segments", [])
    part_paths = sorted(split_dir.glob("*.pdf"))
    if len(segments) <= 1 or not part_paths:
        artifacts = run_fast_pipeline(
            pdf_path,
            output_dir / "single",
            keep_work=args.keep_work,
            ocr=args.ocr,
            ocr_jobs=args.ocr_jobs,
        )
        print_summary(artifacts, output_dir / "single")
        return

    manifest = {
        "source_pdf": str(pdf_path),
        "split_report": report,
        "parts": [],
    }
    for idx, part in enumerate(part_paths, start=1):
        part_out = output_dir / f"part_{idx:03d}"
        artifacts = run_fast_pipeline(
            part,
            part_out,
            keep_work=args.keep_work,
            ocr=args.ocr,
            ocr_jobs=args.ocr_jobs,
        )
        manifest["parts"].append({
            "part": idx,
            "pdf": str(part),
            "output_dir": str(part_out),
            "page_count": artifacts.page_count,
            "heading_count": len(artifacts.headings),
            "chunk_count": len(artifacts.chunks),
            "document_title": artifacts.document_title,
            "document_context": asdict(artifacts.document_context),
        })
        print_summary(artifacts, part_out)
    write_json(output_dir / "manifest.json", manifest)


def print_summary(artifacts: DocumentArtifacts, output_dir: Path) -> None:
    print(
        f"{Path(artifacts.source_pdf).name}: "
        f"{artifacts.page_count} pages, "
        f"{len(artifacts.headings)} headings, "
        f"{len(artifacts.chunks)} chunks -> {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast CPU-first PDF heading/breadcrumb pipeline.")
    parser.add_argument("pdf", help="Input PDF")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory for output artifacts")
    parser.add_argument("--split", action="store_true", help="Run existing PDFSplitter first")
    parser.add_argument("--split-threshold", type=float, default=3.0, help="PDFSplitter score threshold")
    parser.add_argument("--min-doc-pages", type=int, default=4, help="Minimum pages per split document")
    parser.add_argument("--keep-work", action="store_true", help="Keep extracted intermediate XML under output")
    parser.add_argument("--ocr", action="store_true", help="Run ocrmypdf --skip-text before extraction")
    parser.add_argument("--ocr-jobs", type=int, default=4, help="ocrmypdf worker count")
    args = parser.parse_args()
    run_with_optional_split(args)


if __name__ == "__main__":
    main()
