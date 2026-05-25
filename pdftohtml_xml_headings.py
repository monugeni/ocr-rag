#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


DECIMAL_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:\.)?\s*(.+)$")
ALPHA_HEADING_RE = re.compile(r"^([A-Za-z])(?:\.)\s+(.+)$")
ROMAN_HEADING_RE = re.compile(r"^([IVXLCMivxlcm]+)(?:\.)\s+(.+)$")
PAGE_NUM_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)
DESCRIPTION_ROW_RE = re.compile(r"^DESCRIPTION\s*=>", re.IGNORECASE)
SECTION_LABEL_RE = re.compile(
    r"^(?:section|part|volume|annex(?:ure)?|appendix|attachment|enclosure|"
    r"exhibit|schedule|table)\s*[-:\s]*[A-Za-z0-9ivxlc]+",
    re.IGNORECASE,
)


@dataclass
class FontSpec:
    font_id: str
    size: float
    family: str
    color: str


@dataclass
class TextNode:
    page_num: int
    top: float
    left: float
    width: float
    height: float
    font_id: str
    font_size: float
    font_family: str
    raw_text: str
    text: str
    bold: bool

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2


@dataclass
class HeadingCandidate:
    page_num: int
    level: int
    score: float
    text: str
    breadcrumb: str
    top: float
    left: float
    max_font_size: float
    centered: bool
    reasons: list[str] = field(default_factory=list)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_comparison(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"[_\-–—]+", " ", text)
    return normalize_space(text)


def iter_text_content(elem: ET.Element) -> str:
    return normalize_space("".join(elem.itertext()))


def is_centered(node: TextNode, page_width: float) -> bool:
    return abs(node.center_x - page_width / 2) <= page_width * 0.07


def load_xml_root(xml_path: Path):
    try:
        tree = ET.parse(xml_path)
        return tree.getroot()
    except ET.ParseError:
        try:
            from lxml import etree as LET  # type: ignore
        except ImportError as exc:
            raise

        parser = LET.XMLParser(recover=True)
        tree = LET.parse(str(xml_path), parser)
        return tree.getroot()


def parse_xml(xml_path: Path) -> tuple[list[dict], list[TextNode]]:
    root = load_xml_root(xml_path)
    pages: list[dict] = []
    nodes: list[TextNode] = []
    known_fonts: dict[str, FontSpec] = {}

    for page in root.findall("page"):
        page_num = int(page.attrib["number"])
        page_width = float(page.attrib["width"])
        page_height = float(page.attrib["height"])
        fonts: dict[str, FontSpec] = dict(known_fonts)
        for font in page.findall("fontspec"):
            font_spec = FontSpec(
                font_id=font.attrib["id"],
                size=float(font.attrib["size"]),
                family=font.attrib.get("family", ""),
                color=font.attrib.get("color", ""),
            )
            fonts[font.attrib["id"]] = font_spec
            known_fonts[font.attrib["id"]] = font_spec

        pages.append(
            {
                "page_num": page_num,
                "width": page_width,
                "height": page_height,
                "fonts": fonts,
            }
        )

        for text_elem in page.findall("text"):
            text = iter_text_content(text_elem)
            if not text:
                continue
            font_id = text_elem.attrib["font"]
            font = fonts.get(font_id, FontSpec(font_id, 0.0, "", ""))
            nodes.append(
                TextNode(
                    page_num=page_num,
                    top=float(text_elem.attrib["top"]),
                    left=float(text_elem.attrib["left"]),
                    width=float(text_elem.attrib["width"]),
                    height=float(text_elem.attrib["height"]),
                    font_id=font_id,
                    font_size=font.size,
                    font_family=font.family,
                    raw_text="".join(text_elem.itertext()),
                    text=text,
                    bold=any(child.tag == "b" for child in text_elem.iter()),
                )
            )

    return pages, nodes


def estimate_body_font_size(nodes: list[TextNode]) -> float:
    weighted_sizes: list[float] = []
    for node in nodes:
        if PAGE_NUM_RE.match(node.text):
            continue
        if len(node.text) < 3:
            continue
        weight = min(max(len(node.text), 1), 80)
        weighted_sizes.extend([node.font_size] * weight)
    if not weighted_sizes:
        return 12.0
    counts = Counter(weighted_sizes)
    return counts.most_common(1)[0][0]


def numbering_depth(text: str) -> int:
    m = DECIMAL_HEADING_RE.match(text)
    if m:
        return len(m.group(1).split("."))
    if ALPHA_HEADING_RE.match(text):
        return 2
    if ROMAN_HEADING_RE.match(text):
        return 2
    return 0


def is_all_caps_short(text: str) -> bool:
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    return text == text.upper() and len(text.split()) <= 10


def looks_like_sentence(text: str) -> bool:
    if len(text) > 120:
        return True
    return bool(re.search(r"[a-z].*,", text)) and not text.endswith(":")


def looks_like_table_header(text: str) -> bool:
    lower = text.lower()
    if "description" in lower and any(
        keyword in lower
        for keyword in ("uom", "location", "quantity", "schedule", "gstin", "hsn")
    ):
        return True
    tokens = [token.strip("()[],:.") for token in text.split()]
    coded = 0
    for token in tokens:
        upper = token.upper()
        if re.fullmatch(r"[A-Z]{1,3}-?\d+[A-Z]?", upper):
            coded += 1
        elif upper in {"DESCRIPTION", "UOM", "GSTIN", "HSN", "QTY"}:
            coded += 1
    return lower.startswith("description") and coded >= 3


def is_value_like(text: str) -> bool:
    compact = text.strip()
    if re.fullmatch(r"[\d:\-./ ]+", compact):
        return True
    if re.fullmatch(r"[A-Z0-9]{6,}", compact):
        return True
    return compact in {
        "Limited",
        "Domestic",
        "Two Bid",
        "Applicable",
        "Not Applicable",
        "NA",
        "INR",
        "Yes",
        "No",
    }


def row_density_map(page_nodes: list[TextNode]) -> dict[int, int]:
    rounded_rows = Counter(round(node.top / 6) for node in page_nodes)
    return {round(node.top / 6): rounded_rows[round(node.top / 6)] for node in page_nodes}


def build_logical_lines(page_nodes: list[TextNode]) -> list[TextNode]:
    if not page_nodes:
        return []

    rows: list[list[TextNode]] = []
    for node in sorted(page_nodes, key=lambda item: (item.top, item.left)):
        if not rows:
            rows.append([node])
            continue
        row_top = statistics.median(item.top for item in rows[-1])
        if abs(node.top - row_top) <= 3:
            rows[-1].append(node)
        else:
            rows.append([node])

    logical_lines: list[TextNode] = []
    for row in rows:
        pieces = sorted(row, key=lambda item: item.left)
        text = normalize_space(" ".join(piece.text for piece in pieces))
        if not text:
            continue
        left = min(piece.left for piece in pieces)
        right = max(piece.left + piece.width for piece in pieces)
        tallest = max(pieces, key=lambda item: (item.font_size, item.height))
        logical_lines.append(
            TextNode(
                page_num=pieces[0].page_num,
                top=min(piece.top for piece in pieces),
                left=left,
                width=right - left,
                height=max(piece.height for piece in pieces),
                font_id=tallest.font_id,
                font_size=tallest.font_size,
                font_family=tallest.font_family,
                raw_text=" ".join(piece.raw_text for piece in pieces),
                text=text,
                bold=any(piece.bold for piece in pieces),
            )
        )

    return logical_lines


def detect_running_band_lines(page_map: dict[int, list[TextNode]], total_pages: int) -> set[str]:
    header_candidates = Counter()
    footer_candidates = Counter()

    for page_nodes in page_map.values():
        logical_lines = build_logical_lines(page_nodes)
        if not logical_lines:
            continue
        top_lines = logical_lines[:3]
        bottom_lines = logical_lines[-3:]
        for line in top_lines:
            normalized = normalize_for_comparison(line.text)
            if len(normalized) >= 8:
                header_candidates[normalized] += 1
        for line in bottom_lines:
            normalized = normalize_for_comparison(line.text)
            if len(normalized) >= 8:
                footer_candidates[normalized] += 1

    threshold = max(2, int(total_pages * 0.6))
    running = {
        text for text, count in header_candidates.items() if count >= threshold
    }
    running.update(
        text for text, count in footer_candidates.items() if count >= threshold
    )
    return running


def line_score(
    node: TextNode,
    body_font_size: float,
    page_width: float,
    page_height: float,
    repeated_lines: set[str],
    running_band_lines: set[str],
    left_margin: float,
    row_density: int,
) -> tuple[float, list[str]]:
    text = node.text
    reasons: list[str] = []
    score = 0.0
    centered = is_centered(node, page_width)
    words = text.split()
    num_depth = numbering_depth(text)
    normalized = normalize_for_comparison(text)

    if PAGE_NUM_RE.match(text):
        return -10.0, ["page_number"]
    if DESCRIPTION_ROW_RE.match(text):
        return -4.0, ["description_row"]
    if normalized in running_band_lines:
        return -8.0, ["running_header_footer"]
    if is_value_like(text):
        return -5.0, ["value_like"]
    if looks_like_table_header(text):
        return -6.0, ["table_header"]

    font_delta = node.font_size - body_font_size
    if font_delta >= 6:
        score += 5
        reasons.append("very_large_font")
    elif font_delta >= 3:
        score += 3
        reasons.append("large_font")
    elif font_delta >= 1.5:
        score += 1.5
        reasons.append("slightly_large_font")

    if node.bold and len(text) <= 120:
        score += 1.5
        reasons.append("bold")

    if num_depth and len(text) <= 80 and not looks_like_sentence(text) and (
        text.endswith(":") or len(words) <= 10 or font_delta >= 0.5
    ):
        score += 3.5
        reasons.append(f"numbering_depth_{num_depth}")

    if is_all_caps_short(text) and len(words) >= 2:
        score += 2
        reasons.append("all_caps_short")

    if SECTION_LABEL_RE.match(text) and len(words) <= 18:
        score += 2.5
        reasons.append("section_label")

    if text.endswith(":") and len(words) <= 18:
        score += 1.5
        reasons.append("trailing_colon")

    if centered and len(words) <= 18 and len(text) <= 140:
        score += 1.5
        reasons.append("centered")
        if len(words) <= 6:
            score += 0.75
            reasons.append("short_centered")

    if node.left <= left_margin + 24 and len(words) <= 18 and (text.endswith(":") or num_depth):
        score += 1
        reasons.append("left_aligned")

    if text in repeated_lines and node.top <= page_height * 0.14 and not centered:
        score -= 3
        reasons.append("repeated_top_band")
    elif text in repeated_lines and not centered:
        score -= 1.5
        reasons.append("repeated_line")

    if row_density >= 5 and not centered and font_delta < 2.5:
        score -= 4
        reasons.append("dense_row")

    if len(words) <= 3 and not centered and not text.endswith(":") and font_delta < 2:
        score -= 3
        reasons.append("short_field_or_value")

    if not centered and num_depth == 0 and not text.endswith(":") and not is_all_caps_short(text):
        score -= 3
        reasons.append("not_heading_shaped")

    if looks_like_sentence(text):
        score -= 2
        reasons.append("sentence_like")

    if len(text) > 180:
        score -= 3
        reasons.append("very_long")

    return score, reasons


def detect_repeated_lines(nodes: list[TextNode], total_pages: int) -> set[str]:
    counts = Counter(node.text for node in nodes if len(node.text) >= 6)
    threshold = max(3, int(total_pages * 0.3))
    return {text for text, count in counts.items() if count >= threshold}


def group_heading_lines(
    page_nodes: list[TextNode],
    body_font_size: float,
    page_width: float,
    repeated_lines: set[str],
    running_band_lines: set[str],
) -> list[dict]:
    if not page_nodes:
        return []

    logical_lines = build_logical_lines(page_nodes)
    left_margin = min(node.left for node in logical_lines)
    row_density_lookup = row_density_map(page_nodes)
    scored: list[dict] = []
    for node in logical_lines:
        score, reasons = line_score(
            node,
            body_font_size,
            page_width,
            max(node.top + node.height for node in logical_lines),
            repeated_lines,
            running_band_lines,
            left_margin,
            row_density_lookup.get(round(node.top / 6), 1),
        )
        if score < 2.8:
            continue
        scored.append(
            {
                "node": node,
                "score": score,
                "reasons": reasons,
                "centered": is_centered(node, page_width),
            }
        )

    groups: list[list[dict]] = []
    for entry in sorted(scored, key=lambda item: (item["node"].top, item["node"].left)):
        if not groups:
            groups.append([entry])
            continue

        prev = groups[-1][-1]
        gap = entry["node"].top - prev["node"].top
        same_page = entry["node"].page_num == prev["node"].page_num
        similar_left = abs(entry["node"].left - prev["node"].left) <= 40
        both_centered = entry["centered"] and prev["centered"]
        similar_font = abs(entry["node"].font_size - prev["node"].font_size) <= 2.5
        mergeable_gap = (
            gap <= 28 if both_centered else gap <= 10
        )

        centered_mergeable = both_centered and (
            entry["node"].width <= page_width * 0.65 and prev["node"].width <= page_width * 0.65
        )
        if same_page and mergeable_gap and (centered_mergeable or (similar_font and similar_left)):
            groups[-1].append(entry)
        else:
            groups.append([entry])

    grouped: list[dict] = []
    for group in groups:
        text = " ".join(item["node"].text for item in group)
        text = normalize_space(text)
        if not text or len(text) < 3:
            continue
        if PAGE_NUM_RE.match(text):
            continue
        if is_value_like(text):
            continue
        if looks_like_sentence(text) and numbering_depth(text) == 0 and max(item["node"].font_size for item in group) <= body_font_size + 2:
            continue
        if len(text.split()) <= 3 and not text.endswith(":") and not any(item["centered"] for item in group):
            continue

        grouped.append(
            {
                "page_num": group[0]["node"].page_num,
                "text": text,
                "top": min(item["node"].top for item in group),
                "left": min(item["node"].left for item in group),
                "max_font_size": max(item["node"].font_size for item in group),
                "score": round(sum(item["score"] for item in group), 2),
                "centered": any(item["centered"] for item in group),
                "reasons": sorted({reason for item in group for reason in item["reasons"]}),
            }
        )

    return grouped


def assign_levels(groups: list[dict], body_font_size: float) -> list[dict]:
    sizes = sorted({round(group["max_font_size"], 1) for group in groups}, reverse=True)
    size_to_rank = {size: idx + 1 for idx, size in enumerate(sizes)}

    for group in groups:
        depth = numbering_depth(group["text"])
        if depth:
            level = min(depth + 1, 6)
        elif group["centered"] and group["max_font_size"] >= body_font_size + 4:
            level = 1
        else:
            level = min(size_to_rank[round(group["max_font_size"], 1)], 6)
        group["level"] = level
    return groups


def attach_breadcrumbs(groups: list[dict]) -> list[HeadingCandidate]:
    stack: list[tuple[int, str]] = []
    results: list[HeadingCandidate] = []
    for group in sorted(groups, key=lambda item: (item["page_num"], item["top"], item["left"])):
        while stack and stack[-1][0] >= group["level"]:
            stack.pop()
        stack.append((group["level"], group["text"]))
        breadcrumb = " > ".join(text for _, text in stack)
        results.append(
            HeadingCandidate(
                page_num=group["page_num"],
                level=group["level"],
                score=group["score"],
                text=group["text"],
                breadcrumb=breadcrumb,
                top=group["top"],
                left=group["left"],
                max_font_size=group["max_font_size"],
                centered=group["centered"],
                reasons=group["reasons"],
            )
        )
    return results


def extract_headings(xml_path: Path) -> dict:
    pages, nodes = parse_xml(xml_path)
    body_font_size = estimate_body_font_size(nodes)
    repeated_lines = detect_repeated_lines(nodes, len(pages))

    page_map: dict[int, list[TextNode]] = {}
    page_widths = {page["page_num"]: page["width"] for page in pages}
    for node in nodes:
        page_map.setdefault(node.page_num, []).append(node)

    running_band_lines = detect_running_band_lines(page_map, len(pages))

    groups: list[dict] = []
    for page_num, page_nodes in sorted(page_map.items()):
        groups.extend(
            group_heading_lines(
                page_nodes=page_nodes,
                body_font_size=body_font_size,
                page_width=page_widths[page_num],
                repeated_lines=repeated_lines,
                running_band_lines=running_band_lines,
            )
        )

    groups = assign_levels(groups, body_font_size)
    headings = attach_breadcrumbs(groups)

    return {
        "source_xml": str(xml_path),
        "body_font_size": body_font_size,
        "page_count": len(pages),
        "heading_count": len(headings),
        "headings": [asdict(heading) for heading in headings],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract heading candidates from pdftohtml -xml output."
    )
    parser.add_argument("xml_path", help="Path to pdftohtml XML file")
    parser.add_argument(
        "--output-json",
        help="Optional path to write JSON output. Defaults next to input XML.",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml_path).resolve()
    result = extract_headings(xml_path)

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else xml_path.with_suffix(".headings.json")
    )
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {output_json}")
    for heading in result["headings"]:
        print(
            f"p{heading['page_num']:>3}  L{heading['level']}  "
            f"{heading['text']}"
        )


if __name__ == "__main__":
    main()
