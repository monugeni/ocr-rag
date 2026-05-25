#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest import parse_marker_json  # noqa: E402
from pdftohtml_xml_headings import extract_headings  # noqa: E402


@dataclass
class RenderedIngestion:
    name: str
    pages: list[dict]
    sections: list[dict]
    markdown: str


def split_pdftotext_pages(pdf_path: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    text = result.stdout
    if not text:
        return []
    pages = text.split("\f")
    return [page.rstrip() for page in pages if page.strip()]


def build_xml_pages(pdf_path: Path, xml_path: Path) -> tuple[list[dict], list[dict]]:
    headings_data = extract_headings(xml_path)
    text_pages = split_pdftotext_pages(pdf_path)
    headings = headings_data["headings"]

    current_breadcrumb = ""
    current_section_id = None
    next_heading_idx = 0
    sections: list[dict] = []
    pages: list[dict] = []

    for idx, heading in enumerate(headings, start=1):
        sections.append(
            {
                "heading": heading["text"],
                "level": heading["level"],
                "page_num": heading["page_num"],
                "breadcrumb": heading["breadcrumb"],
                "section_id": idx,
            }
        )

    for page_num, page_text in enumerate(text_pages, start=1):
        while next_heading_idx < len(headings) and headings[next_heading_idx]["page_num"] <= page_num:
            current_breadcrumb = headings[next_heading_idx]["breadcrumb"]
            current_section_id = next_heading_idx + 1
            next_heading_idx += 1

        pages.append(
            {
                "page_num": page_num,
                "content": page_text.strip(),
                "breadcrumb": current_breadcrumb,
                "section_id": current_section_id,
            }
        )

    return pages, sections


def render_ingestion(name: str, title: str, pages: list[dict], sections: list[dict]) -> RenderedIngestion:
    lines = [f"# {title}", "", f"## Source", "", name, "", "## Sections", ""]
    for section in sections:
        lines.append(
            f"- Page {section['page_num']}: "
            f"{'  ' * max(section['level'] - 1, 0)}{section['heading']}"
        )

    if not sections:
        lines.append("- No sections detected")

    lines.extend(["", "## Pages", ""])
    for page in pages:
        lines.extend(
            [
                f"### Page {page['page_num']}",
                "",
                f"_Breadcrumb: {page.get('breadcrumb', '') or '(none)'}_",
                "",
                page["content"].strip() or "(empty)",
                "",
            ]
        )

    return RenderedIngestion(
        name=name,
        pages=pages,
        sections=sections,
        markdown="\n".join(lines).rstrip() + "\n",
    )


def marker_to_rendered(pdf_path: Path, marker_json_path: Path) -> RenderedIngestion:
    pages, sections = parse_marker_json(str(marker_json_path), str(pdf_path))
    for idx, section in enumerate(sections, start=1):
        section["section_id"] = idx
    current_section_idx = 0
    for page in pages:
        while current_section_idx + 1 < len(sections) and sections[current_section_idx + 1]["page_num"] <= page["page_num"]:
            current_section_idx += 1
        page["section_id"] = sections[current_section_idx]["section_id"] if sections else None
    return render_ingestion("existing_marker_ingestion", pdf_path.name, pages, sections)


def xml_to_rendered(pdf_path: Path, xml_path: Path) -> RenderedIngestion:
    pages, sections = build_xml_pages(pdf_path, xml_path)
    return render_ingestion("new_xml_ingestion", pdf_path.name, pages, sections)


def summarize(marker: RenderedIngestion, xml: RenderedIngestion) -> dict:
    page_count = max(len(marker.pages), len(xml.pages))
    page_samples: list[dict] = []
    for page_num in range(1, page_count + 1):
        marker_page = next((p for p in marker.pages if p["page_num"] == page_num), None)
        xml_page = next((p for p in xml.pages if p["page_num"] == page_num), None)
        if not marker_page and not xml_page:
            continue
        page_samples.append(
            {
                "page_num": page_num,
                "marker_breadcrumb": marker_page["breadcrumb"] if marker_page else "",
                "xml_breadcrumb": xml_page["breadcrumb"] if xml_page else "",
                "marker_chars": len(marker_page["content"]) if marker_page else 0,
                "xml_chars": len(xml_page["content"]) if xml_page else 0,
            }
        )

    return {
        "marker": {
            "page_count": len(marker.pages),
            "section_count": len(marker.sections),
            "char_count": sum(len(page["content"]) for page in marker.pages),
        },
        "xml": {
            "page_count": len(xml.pages),
            "section_count": len(xml.sections),
            "char_count": sum(len(page["content"]) for page in xml.pages),
        },
        "page_samples": page_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Marker ingestion with XML+pdftotext ingestion.")
    parser.add_argument("--pdf", required=True, help="Source PDF path")
    parser.add_argument("--marker-json", required=True, help="Marker JSON output path")
    parser.add_argument("--xml", required=True, help="pdftohtml XML output path")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison artifacts")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    marker_json_path = Path(args.marker_json).resolve()
    xml_path = Path(args.xml).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = marker_to_rendered(pdf_path, marker_json_path)
    xml = xml_to_rendered(pdf_path, xml_path)
    summary = summarize(marker, xml)

    (output_dir / "existing_marker_ingestion.md").write_text(marker.markdown, encoding="utf-8")
    (output_dir / "new_xml_ingestion.md").write_text(xml.markdown, encoding="utf-8")
    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
