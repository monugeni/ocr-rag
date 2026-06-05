"""Map Findings → pdf-annotator annotations on the submitted PDF.

Uses the pdf-annotator library directly (text-anchored highlight preferred,
coord rectangle for drawings / explicit bbox, page-level sticky note as the last
resort). A finding is never dropped just because placement failed.
"""
from __future__ import annotations

from .schema import SEVERITY_COLORS, Finding


def _content(f: Finding) -> str:
    """Vendor-appropriate comment text — NO severity or internal category labels.

    The annotated PDF may be sent to the vendor as-is, so the comment reads as a clean
    review remark plus the governing reference. Severity/category are stored separately
    (DB + web UI) only.
    """
    ref = f.citation or {}
    ref_bits = [b for b in (ref.get("doc_title"), ref.get("heading")) if b]
    suffix = f"  (ref: {' — '.join(ref_bits)})" if ref_bits else ""
    body = (f.detail or "").strip() or (f.title or "").strip()
    return f"{body}{suffix}"


def annotate_finding(pdf_path: str, finding: Finding) -> tuple[int | None, str]:
    """Write one finding annotation onto ``pdf_path`` in place.

    Returns ``(annotation_xref, method)``. ``method`` is one of
    ``highlight`` | ``rect`` | ``sticky`` | ``failed`` for diagnostics.
    """
    from ..pdfannotator import add_annotations, add_sticky_note, find_text, highlight_text

    color = SEVERITY_COLORS.get(finding.severity, "#888888")
    content = _content(finding)

    # 1. Text-anchored highlight (preferred for native PDFs).
    if finding.anchor_text:
        try:
            res = highlight_text(
                pdf_path,
                query=finding.anchor_text,
                page=finding.page_num,
                kind="highlight",
                color=color,
                content=content,
                author="Checker",
                in_place=True,
            )
            matches = res.get("matches") or []
            if matches:
                return matches[0].get("id"), "highlight"
        except Exception:  # noqa: BLE001 — fall through to next strategy
            pass

    # 2. Coordinate rectangle (drawings, or explicit bbox).
    if finding.bbox:
        try:
            res = add_annotations(
                pdf_path,
                items=[
                    {
                        "type": "rectangle",
                        "page": finding.page_num,
                        "rect": finding.bbox,
                        "content": content,
                        "color": color,
                        "author": "Checker",
                    }
                ],
                in_place=True,
            )
            anns = res.get("annotations") or []
            if anns:
                return anns[0].get("id"), "rect"
        except Exception:  # noqa: BLE001
            pass

    # 3. Page-level sticky note — last resort, never drop the finding.
    try:
        res = add_sticky_note(
            pdf_path,
            page=finding.page_num or 1,
            point=[36, 36],
            text=content,
            color=color,
            author="Checker",
            in_place=True,
        )
        return res.get("id"), "sticky"
    except Exception:  # noqa: BLE001
        return None, "failed"
