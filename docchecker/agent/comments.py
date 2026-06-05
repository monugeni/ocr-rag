"""Read prior review comments from an old commented PDF (revision check).

Comments are PDF annotations. We read them via pdf-annotator's
``list_annotations`` and turn each reviewer comment into a ``CommentStatus``.
In M7 the verdict is a placeholder; M8's agent judges actual incorporation in
the new revision.
"""
from __future__ import annotations

from typing import Any

from .schema import CommentStatus

# CAD noise authors to drop by default.
_NOISE_AUTHORS = {"AutoCAD SHX Text"}


def _referenced_text(doc, page_num: int, rect, ann_type: str | None) -> str:
    """Extract the text a comment marks/points at (its context).

    For text markup (highlight/underline/strikeout/squiggly) the rect covers the
    marked words. For point notes (sticky/callout) the icon rect is tiny, so we read
    a band around it to capture the nearby line(s).
    """
    if not rect:
        return ""
    try:
        import fitz

        page = doc[page_num - 1]
        r = fitz.Rect(rect)
        markup = {"Highlight", "Underline", "StrikeOut", "Squiggly"}

        # Text markup covers the marked words → read straight from the rect.
        if ann_type in markup:
            txt = page.get_text("text", clip=r + (-2, -2, 2, 2)).strip()
            if txt:
                return " ".join(txt.split())[:600]

        # Point notes (sticky/callout) or empty markup → nearest text block(s).
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        scored = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
            if not (btext or "").strip():
                continue
            dx = max(x0 - cx, 0, cx - x1)
            dy = max(y0 - cy, 0, cy - y1)
            scored.append(((dx * dx + dy * dy) ** 0.5, btext.strip()))
        scored.sort(key=lambda s: s[0])
        # Nearest block, plus the next-nearest if very close (multi-line context).
        picked = [t for d, t in scored[:2] if d < 80] or ([scored[0][1]] if scored else [])
        return " ".join(" ".join(picked).split())[:600]
    except Exception:  # noqa: BLE001
        return ""


def read_prior_comments(pdf_path: str) -> list[dict[str, Any]]:
    """Return reviewer comments with text, location, and the context they refer to."""
    from ..pdfannotator import list_annotations

    try:
        anns = list_annotations(pdf_path, exclude_authors=list(_NOISE_AUTHORS))
    except Exception:  # noqa: BLE001 — unreadable / no annots
        return []

    doc = None
    try:
        import fitz

        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        doc = None

    comments = []
    try:
        for a in anns:
            if a.get("author") == "Checker" or a.get("in_reply_to"):
                continue
            content = (a.get("content") or "").strip()
            text = content or f"({a.get('type') or 'markup'} with no note)"
            referenced = (
                _referenced_text(doc, a.get("page"), a.get("rect"), a.get("type"))
                if doc is not None
                else ""
            )
            comments.append(
                {
                    "xref": a.get("id"),
                    "page": a.get("page"),
                    "content": text,
                    "referenced_text": referenced,
                    "has_text": bool(content),
                    "author": a.get("author"),
                    "rect": a.get("rect"),
                    "type": a.get("type"),
                }
            )
    finally:
        if doc is not None:
            doc.close()
    return comments


def stub_comment_statuses(comments: list[dict[str, Any]]) -> list[CommentStatus]:
    """Placeholder verdicts for M7 — extraction proven, judging lands in M8."""
    out = []
    for c in comments:
        out.append(
            CommentStatus(
                prior_comment_ref=str(c.get("xref")) if c.get("xref") is not None else None,
                prior_comment_text=c["content"],
                prior_page=c.get("page"),
                verdict="not_applicable",
                detail="Incorporation check pending (implemented in M8).",
                evidence={},
            )
        )
    return out
