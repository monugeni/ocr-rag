"""
pdf-annot MCP server.

Read and create PDF annotations (highlights, shapes, callouts, sticky notes,
ink, stamps) in the same style Adobe Reader and Acrobat produce, so files
remain interoperable with reviewers using those tools.

Two ways to place annotations
-----------------------------
1. **Text-anchored (preferred for prose / procedures / specs).** Pass a text
   query — `highlight_text("V-anchor at 200 mm")`,
   `add_sticky_note_at_text("Clause 7.4", "needs revision")`,
   `add_callout_at_text("VACCUM", "typo: VACUUM")` — and the server
   locates the text and places the markup directly. No image rendering or
   coordinate guessing needed. This is the right path whenever the PDF
   has a real text layer (most procedures, contracts, specs, native PDFs).

2. **Coordinate-based (CAD drawings, scanned PDFs, fine layout).** Pass
   explicit rects/points in PDF points (or grid units, see below). Use
   `render_page` / `render_region` / `render_tile` to see the page first.

Coordinate system
-----------------
All coordinates use a top-left origin, in PDF points (1 point = 1/72 inch).
A US Letter page is 612 x 792 pt; A4 is 595 x 842 pt. Page numbers are
1-indexed. PyMuPDF handles internal conversion to PDF's bottom-left space.

Worked example. To draw a 1-inch-square rectangle in the top-left corner
of an A4 page (595 x 842 pt), pass `rect=[0, 0, 72, 72]`. The point
`(72, 72)` is one inch in from the top-left. To put a sticky note at the
center of the same page, pass `point=[297, 421]` — the icon's top-left
will land at that point, and the icon is roughly 20 pt square in Acrobat.

Note: a few CAD-exported PDFs apply non-identity transforms or rotations
that make annotation rects appear inverted in `list_annotations`. If the
rendered page disagrees with the rect coordinates you read, render the
page once with `overlay_grid=True` to see the page's actual coordinate
space.

Colors
------
Colors may be supplied as:
- 6-digit hex: "#ff0000" or "ff0000"
- 3-digit hex: "#f00"
- RGB list of floats in [0, 1]: [1.0, 0.0, 0.0]
- RGB list of ints in [0, 255]: [255, 0, 0]

File references
---------------
Tools that take a `file` argument accept either:
- An absolute path: /home/tan/projects/J269/tender.pdf
- A project-scoped reference: J269:tender.pdf or J269:subdir/file.pdf
  (project roots are configured in ~/.config/pdf-annot/config.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get(
    "PDF_ANNOT_CONFIG",
    str(Path.home() / ".config" / "pdf-annot" / "config.json"),
)


def _load_config() -> dict:
    p = Path(CONFIG_PATH)
    if not p.exists():
        return {"projects": {}, "default_author": "Claude"}
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return {"projects": {}, "default_author": "Claude"}


CONFIG = _load_config()
PROJECTS: dict[str, str] = CONFIG.get("projects", {})
DEFAULT_AUTHOR: str = CONFIG.get("default_author", "Claude")


def _resolve_file(file: str) -> Path:
    """Resolve a project-scoped or absolute file reference."""
    # Project-scoped: "J269:relative/path.pdf"
    # Avoid eating Windows drive letters (length-1 prefix before ':')
    if ":" in file and not file.startswith("/") and not (
        len(file) > 1 and file[1] == ":" and file[0].isalpha()
    ):
        project, rel = file.split(":", 1)
        if project not in PROJECTS:
            raise ValueError(
                f"Unknown project '{project}'. Configured: {list(PROJECTS)}"
            )
        path = Path(PROJECTS[project]) / rel
    else:
        path = Path(file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    return path.resolve()


# ---------------------------------------------------------------------------
# Color and geometry helpers
# ---------------------------------------------------------------------------

def _parse_color(c: Any) -> Optional[tuple[float, float, float]]:
    if c is None:
        return None
    if isinstance(c, str):
        s = c.lstrip("#").strip()
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            raise ValueError(f"Invalid hex color: {c}")
        return (
            int(s[0:2], 16) / 255,
            int(s[2:4], 16) / 255,
            int(s[4:6], 16) / 255,
        )
    if isinstance(c, (list, tuple)) and len(c) == 3:
        vals = [float(v) for v in c]
        if max(vals) > 1.0:
            vals = [v / 255 for v in vals]
        return (vals[0], vals[1], vals[2])
    raise ValueError(f"Cannot parse color: {c!r}")


def _color_to_hex(c) -> Optional[str]:
    if not c or len(c) < 3:
        return None
    r, g, b = c[:3]
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# Grid step used by render_page's overlay and by `grid_coords=True` on the
# write tools. Each grid cell is GRID_STEP points wide/tall, so a rect
# given as [2, 3, 5, 6] in grid units expands to [100, 150, 250, 300] pt.
GRID_STEP = 50


def _rect(r: list[float], scale: float = 1.0) -> fitz.Rect:
    if not isinstance(r, (list, tuple)) or len(r) != 4:
        raise ValueError(f"Rect must be [x0, y0, x1, y1], got {r!r}")
    if scale != 1.0:
        return fitz.Rect(r[0] * scale, r[1] * scale,
                         r[2] * scale, r[3] * scale)
    return fitz.Rect(*r)


def _point(p: list[float], scale: float = 1.0) -> fitz.Point:
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        raise ValueError(f"Point must be [x, y], got {p!r}")
    if scale != 1.0:
        return fitz.Point(p[0] * scale, p[1] * scale)
    return fitz.Point(*p)


def _scale_factor(grid_coords: bool) -> float:
    return float(GRID_STEP) if grid_coords else 1.0


# ---------------------------------------------------------------------------
# Save and common annotation setup
# ---------------------------------------------------------------------------

def _open(path: Path) -> fitz.Document:
    return fitz.open(str(path))


def _get_page(doc: fitz.Document, page: int) -> fitz.Page:
    if page < 1 or page > doc.page_count:
        raise ValueError(f"Page {page} out of range (1..{doc.page_count})")
    return doc[page - 1]


def _resolve_write(
    file: str,
    in_place: bool,
    output_path: Optional[str],
) -> tuple[Path, Path, str]:
    """
    Decide where to read from and where to write to.

    Default behavior is to leave the source untouched and write annotations
    to a sibling file named "<source>.annotated.pdf". Subsequent calls
    against the same source detect the existing sibling and append to it
    incrementally so annotations accumulate.

    Returns (read_path, write_path, mode) where mode is "incremental"
    (Acrobat-style append) or "fresh" (clean write).
    """
    src = _resolve_file(file)
    if in_place:
        return src, src, "incremental"
    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        # If the user explicitly targets the source, that *is* in-place.
        if out == src:
            return src, src, "incremental"
        return src, out, "fresh"
    # If the caller already passed the .annotated.pdf working copy back to
    # us (typical: they used `saved_to` from a prior add_*), keep editing
    # it in place instead of nesting `.annotated.annotated.pdf`.
    if src.name.endswith(".annotated.pdf"):
        return src, src, "incremental"
    derived = src.with_name(src.stem + ".annotated.pdf")
    if derived.exists():
        return derived, derived, "incremental"
    return src, derived, "fresh"


def _save(doc: fitz.Document, write_path: Path, mode: str) -> str:
    """
    Persist the document. mode="incremental" preserves existing objects
    (Acrobat-style save); mode="fresh" writes a clean new file.
    """
    if mode == "incremental":
        doc.save(str(write_path), incremental=True,
                 encryption=fitz.PDF_ENCRYPT_KEEP)
    else:
        write_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(write_path))
    return str(write_path)


# ---------------------------------------------------------------------------
# Image rendering with size cap
# ---------------------------------------------------------------------------

# Claude Desktop and other MCP clients reject tool results larger than 1 MB.
# Stay comfortably below that so the result fits with overhead.
MAX_IMAGE_BYTES = 900_000


def _render_image(
    get_pixmap,
    dpi: int,
    max_bytes: int = MAX_IMAGE_BYTES,
    prefer_format: str = "auto",
) -> tuple[Image, dict]:
    """
    Render with auto-downscale + JPEG fallback so the result stays under
    `max_bytes`.

    Strategy at each DPI step:
    1. Try PNG (lossless, best for diagrams/text/grid overlays) unless
       prefer_format="jpeg".
    2. If too big (or prefer_format="jpeg"), try JPEG at decreasing
       quality (much smaller for photographic content / large drawings).
    3. If still too big, drop DPI by 25% and retry.

    Returns (Image, info) where info reports what was actually produced:
      {"actual_dpi", "requested_dpi", "format", "size_bytes", "downscaled"}
    """
    requested = dpi
    current = max(24, dpi)
    last_data: bytes = b""
    last_fmt = "png"
    last_dpi = current
    try_png = prefer_format != "jpeg"
    try_jpeg = prefer_format != "png"
    for _ in range(8):
        pix = get_pixmap(current)
        last_dpi = current
        if try_png:
            png = pix.tobytes("png")
            last_data, last_fmt = png, "png"
            if len(png) <= max_bytes:
                break
        if try_jpeg:
            jpg = None
            for q in (85, 70, 55, 40):
                try:
                    jpg = pix.tobytes("jpeg", jpg_quality=q)
                except Exception:
                    jpg = None
                    break
                last_data, last_fmt = jpg, "jpeg"
                if len(jpg) <= max_bytes:
                    break
            if jpg is not None and len(jpg) <= max_bytes:
                break
        if current <= 24:
            break
        current = max(24, int(current * 0.75))
    info = {
        "actual_dpi": last_dpi,
        "requested_dpi": requested,
        "format": last_fmt,
        "size_bytes": len(last_data),
        "downscaled": last_dpi != requested,
    }
    return Image(data=last_data, format=last_fmt), info


def _set_common(
    annot: fitz.Annot,
    *,
    content: Optional[str] = None,
    author: Optional[str] = None,
    subject: Optional[str] = None,
    color: Any = None,
    fill: Any = None,
    opacity: Optional[float] = None,
    border_width: Optional[float] = None,
    cloud_intensity: Optional[float] = None,
) -> None:
    info = annot.info
    if content is not None:
        info["content"] = content
    if subject is not None:
        info["subject"] = subject
    info["title"] = author or info.get("title") or DEFAULT_AUTHOR
    annot.set_info(info)

    stroke = _parse_color(color)
    fill_c = _parse_color(fill)
    if stroke is not None or fill_c is not None:
        colors = annot.colors or {}
        if stroke is not None:
            colors["stroke"] = stroke
        if fill_c is not None:
            colors["fill"] = fill_c
        annot.set_colors(colors)

    if opacity is not None:
        annot.set_opacity(opacity)

    border: dict = {}
    if border_width is not None:
        border["width"] = border_width
    if cloud_intensity is not None and cloud_intensity > 0:
        # PyMuPDF supports cloudy borders via the "clouds" key; intensity 1-2
        # is typical (matches Acrobat's "Cloudy" style).
        border["clouds"] = cloud_intensity
    if border:
        try:
            annot.set_border(border)
        except Exception:
            pass

    annot.update()


def _annot_to_dict(annot: fitz.Annot) -> dict:
    info = annot.info
    rect = annot.rect
    out: dict[str, Any] = {
        "id": annot.xref,
        "type": annot.type[1] if annot.type else None,
        "page": annot.parent.number + 1,
        "rect": [round(rect.x0, 2), round(rect.y0, 2),
                 round(rect.x1, 2), round(rect.y1, 2)],
        "author": info.get("title") or None,
        "content": info.get("content") or None,
        "subject": info.get("subject") or None,
        "created": info.get("creationDate") or None,
        "modified": info.get("modDate") or None,
        "color": _color_to_hex((annot.colors or {}).get("stroke")),
        "fill": _color_to_hex((annot.colors or {}).get("fill")),
        "opacity": annot.opacity if annot.opacity != -1 else None,
    }
    # In-reply-to (reply chains)
    try:
        irt = annot.parent.parent.xref_get_key(annot.xref, "IRT")
        if irt and irt[0] == "xref":
            out["in_reply_to"] = int(irt[1].split()[0])
    except Exception:
        pass
    # Vertices for polygons/polylines/lines/ink
    try:
        verts = annot.vertices
        if verts:
            out["vertices"] = [[round(v.x, 2), round(v.y, 2)] for v in verts]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("pdf-annot")


# Internal helpers retain their per-type signatures for clarity in tracebacks
# and for dispatch from the unified `annotate` / `modify` / `inspect` tools,
# but are NOT registered as MCP tools to keep the public surface small.
def _internal(fn):
    return fn


# --- Discovery -------------------------------------------------------------

@_internal
def list_projects() -> dict:
    """List configured project folders and the active config path."""
    return {"projects": PROJECTS, "config_path": CONFIG_PATH,
            "default_author": DEFAULT_AUTHOR}


@_internal
def list_pdfs(location: str) -> list[str]:
    """
    List PDFs under a project name or absolute folder path.

    Args:
        location: Project name (e.g. "J269") or absolute path.
    """
    root = Path(PROJECTS[location]) if location in PROJECTS \
        else Path(location).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    return sorted(p.relative_to(root).as_posix()
                  for p in root.rglob("*.pdf"))


@_internal
def page_info(file: str) -> list[dict]:
    """Return page count and dimensions (width, height in points) for a PDF."""
    path = _resolve_file(file)
    doc = _open(path)
    try:
        return [{"page": i + 1,
                 "width": round(doc[i].rect.width, 2),
                 "height": round(doc[i].rect.height, 2)}
                for i in range(doc.page_count)]
    finally:
        doc.close()


@_internal
def render_page(
    file: str,
    page: int,
    dpi: int = 120,
    with_annotations: bool = True,
    overlay_grid: bool = False,
    max_bytes: int = MAX_IMAGE_BYTES,
    format: str = "auto",
) -> list:
    """
    Render a PDF page as a PNG image so you can see what's on it before
    placing annotations. This is the visual counterpart to list_annotations.

    Use this whenever you need to know where things are on the page, where
    to point a callout, or to verify your annotations after creating them.

    Coordinate system is DPI-agnostic. Both the gridlines and the labels
    are anchored in PDF point space (1 pt = 1/72 inch, page-relative), so
    the same coordinate values can be used at any render DPI — only the
    pixel size of the image changes.

    Args:
        dpi: Render resolution. Affects only how big the image is in
            pixels; coordinates are unaffected. 100-120 suits most pages,
            200+ is useful for dense drawings.
        with_annotations: If True, includes existing annotations in the
            render. Set False to see the underlying page only.
        overlay_grid: If True, overlays a 50-pt coordinate grid with
            labels of the form "x=100pt (g2)" — the PDF-point coordinate
            and the grid-cell index. To place annotations from what you
            see on the grid, either:
              - read the PDF-point label and pass it as-is (default), or
              - read the grid-cell index (g0, g1, ...) and pass it with
                grid_coords=True on the matching add_* call.

    Output is auto-downscaled / re-encoded as JPEG to stay under
    `max_bytes` (default 900 KB to fit the MCP 1 MB tool-result cap).
    The response is a 2-item list: [Image, info_dict] where info_dict =
        {actual_dpi, requested_dpi, format, size_bytes, downscaled,
         page_size_pt: [w, h]}.
    Read `info_dict.actual_dpi` to know what you actually got back — if
    it's lower than requested, the image was auto-downsampled to fit.

    Args (added):
        max_bytes: Override the 900 KB byte budget. The renderer will
            keep dropping DPI / quality until the encoded output fits.
        format: "auto" (default; PNG, falling back to JPEG when oversize),
            "png" (force lossless), or "jpeg" (force JPEG; useful when
            you want max DPI for visual verification of a large drawing).
    """
    path = _resolve_file(file)
    doc = _open(path)
    try:
        pg = _get_page(doc, page)
        page_w, page_h = pg.rect.width, pg.rect.height
        if overlay_grid:
            # Draw a non-persistent coordinate grid by working on a copy.
            scratch = fitz.open()
            scratch.insert_pdf(doc, from_page=page - 1, to_page=page - 1)
            sp = scratch[0]
            w, h = sp.rect.width, sp.rect.height
            grid_color = (0.7, 0.7, 0.85)
            label_color = (0.4, 0.4, 0.6)
            shape = sp.new_shape()
            x = 0
            while x <= w:
                shape.draw_line(fitz.Point(x, 0), fitz.Point(x, h))
                x += GRID_STEP
            y = 0
            while y <= h:
                shape.draw_line(fitz.Point(0, y), fitz.Point(w, y))
                y += GRID_STEP
            shape.finish(color=grid_color, width=0.3)
            shape.commit()
            # Labels every 2 grid cells along the top and left edges. Show
            # both the PDF-point coord and the grid index so the model can
            # use either (the latter via grid_coords=True on add_* tools).
            label_step = GRID_STEP * 2
            for x in range(0, int(w) + 1, label_step):
                sp.insert_text((x + 2, 10),
                               f"x={x}pt (g{x // GRID_STEP})",
                               fontsize=6, color=label_color)
            for y in range(0, int(h) + 1, label_step):
                sp.insert_text((4, y + 8),
                               f"y={y}pt (g{y // GRID_STEP})",
                               fontsize=6, color=label_color)
            try:
                img, info = _render_image(
                    lambda d: sp.get_pixmap(dpi=d, annots=with_annotations),
                    dpi,
                    max_bytes=max_bytes,
                    prefer_format=format,
                )
            finally:
                scratch.close()
        else:
            img, info = _render_image(
                lambda d: pg.get_pixmap(dpi=d, annots=with_annotations),
                dpi,
                max_bytes=max_bytes,
                prefer_format=format,
            )
        info["page_size_pt"] = [round(page_w, 2), round(page_h, 2)]
        return [img, info]
    finally:
        doc.close()


@_internal
def render_region(
    file: str,
    page: int,
    rect: list[float],
    dpi: int = 200,
    with_annotations: bool = True,
    grid_coords: bool = False,
    max_bytes: int = MAX_IMAGE_BYTES,
    format: str = "auto",
) -> list:
    """
    Render a specific region of a page at higher zoom. Best way to verify
    a placed annotation on a large sheet (A1, A0) without exceeding the
    1 MB tool-result cap.

    Returns [Image, info_dict] (same shape as render_page).

    Args:
        rect: [x0, y0, x1, y1] region to render, in points (top-left origin).
        dpi: Higher than render_page since the area is smaller.
        grid_coords: If True, interpret `rect` in 50-pt grid units (matching
            render_page's overlay grid).
        max_bytes / format: see render_page.
    """
    path = _resolve_file(file)
    doc = _open(path)
    try:
        pg = _get_page(doc, page)
        clip = _rect(rect, _scale_factor(grid_coords))
        img, info = _render_image(
            lambda d: pg.get_pixmap(dpi=d, clip=clip, annots=with_annotations),
            dpi,
            max_bytes=max_bytes,
            prefer_format=format,
        )
        info["clip_pt"] = [round(clip.x0, 2), round(clip.y0, 2),
                           round(clip.x1, 2), round(clip.y1, 2)]
        return [img, info]
    finally:
        doc.close()


@_internal
def render_tile(
    file: str,
    page: int,
    row: int,
    col: int,
    rows: int = 2,
    cols: int = 2,
    dpi: int = 150,
    with_annotations: bool = True,
    overlap: float = 20.0,
    max_bytes: int = MAX_IMAGE_BYTES,
    format: str = "auto",
) -> list:
    """
    Render one tile of a page sliced into a `rows` x `cols` grid.

    Use when render_page is forced to drop DPI so far that text becomes
    illegible (typical on A1/A0 sheets). Walk through (row, col) cells
    at a useful DPI (150-200) to read every region without hitting the
    1 MB result cap on any single tile.

    Tiles overlap by `overlap` PDF points on each shared edge so text
    cut by a tile boundary is still readable in at least one tile.

    Args:
        row: 0-indexed row, top to bottom.
        col: 0-indexed column, left to right.
        rows, cols: Grid dimensions. 2x2 is a good starting point for A1.
        dpi: Render DPI for the tile. Stays high because each tile is
            small enough to fit.
        overlap: PDF points of overlap between adjacent tiles.

    Returns [Image, info] where info adds {tile: [row, col],
    grid: [rows, cols], page_size_pt: [w, h], clip_pt: [x0,y0,x1,y1]}.
    """
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(
            f"tile ({row},{col}) outside grid ({rows} rows x {cols} cols)"
        )
    path = _resolve_file(file)
    doc = _open(path)
    try:
        pg = _get_page(doc, page)
        w, h = pg.rect.width, pg.rect.height
        tw, th = w / cols, h / rows
        x0 = max(0.0, col * tw - overlap)
        y0 = max(0.0, row * th - overlap)
        x1 = min(w, (col + 1) * tw + overlap)
        y1 = min(h, (row + 1) * th + overlap)
        clip = fitz.Rect(x0, y0, x1, y1)
        img, info = _render_image(
            lambda d: pg.get_pixmap(dpi=d, clip=clip, annots=with_annotations),
            dpi,
            max_bytes=max_bytes,
            prefer_format=format,
        )
        info["tile"] = [row, col]
        info["grid"] = [rows, cols]
        info["page_size_pt"] = [round(w, 2), round(h, 2)]
        info["clip_pt"] = [round(clip.x0, 2), round(clip.y0, 2),
                           round(clip.x1, 2), round(clip.y1, 2)]
        return [img, info]
    finally:
        doc.close()


@_internal
def find_text(
    file: str,
    page: int,
    query: str,
    quads: bool = False,
) -> list:
    """
    Locate every occurrence of a text string on a page and return its
    bounding boxes. Feed these directly into add_highlight, add_underline,
    add_strikeout, or add_squiggly to mark up specific words or phrases
    without having to read coordinates off a rendered image.

    Args:
        query: The text to search for. Case-sensitive by default.
        quads: If True, return quadrilateral coordinates suitable for
            multi-line spans. If False, return [x0,y0,x1,y1] rectangles.
    """
    path = _resolve_file(file)
    doc = _open(path)
    try:
        pg = _get_page(doc, page)
        if quads:
            results = pg.search_for(query, quads=True)
            return [[[round(p.x, 2), round(p.y, 2)] for p in
                     (q.ul, q.ur, q.lr, q.ll)] for q in results]
        results = pg.search_for(query)
        return [[round(r.x0, 2), round(r.y0, 2),
                 round(r.x1, 2), round(r.y1, 2)] for r in results]
    finally:
        doc.close()


@_internal
def get_text_blocks(file: str, page: int) -> list[dict]:
    """
    Return every text block on a page with its bounding box and text content.
    Useful when you want to programmatically locate sections of text (table
    cells, paragraphs, headings) for annotation without rendering the page.

    Each block: {"bbox": [x0, y0, x1, y1], "text": "...", "block_no": int}
    """
    path = _resolve_file(file)
    doc = _open(path)
    try:
        pg = _get_page(doc, page)
        out = []
        for b in pg.get_text("blocks"):
            x0, y0, x1, y1, text, block_no, btype = b[:7]
            if btype != 0:  # 0 = text block, 1 = image
                continue
            out.append({
                "bbox": [round(x0, 2), round(y0, 2),
                         round(x1, 2), round(y1, 2)],
                "text": text.strip(),
                "block_no": block_no,
            })
        return out
    finally:
        doc.close()


@_internal
def list_annotations(
    file: str,
    page: Optional[int] = None,
    authors: Optional[list[str]] = None,
    exclude_authors: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
    only_review: bool = False,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Read annotations from a PDF, with optional filtering.

    CAD-exported PDFs often contain hundreds of "Square" annotations
    representing SHX text fragments (grid letters, dimension labels).
    Use `only_review=True` (or `exclude_authors=["AutoCAD SHX Text"]`) to
    hide them. Use `limit` to cap the response size.

    Args:
        file: Project-scoped reference (J269:foo.pdf) or absolute path.
        page: Optional 1-indexed page number to filter by.
        authors: If set, keep only annotations whose author matches one
            of these (case-sensitive substring match on the title field).
        exclude_authors: Drop annotations from these authors. Common:
            ["AutoCAD SHX Text"] for CAD-imported drawings.
        types: Whitelist of PDF subtypes to keep (e.g. ["Highlight",
            "Square", "FreeText"]).
        exclude_types: Blacklist of subtypes to drop.
        only_review: Convenience: drop annotations from "AutoCAD SHX
            Text" authors (the most common noise source on CAD PDFs).
        limit: Cap the returned list to N annotations.
    """
    path = _resolve_file(file)
    doc = _open(path)
    try:
        out: list[dict] = []
        pages = [page] if page else range(1, doc.page_count + 1)
        ex_authors = set(exclude_authors or [])
        if only_review:
            ex_authors.add("AutoCAD SHX Text")
        keep_authors = set(authors) if authors else None
        keep_types = set(types) if types else None
        ex_types = set(exclude_types or [])
        for p in pages:
            pg = _get_page(doc, p)
            for annot in pg.annots() or []:
                d = _annot_to_dict(annot)
                a = d.get("author") or ""
                t = d.get("type") or ""
                if a in ex_authors:
                    continue
                if keep_authors is not None and a not in keep_authors:
                    continue
                if t in ex_types:
                    continue
                if keep_types is not None and t not in keep_types:
                    continue
                out.append(d)
                if limit and len(out) >= limit:
                    return out
        return out
    finally:
        doc.close()


# --- Text-anchored annotations --------------------------------------------
#
# These tools accept a text query and locate the placement automatically.
# They are the preferred path for any PDF with a real text layer (specs,
# procedures, contracts, native PDFs). For CAD drawings or scanned PDFs
# without a text layer, fall back to the coordinate-based add_* tools.


def _find_quads(doc: fitz.Document, query: str,
                page: Optional[int]) -> list[tuple[int, fitz.Quad]]:
    """Search for `query`. Returns list of (page_num, Quad)."""
    out: list[tuple[int, fitz.Quad]] = []
    pages = [page] if page else range(1, doc.page_count + 1)
    for p in pages:
        pg = _get_page(doc, p)
        for q in pg.search_for(query, quads=True) or []:
            out.append((p, q))
    return out


def _select_matches(matches: list, occurrence: Optional[int]) -> list:
    if not matches:
        return []
    if occurrence is None:
        return matches
    if not (0 <= occurrence < len(matches)):
        raise ValueError(
            f"occurrence {occurrence} out of range "
            f"(found {len(matches)} matches; valid 0..{len(matches)-1})"
        )
    return [matches[occurrence]]


@_internal
def highlight_text(
    file: str,
    query: str,
    page: Optional[int] = None,
    occurrence: Optional[int] = None,
    kind: str = "highlight",
    color: Any = None,
    content: Optional[str] = None,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Find a text string and apply a markup over each match. No coordinates
    needed. Multi-line spans are handled correctly via PDF quads.

    Args:
        query: Exact substring to find. Case-sensitive.
        page: Search this 1-indexed page only (default: all pages).
        occurrence: 0-indexed match to mark (default: mark every match).
        kind: highlight | underline | strikeout | squiggly. Defaults to
            highlight; pass "strikeout" for typos, "squiggly" for "review
            this", "underline" for emphasis.
        color: Override the default markup color (yellow / blue / red /
            orange respectively).
        content: Optional comment that travels with the markup (visible
            in Acrobat's comments panel and in list_annotations).

    Returns: {saved_to, kind, count, total_found, matches: [{id, page,
        rect}, ...]}.  total_found is how many matches existed in the PDF
        (so you can tell whether `occurrence` was needed).
    """
    if kind not in ("highlight", "underline", "strikeout", "squiggly"):
        raise ValueError(
            f"kind must be highlight|underline|strikeout|squiggly, got {kind!r}"
        )
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        matches = _find_quads(doc, query, page)
        if not matches:
            scope = f" on page {page}" if page else ""
            raise ValueError(f"No matches for {query!r}{scope}")
        selected = _select_matches(matches, occurrence)

        defaults = {"highlight": "#ffff00", "underline": "#0000ff",
                    "strikeout": "#ff0000", "squiggly": "#ff8800"}
        clr = color if color is not None else defaults[kind]

        adders = {
            "highlight": "add_highlight_annot",
            "underline": "add_underline_annot",
            "strikeout": "add_strikeout_annot",
            "squiggly": "add_squiggly_annot",
        }
        results = []
        for p, quad in selected:
            pg = _get_page(doc, p)
            annot = getattr(pg, adders[kind])(quad)
            _set_common(annot, content=content, author=author, color=clr)
            r = annot.rect
            results.append({
                "id": annot.xref, "page": p,
                "rect": [round(r.x0, 2), round(r.y0, 2),
                         round(r.x1, 2), round(r.y1, 2)],
            })
        saved = _save(doc, write_path, mode)
        return {"saved_to": saved, "kind": kind, "count": len(results),
                "total_found": len(matches), "matches": results}
    finally:
        doc.close()


def _placement_point(rect: fitz.Rect, placement: str, offset: float,
                     page_rect: fitz.Rect) -> fitz.Point:
    """
    Return a placement point relative to a text rect. The sticky-note
    icon is anchored at its top-left, so this returns the point where
    that top-left should sit. Icons render at roughly 20 pt square.
    """
    if placement in ("after", "right"):
        return fitz.Point(min(rect.x1 + offset, page_rect.x1 - 22),
                          rect.y0)
    if placement in ("before", "left"):
        return fitz.Point(max(rect.x0 - offset - 20, 2), rect.y0)
    if placement == "above":
        return fitz.Point(rect.x0, max(rect.y0 - 22 - offset, 2))
    if placement == "below":
        return fitz.Point(rect.x0,
                          min(rect.y1 + offset, page_rect.y1 - 22))
    if placement in ("overlay", "on"):
        return fitz.Point(rect.x0, rect.y0)
    raise ValueError(
        f"placement must be after|before|above|below|overlay, got {placement!r}"
    )


@_internal
def add_sticky_note_at_text(
    file: str,
    query: str,
    text: str,
    page: Optional[int] = None,
    occurrence: Optional[int] = 0,
    icon: str = "Note",
    placement: str = "after",
    offset: float = 6,
    color: Any = "#ffff00",
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Find a text string and drop a sticky note next to it. No coordinates.

    Args:
        query: Substring to find. Case-sensitive.
        text: The note's content.
        page: Restrict to this 1-indexed page (default: all).
        occurrence: 0-indexed match number (default: first match). Pass
            None to attach a note to every match.
        icon: Comment | Note | Help | Key | NewParagraph | Paragraph |
            Insert. The icon anchors at its top-left, ~20 pt square.
        placement: Where the icon sits relative to the matched text:
            "after" (right of the text, default), "before", "above",
            "below", or "overlay" (covers it).
        offset: Gap in PDF points between icon and text.

    Returns: {saved_to, count, total_found, notes: [{id, page,
        anchor_rect, point}, ...]}.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        matches = _find_quads(doc, query, page)
        if not matches:
            scope = f" on page {page}" if page else ""
            raise ValueError(f"No matches for {query!r}{scope}")
        selected = _select_matches(matches, occurrence)

        results = []
        for p, quad in selected:
            pg = _get_page(doc, p)
            r = quad.rect
            point = _placement_point(r, placement, offset, pg.rect)
            annot = pg.add_text_annot(point, text, icon=icon)
            _set_common(annot, content=text, author=author, color=color)
            results.append({
                "id": annot.xref, "page": p,
                "anchor_rect": [round(r.x0, 2), round(r.y0, 2),
                                round(r.x1, 2), round(r.y1, 2)],
                "point": [round(point.x, 2), round(point.y, 2)],
            })
        saved = _save(doc, write_path, mode)
        return {"saved_to": saved, "count": len(results),
                "total_found": len(matches), "notes": results}
    finally:
        doc.close()


def _auto_callout_geometry(text_rect: fitz.Rect, page_rect: fitz.Rect,
                           placement: str, box_w: float, box_h: float
                           ) -> tuple[fitz.Point, fitz.Point, fitz.Rect]:
    """
    Compute (callout_point, knee_point, text_box_rect) for a callout
    pointing at text_rect. Tries to put the box in the nearest empty
    margin. `placement` is "auto" | "right" | "left" | "above" | "below".
    """
    margin = 12
    cx = (text_rect.x0 + text_rect.x1) / 2
    cy = (text_rect.y0 + text_rect.y1) / 2

    if placement == "auto":
        right_space = page_rect.x1 - text_rect.x1
        left_space = text_rect.x0 - page_rect.x0
        placement = "right" if right_space >= left_space else "left"

    if placement == "right":
        callout = fitz.Point(text_rect.x1, cy)
        bx0 = min(page_rect.x1 - box_w - margin, text_rect.x1 + 60)
        bx0 = max(bx0, text_rect.x1 + 20)
        by0 = max(margin, cy - box_h / 2)
        by0 = min(by0, page_rect.y1 - box_h - margin)
        text_box = fitz.Rect(bx0, by0, bx0 + box_w, by0 + box_h)
        knee = fitz.Point((callout.x + text_box.x0) / 2, callout.y)
    elif placement == "left":
        callout = fitz.Point(text_rect.x0, cy)
        bx1 = max(page_rect.x0 + box_w + margin, text_rect.x0 - 60)
        bx1 = min(bx1, text_rect.x0 - 20)
        by0 = max(margin, cy - box_h / 2)
        by0 = min(by0, page_rect.y1 - box_h - margin)
        text_box = fitz.Rect(bx1 - box_w, by0, bx1, by0 + box_h)
        knee = fitz.Point((callout.x + text_box.x1) / 2, callout.y)
    elif placement == "above":
        callout = fitz.Point(cx, text_rect.y0)
        by1 = max(page_rect.y0 + box_h + margin, text_rect.y0 - 30)
        by1 = min(by1, text_rect.y0 - 8)
        bx0 = max(margin, cx - box_w / 2)
        bx0 = min(bx0, page_rect.x1 - box_w - margin)
        text_box = fitz.Rect(bx0, by1 - box_h, bx0 + box_w, by1)
        knee = fitz.Point(callout.x, (callout.y + text_box.y1) / 2)
    elif placement == "below":
        callout = fitz.Point(cx, text_rect.y1)
        by0 = min(page_rect.y1 - box_h - margin, text_rect.y1 + 30)
        by0 = max(by0, text_rect.y1 + 8)
        bx0 = max(margin, cx - box_w / 2)
        bx0 = min(bx0, page_rect.x1 - box_w - margin)
        text_box = fitz.Rect(bx0, by0, bx0 + box_w, by0 + box_h)
        knee = fitz.Point(callout.x, (callout.y + text_box.y0) / 2)
    else:
        raise ValueError(
            f"placement must be auto|right|left|above|below, got {placement!r}"
        )
    return callout, knee, text_box


@_internal
def add_callout_at_text(
    file: str,
    query: str,
    text: str,
    page: Optional[int] = None,
    occurrence: Optional[int] = 0,
    placement: str = "auto",
    color: Any = "#ff0000",
    fill: Any = "#ffffe0",
    text_color: Any = "#000000",
    font_size: float = 10,
    box_width: float = 220,
    box_height: float = 60,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Find a text string and place a callout pointing at it. The text box
    is auto-positioned in the nearest margin (right by default; falls
    back to left if the right margin is too narrow).

    Args:
        query: Substring to find.
        text: The callout's comment text.
        placement: "auto" (default; pick nearest margin) | "right" |
            "left" | "above" | "below".
        box_width / box_height: Text box dimensions in points. Tune for
            longer comments.

    Returns: {saved_to, count, total_found, callouts: [{id, page,
        anchor_rect, callout_point, knee_point, text_rect}, ...]}.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        matches = _find_quads(doc, query, page)
        if not matches:
            scope = f" on page {page}" if page else ""
            raise ValueError(f"No matches for {query!r}{scope}")
        selected = _select_matches(matches, occurrence)

        text_c = _parse_color(text_color) or (0, 0, 0)
        fill_c = _parse_color(fill)
        border_c = _parse_color(color)

        results = []
        for p, quad in selected:
            pg = _get_page(doc, p)
            anchor = quad.rect
            callout_pt, knee_pt, tbox = _auto_callout_geometry(
                anchor, pg.rect, placement, box_width, box_height
            )
            annot = pg.add_freetext_annot(
                tbox, text,
                fontsize=font_size,
                text_color=text_c,
                fill_color=fill_c,
                callout=[callout_pt, knee_pt,
                         fitz.Point(tbox.x0, tbox.y0)],
                line_end=fitz.PDF_ANNOT_LE_OPEN_ARROW,
            )
            info = annot.info
            info["title"] = author or DEFAULT_AUTHOR
            annot.set_info(info)
            if border_c is not None:
                try:
                    cols = annot.colors or {}
                    cols["stroke"] = border_c
                    annot.set_colors(cols)
                except Exception:
                    pass
            try:
                annot.set_border({"width": 1.0})
            except Exception:
                pass
            annot.update()
            results.append({
                "id": annot.xref, "page": p,
                "anchor_rect": [round(anchor.x0, 2), round(anchor.y0, 2),
                                round(anchor.x1, 2), round(anchor.y1, 2)],
                "callout_point": [round(callout_pt.x, 2),
                                  round(callout_pt.y, 2)],
                "knee_point": [round(knee_pt.x, 2), round(knee_pt.y, 2)],
                "text_rect": [round(tbox.x0, 2), round(tbox.y0, 2),
                              round(tbox.x1, 2), round(tbox.y1, 2)],
            })
        saved = _save(doc, write_path, mode)
        return {"saved_to": saved, "count": len(results),
                "total_found": len(matches), "callouts": results}
    finally:
        doc.close()


# --- Shape annotations -----------------------------------------------------

@_internal
def add_rectangle(
    file: str,
    page: int,
    rect: list[float],
    content: Optional[str] = None,
    color: Any = "#ff0000",
    fill: Any = None,
    border_width: float = 1.5,
    cloud_intensity: float = 0,
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a rectangle (Square) annotation. Set cloud_intensity > 0 (typical 1-2)
    to get the cloudy border Acrobat reviewers use for markups.

    Args:
        rect: [x0, y0, x1, y1] in points, top-left origin. If
            grid_coords=True, interpret as multiples of GRID_STEP (50 pt),
            matching the overlay grid drawn by render_page.
        in_place: If True, modify the source file directly. Default False
            saves to <source>.annotated.pdf next to the source.
        grid_coords: Read coords as grid-cell units (one cell = 50 pt) so
            you can place annotations using the same numbers shown on the
            render_page overlay grid.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        annot = pg.add_rect_annot(_rect(rect, s))
        _set_common(annot, content=content, author=author,
                    color=color, fill=fill, opacity=opacity,
                    border_width=border_width,
                    cloud_intensity=cloud_intensity)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_circle(
    file: str,
    page: int,
    rect: list[float],
    content: Optional[str] = None,
    color: Any = "#ff0000",
    fill: Any = None,
    border_width: float = 1.5,
    cloud_intensity: float = 0,
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a circle/oval (Circle) annotation. The shape is inscribed in the
    given bounding rectangle. For a true circle, pass a square rect. Set
    grid_coords=True to read `rect` in 50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        annot = pg.add_circle_annot(_rect(rect, s))
        _set_common(annot, content=content, author=author,
                    color=color, fill=fill, opacity=opacity,
                    border_width=border_width,
                    cloud_intensity=cloud_intensity)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_polygon(
    file: str,
    page: int,
    points: list[list[float]],
    content: Optional[str] = None,
    color: Any = "#ff0000",
    fill: Any = None,
    border_width: float = 1.5,
    cloud_intensity: float = 0,
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a closed polygon. Use cloud_intensity > 0 to get a cloudy border,
    which is the standard markup style for "this whole region needs revision".

    Args:
        points: [[x, y], ...] vertices in document order.
        grid_coords: Read points in 50-pt grid units instead of PDF points.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        pts = [_point(p, s) for p in points]
        annot = pg.add_polygon_annot(pts)
        _set_common(annot, content=content, author=author,
                    color=color, fill=fill, opacity=opacity,
                    border_width=border_width,
                    cloud_intensity=cloud_intensity)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_line(
    file: str,
    page: int,
    start: list[float],
    end: list[float],
    content: Optional[str] = None,
    color: Any = "#ff0000",
    border_width: float = 1.5,
    line_start: str = "None",
    line_end: str = "None",
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a line annotation. Set line_end="OpenArrow" for an arrow.

    Valid line endings: None, Square, Circle, Diamond, OpenArrow,
    ClosedArrow, Butt, ROpenArrow, RClosedArrow, Slash.

    grid_coords=True reads start/end in 50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        annot = pg.add_line_annot(_point(start, s), _point(end, s))
        # Line endings
        try:
            le_map = {
                "None": fitz.PDF_ANNOT_LE_NONE,
                "Square": fitz.PDF_ANNOT_LE_SQUARE,
                "Circle": fitz.PDF_ANNOT_LE_CIRCLE,
                "Diamond": fitz.PDF_ANNOT_LE_DIAMOND,
                "OpenArrow": fitz.PDF_ANNOT_LE_OPEN_ARROW,
                "ClosedArrow": fitz.PDF_ANNOT_LE_CLOSED_ARROW,
                "Butt": fitz.PDF_ANNOT_LE_BUTT,
                "ROpenArrow": fitz.PDF_ANNOT_LE_R_OPEN_ARROW,
                "RClosedArrow": fitz.PDF_ANNOT_LE_R_CLOSED_ARROW,
                "Slash": fitz.PDF_ANNOT_LE_SLASH,
            }
            annot.set_line_ends(le_map.get(line_start, fitz.PDF_ANNOT_LE_NONE),
                                le_map.get(line_end, fitz.PDF_ANNOT_LE_NONE))
        except Exception:
            pass
        _set_common(annot, content=content, author=author, color=color,
                    opacity=opacity, border_width=border_width)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_arrow(
    file: str,
    page: int,
    start: list[float],
    end: list[float],
    content: Optional[str] = None,
    color: Any = "#ff0000",
    border_width: float = 1.5,
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """Convenience: add a line with an open arrow at the end."""
    return add_line(file=file, page=page, start=start, end=end,
                    content=content, color=color, border_width=border_width,
                    line_start="None", line_end="OpenArrow",
                    opacity=opacity, author=author,
                    output_path=output_path, in_place=in_place,
                    grid_coords=grid_coords)


# --- Text-region markup ----------------------------------------------------

def _markup(kind: str, file: str, page: int, rect: list[float],
            quads: Optional[list[list[float]]],
            content, color, opacity, author, output_path, in_place,
            grid_coords) -> dict:
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        if quads:
            qlist = []
            for q in quads:
                if len(q) == 4:
                    qlist.append(fitz.Quad(_rect(q, s).quad))
                else:
                    qlist.append(fitz.Quad([_point(p, s) for p in q]))
            target = qlist
        else:
            target = _rect(rect, s)
        if kind == "highlight":
            annot = pg.add_highlight_annot(target)
        elif kind == "underline":
            annot = pg.add_underline_annot(target)
        elif kind == "strikeout":
            annot = pg.add_strikeout_annot(target)
        elif kind == "squiggly":
            annot = pg.add_squiggly_annot(target)
        else:
            raise ValueError(f"Unknown markup kind: {kind}")
        _set_common(annot, content=content, author=author, color=color,
                    opacity=opacity)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_highlight(
    file: str, page: int,
    rect: Optional[list[float]] = None,
    quads: Optional[list[list[float]]] = None,
    content: Optional[str] = None,
    color: Any = "#ffff00",
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Highlight text. Provide either a rect (single span) or quads (list of
    quadrilaterals for multi-line selections). grid_coords=True reads
    rect/quads in 50-pt grid units.
    """
    return _markup("highlight", file, page, rect, quads,
                   content, color, opacity, author, output_path, in_place,
                   grid_coords)


@_internal
def add_underline(
    file: str, page: int,
    rect: Optional[list[float]] = None,
    quads: Optional[list[list[float]]] = None,
    content: Optional[str] = None,
    color: Any = "#0000ff",
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """Underline a text region."""
    return _markup("underline", file, page, rect, quads,
                   content, color, opacity, author, output_path, in_place,
                   grid_coords)


@_internal
def add_strikeout(
    file: str, page: int,
    rect: Optional[list[float]] = None,
    quads: Optional[list[list[float]]] = None,
    content: Optional[str] = None,
    color: Any = "#ff0000",
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """Strike through a text region."""
    return _markup("strikeout", file, page, rect, quads,
                   content, color, opacity, author, output_path, in_place,
                   grid_coords)


@_internal
def add_squiggly(
    file: str, page: int,
    rect: Optional[list[float]] = None,
    quads: Optional[list[list[float]]] = None,
    content: Optional[str] = None,
    color: Any = "#ff8800",
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """Squiggly underline (typical "review this" mark)."""
    return _markup("squiggly", file, page, rect, quads,
                   content, color, opacity, author, output_path, in_place,
                   grid_coords)


# --- Note / text annotations ----------------------------------------------

@_internal
def add_sticky_note(
    file: str,
    page: int,
    point: list[float],
    text: str,
    icon: str = "Note",
    color: Any = "#ffff00",
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a sticky note (Text annotation). Icons: Comment, Note, Help, Key,
    NewParagraph, Paragraph, Insert. grid_coords=True reads `point` in
    50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        annot = pg.add_text_annot(_point(point, s), text, icon=icon)
        _set_common(annot, content=text, author=author, color=color)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_freetext(
    file: str,
    page: int,
    rect: list[float],
    text: str,
    font_size: float = 11,
    font: str = "helv",
    color: Any = "#000000",
    fill: Any = None,
    border_color: Any = None,
    border_width: float = 0,
    align: str = "left",
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a free-floating text box (FreeText annotation).

    Args:
        align: left | center | right
        font: helv | tiro | cour (Helvetica / Times / Courier)
        grid_coords: Read `rect` in 50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        align_map = {"left": fitz.TEXT_ALIGN_LEFT,
                     "center": fitz.TEXT_ALIGN_CENTER,
                     "right": fitz.TEXT_ALIGN_RIGHT}
        text_color_t = _parse_color(color) or (0, 0, 0)
        fill_color_t = _parse_color(fill)
        border_c = _parse_color(border_color)
        annot = pg.add_freetext_annot(
            _rect(rect, s), text,
            fontsize=font_size,
            fontname=font,
            text_color=text_color_t,
            fill_color=fill_color_t,
            align=align_map.get(align, fitz.TEXT_ALIGN_LEFT),
        )
        info = annot.info
        info["title"] = author or DEFAULT_AUTHOR
        annot.set_info(info)
        if border_c is not None:
            try:
                colors = annot.colors or {}
                colors["stroke"] = border_c
                annot.set_colors(colors)
            except Exception:
                pass
        if border_width and border_c is not None:
            try:
                annot.set_border({"width": border_width})
            except Exception:
                pass
        annot.update()
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_callout(
    file: str,
    page: int,
    callout_point: list[float],
    knee_point: list[float],
    text_rect: list[float],
    text: str,
    font_size: float = 11,
    color: Any = "#ff0000",
    fill: Any = "#ffffe0",
    text_color: Any = "#000000",
    line_end: str = "OpenArrow",
    border_width: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a FreeText callout: arrow points at callout_point, kinks at
    knee_point, then enters the text box at text_rect. Set grid_coords=True
    to read all three coords in 50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        text_c = _parse_color(text_color) or (0, 0, 0)
        fill_c = _parse_color(fill)
        border_c = _parse_color(color)
        annot = pg.add_freetext_annot(
            _rect(text_rect, s), text,
            fontsize=font_size,
            text_color=text_c,
            fill_color=fill_c,
            callout=[_point(callout_point, s), _point(knee_point, s),
                     _point([text_rect[0], text_rect[1]], s)],
            line_end=getattr(fitz, f"PDF_ANNOT_LE_{line_end.upper()}",
                             fitz.PDF_ANNOT_LE_OPEN_ARROW),
        )
        info = annot.info
        info["title"] = author or DEFAULT_AUTHOR
        annot.set_info(info)
        # Border color goes on the colors dict (the freetext_annot
        # parameter only works with rich_text mode, and set_colors itself
        # is restricted on FreeText, so fall back silently).
        if border_c is not None:
            try:
                colors = annot.colors or {}
                colors["stroke"] = border_c
                annot.set_colors(colors)
            except Exception:
                pass
        if border_width:
            try:
                annot.set_border({"width": border_width})
            except Exception:
                pass
        annot.update()
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


# --- Ink and stamps --------------------------------------------------------

@_internal
def add_ink(
    file: str,
    page: int,
    strokes: list[list[list[float]]],
    content: Optional[str] = None,
    color: Any = "#ff0000",
    border_width: float = 1.5,
    opacity: float = 1.0,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add freehand ink strokes.

    Args:
        strokes: list of strokes; each stroke is a list of [x, y] points.
        grid_coords: Read all stroke points in 50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        ink_list = [[(float(p[0]) * s, float(p[1]) * s) for p in stroke]
                    for stroke in strokes]
        annot = pg.add_ink_annot(ink_list)
        _set_common(annot, content=content, author=author, color=color,
                    opacity=opacity, border_width=border_width)
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def add_stamp(
    file: str,
    page: int,
    rect: list[float],
    stamp: str = "Approved",
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add a rubber-stamp annotation. Common values: Approved, AsIs, Confidential,
    Departmental, Draft, Experimental, Expired, Final, ForComment, ForPublicRelease,
    NotApproved, NotForPublicRelease, Sold, TopSecret.

    grid_coords=True reads `rect` in 50-pt grid units.
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        pg = _get_page(doc, page)
        stamp_map = {
            "Approved": fitz.STAMP_Approved,
            "AsIs": fitz.STAMP_AsIs,
            "Confidential": fitz.STAMP_Confidential,
            "Departmental": fitz.STAMP_Departmental,
            "Draft": fitz.STAMP_Draft,
            "Experimental": fitz.STAMP_Experimental,
            "Expired": fitz.STAMP_Expired,
            "Final": fitz.STAMP_Final,
            "ForComment": fitz.STAMP_ForComment,
            "ForPublicRelease": fitz.STAMP_ForPublicRelease,
            "NotApproved": fitz.STAMP_NotApproved,
            "NotForPublicRelease": fitz.STAMP_NotForPublicRelease,
            "Sold": fitz.STAMP_Sold,
            "TopSecret": fitz.STAMP_TopSecret,
        }
        annot = pg.add_stamp_annot(_rect(rect, s),
                                   stamp_map.get(stamp, fitz.STAMP_Approved))
        info = annot.info
        info["title"] = author or DEFAULT_AUTHOR
        annot.set_info(info)
        annot.update()
        saved = _save(doc, write_path, mode)
        return {"id": annot.xref, "saved_to": saved}
    finally:
        doc.close()


# --- Replies and management ------------------------------------------------

@_internal
def reply_to_annotation(
    file: str,
    annot_id: int,
    text: str,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Reply to an existing annotation (creates a Text annot with IRT pointing
    to the parent, which is how Acrobat reply threads are encoded).

    Args:
        annot_id: xref of the parent annotation (from list_annotations).
    """
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        # Locate the parent annot. Hold a strong reference to the owning
        # page so PyMuPDF doesn't garbage-collect it under the annot.
        parent = None
        parent_page = None
        parent_rect = None
        for i in range(doc.page_count):
            pg = doc[i]
            for a in pg.annots() or []:
                if a.xref == annot_id:
                    parent = a
                    parent_page = pg
                    parent_rect = fitz.Rect(a.rect)
                    break
            if parent:
                break
        if not parent or parent_page is None or parent_rect is None:
            raise ValueError(f"Annotation id {annot_id} not found")

        # Create the reply at the parent's top-left corner
        reply = parent_page.add_text_annot(parent_rect.tl, text)
        info = reply.info
        info["title"] = author or DEFAULT_AUTHOR
        info["content"] = text
        reply.set_info(info)
        reply.update()

        # Set IRT (in-reply-to) and RT (reply type) keys
        doc.xref_set_key(reply.xref, "IRT", f"{parent.xref} 0 R")
        doc.xref_set_key(reply.xref, "RT", "/R")

        saved = _save(doc, write_path, mode)
        return {"id": reply.xref, "in_reply_to": parent.xref, "saved_to": saved}
    finally:
        doc.close()


@_internal
def delete_annotation(
    file: str,
    annot_id: int,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """Delete an annotation by its xref id."""
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        for i in range(doc.page_count):
            pg = doc[i]
            for a in pg.annots() or []:
                if a.xref == annot_id:
                    pg.delete_annot(a)
                    saved = _save(doc, write_path, mode)
                    return {"deleted": annot_id, "saved_to": saved}
        raise ValueError(f"Annotation id {annot_id} not found")
    finally:
        doc.close()


@_internal
def update_annotation(
    file: str,
    annot_id: int,
    content: Optional[str] = None,
    color: Any = None,
    fill: Any = None,
    opacity: Optional[float] = None,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """Modify content, colors, opacity, or author of an existing annotation."""
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        for i in range(doc.page_count):
            pg = doc[i]
            for a in pg.annots() or []:
                if a.xref == annot_id:
                    _set_common(a, content=content, author=author,
                                color=color, fill=fill, opacity=opacity)
                    saved = _save(doc, write_path, mode)
                    return {"id": annot_id, "saved_to": saved}
        raise ValueError(f"Annotation id {annot_id} not found")
    finally:
        doc.close()


# --- Batch operations ------------------------------------------------------
#
# Annotations are identified by their PDF xref id, which is the integer
# returned as `id` from any add_* call and from list_annotations(). Pass
# those ids to update_annotations / delete_annotations to manage many at
# once in a single open/save cycle.

_LE_NAMES = {
    "None": "PDF_ANNOT_LE_NONE", "Square": "PDF_ANNOT_LE_SQUARE",
    "Circle": "PDF_ANNOT_LE_CIRCLE", "Diamond": "PDF_ANNOT_LE_DIAMOND",
    "OpenArrow": "PDF_ANNOT_LE_OPEN_ARROW",
    "ClosedArrow": "PDF_ANNOT_LE_CLOSED_ARROW",
    "Butt": "PDF_ANNOT_LE_BUTT",
    "ROpenArrow": "PDF_ANNOT_LE_R_OPEN_ARROW",
    "RClosedArrow": "PDF_ANNOT_LE_R_CLOSED_ARROW",
    "Slash": "PDF_ANNOT_LE_SLASH",
}


def _le(name: str):
    return getattr(fitz, _LE_NAMES.get(name, "PDF_ANNOT_LE_NONE"),
                   fitz.PDF_ANNOT_LE_NONE)


_STAMP_NAMES = {
    "Approved": "STAMP_Approved", "AsIs": "STAMP_AsIs",
    "Confidential": "STAMP_Confidential", "Departmental": "STAMP_Departmental",
    "Draft": "STAMP_Draft", "Experimental": "STAMP_Experimental",
    "Expired": "STAMP_Expired", "Final": "STAMP_Final",
    "ForComment": "STAMP_ForComment",
    "ForPublicRelease": "STAMP_ForPublicRelease",
    "NotApproved": "STAMP_NotApproved",
    "NotForPublicRelease": "STAMP_NotForPublicRelease",
    "Sold": "STAMP_Sold", "TopSecret": "STAMP_TopSecret",
}


def _create_one(pg: fitz.Page, item: dict, scale: float) -> fitz.Annot:
    """
    Create a single annotation on `pg` from a batch item dict. Mirrors
    the per-type logic of the corresponding add_* tool. Returns the Annot.
    """
    t = item.get("type")
    if not t:
        raise ValueError(f"item missing 'type': {item}")

    color = item.get("color", "#ff0000")
    fill = item.get("fill")
    content = item.get("content")
    author = item.get("author")
    opacity = item.get("opacity", 1.0)
    border_width = item.get("border_width")
    cloud_intensity = item.get("cloud_intensity", 0)

    if t == "rectangle":
        annot = pg.add_rect_annot(_rect(item["rect"], scale))
    elif t == "circle":
        annot = pg.add_circle_annot(_rect(item["rect"], scale))
    elif t == "polygon":
        annot = pg.add_polygon_annot(
            [_point(p, scale) for p in item["points"]]
        )
    elif t in ("line", "arrow"):
        annot = pg.add_line_annot(
            _point(item["start"], scale),
            _point(item["end"], scale),
        )
        ls = item.get("line_start", "None")
        le = item.get("line_end",
                      "OpenArrow" if t == "arrow" else "None")
        try:
            annot.set_line_ends(_le(ls), _le(le))
        except Exception:
            pass
    elif t in ("highlight", "underline", "strikeout", "squiggly"):
        if item.get("quads"):
            target = []
            for q in item["quads"]:
                if len(q) == 4:
                    target.append(fitz.Quad(_rect(q, scale).quad))
                else:
                    target.append(fitz.Quad([_point(p, scale) for p in q]))
        else:
            target = _rect(item["rect"], scale)
        adder = {
            "highlight": pg.add_highlight_annot,
            "underline": pg.add_underline_annot,
            "strikeout": pg.add_strikeout_annot,
            "squiggly": pg.add_squiggly_annot,
        }[t]
        annot = adder(target)
        # Markup defaults differ by type
        if t == "highlight":
            color = item.get("color", "#ffff00")
        elif t == "underline":
            color = item.get("color", "#0000ff")
        elif t == "squiggly":
            color = item.get("color", "#ff8800")
    elif t == "sticky_note":
        annot = pg.add_text_annot(
            _point(item["point"], scale),
            item["text"],
            icon=item.get("icon", "Note"),
        )
        content = content or item["text"]
        color = item.get("color", "#ffff00")
    elif t == "freetext":
        align_map = {"left": fitz.TEXT_ALIGN_LEFT,
                     "center": fitz.TEXT_ALIGN_CENTER,
                     "right": fitz.TEXT_ALIGN_RIGHT}
        annot = pg.add_freetext_annot(
            _rect(item["rect"], scale),
            item["text"],
            fontsize=item.get("font_size", 11),
            fontname=item.get("font", "helv"),
            text_color=_parse_color(item.get("color", "#000000")) or (0, 0, 0),
            fill_color=_parse_color(fill),
            align=align_map.get(item.get("align", "left"),
                                fitz.TEXT_ALIGN_LEFT),
        )
        info = annot.info
        info["title"] = author or DEFAULT_AUTHOR
        annot.set_info(info)
        border_c = _parse_color(item.get("border_color"))
        if border_c is not None:
            try:
                colors = annot.colors or {}
                colors["stroke"] = border_c
                annot.set_colors(colors)
            except Exception:
                pass
        if border_width:
            try:
                annot.set_border({"width": border_width})
            except Exception:
                pass
        annot.update()
        return annot
    elif t == "callout":
        text_rect = _rect(item["text_rect"], scale)
        annot = pg.add_freetext_annot(
            text_rect,
            item["text"],
            fontsize=item.get("font_size", 11),
            text_color=_parse_color(item.get("text_color")) or (0, 0, 0),
            fill_color=_parse_color(item.get("fill", "#ffffe0")),
            callout=[
                _point(item["callout_point"], scale),
                _point(item["knee_point"], scale),
                _point([item["text_rect"][0], item["text_rect"][1]], scale),
            ],
            line_end=_le(item.get("line_end", "OpenArrow")),
        )
        info = annot.info
        info["title"] = author or DEFAULT_AUTHOR
        annot.set_info(info)
        border_c = _parse_color(item.get("color", "#ff0000"))
        if border_c is not None:
            try:
                colors = annot.colors or {}
                colors["stroke"] = border_c
                annot.set_colors(colors)
            except Exception:
                pass
        bw = item.get("border_width", 1.0)
        if bw:
            try:
                annot.set_border({"width": bw})
            except Exception:
                pass
        annot.update()
        return annot
    elif t == "ink":
        ink_list = [
            [(float(p[0]) * scale, float(p[1]) * scale) for p in stroke]
            for stroke in item["strokes"]
        ]
        annot = pg.add_ink_annot(ink_list)
    elif t == "stamp":
        s = item.get("stamp", "Approved")
        stamp_const = getattr(fitz, _STAMP_NAMES.get(s, "STAMP_Approved"),
                              fitz.STAMP_Approved)
        annot = pg.add_stamp_annot(_rect(item["rect"], scale), stamp_const)
        info = annot.info
        info["title"] = author or DEFAULT_AUTHOR
        annot.set_info(info)
        annot.update()
        return annot
    else:
        raise ValueError(f"Unknown annotation type: {t!r}")

    _set_common(annot, content=content, author=author,
                color=color, fill=fill, opacity=opacity,
                border_width=border_width,
                cloud_intensity=cloud_intensity)
    return annot


@_internal
def add_annotations(
    file: str,
    items: list[dict],
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Add many annotations in one open/save cycle. Each new annotation is
    identified by the returned `id` (its PDF xref) — pass that id to
    update_annotations / delete_annotations later.

    Each item is a dict mirroring an add_* tool's kwargs, plus:
      - `type`: one of rectangle, circle, polygon, line, arrow, highlight,
        underline, strikeout, squiggly, sticky_note, freetext, callout,
        ink, stamp.
      - `page`: 1-indexed page number (required on every item).

    Per-item args follow the same names/shapes as the matching single
    add_* tool. Save target (`output_path`, `in_place`) and `grid_coords`
    are call-level — set them once at the top, not on each item.

    Example:
        items=[
          {"type": "rectangle", "page": 1, "rect": [180,320,410,460],
           "cloud_intensity": 1.5, "color": "#ff0000"},
          {"type": "highlight", "page": 1, "rect": [72,95,200,110]},
          {"type": "sticky_note", "page": 1, "point": [400,400],
           "text": "see clause 7.4"},
        ]

    Returns: {saved_to, count, annotations: [{id, type, page}, ...]}
    """
    if not items:
        return {"saved_to": str(_resolve_file(file)), "count": 0,
                "annotations": []}
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    s = _scale_factor(grid_coords)
    doc = _open(read_path)
    try:
        out: list[dict] = []
        for idx, item in enumerate(items):
            page = item.get("page")
            if not page:
                raise ValueError(f"items[{idx}] missing 'page'")
            pg = _get_page(doc, page)
            annot = _create_one(pg, item, s)
            out.append({"id": annot.xref, "type": item["type"],
                        "page": page})
        saved = _save(doc, write_path, mode)
        return {"saved_to": saved, "count": len(out), "annotations": out}
    finally:
        doc.close()


@_internal
def update_annotations(
    file: str,
    updates: list[dict],
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Update many annotations in one save cycle. Each update is a dict with
    `id` (xref, from list_annotations or add_*) plus any of:
    content, color, fill, opacity, author.

    Returns: {saved_to, updated: [ids], missing: [ids]}
        `missing` lists ids that weren't found anywhere in the document.
    """
    if not updates:
        return {"saved_to": str(_resolve_file(file)), "updated": [],
                "missing": []}
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        idx: dict[int, dict] = {}
        for u in updates:
            if "id" not in u:
                raise ValueError(f"update missing 'id': {u}")
            idx[int(u["id"])] = u
        updated: list[int] = []
        for i in range(doc.page_count):
            for a in doc[i].annots() or []:
                if a.xref in idx:
                    u = idx[a.xref]
                    _set_common(a, content=u.get("content"),
                                author=u.get("author"),
                                color=u.get("color"), fill=u.get("fill"),
                                opacity=u.get("opacity"))
                    updated.append(a.xref)
        missing = sorted(k for k in idx if k not in updated)
        saved = _save(doc, write_path, mode)
        return {"saved_to": saved, "updated": updated, "missing": missing}
    finally:
        doc.close()


@_internal
def delete_annotations(
    file: str,
    ids: Optional[list[int]] = None,
    authors: Optional[list[str]] = None,
    exclude_authors: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
    pages: Optional[list[int]] = None,
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Delete annotations either by id list (precise) or by criteria (bulk).
    All filters combine with AND.

    Examples:
        delete_annotations(file, ids=[42, 57])              # precise
        delete_annotations(file, authors=["Tan"], pages=[3])# Tan's marks on p3
        delete_annotations(file, authors=["AutoCAD SHX Text"])  # clear CAD noise

    Refusing to run with no filters is intentional — pass at least one of:
    `ids`, `authors`, `exclude_authors`, `types`, `exclude_types`, `pages`.

    Returns: {saved_to, deleted: [ids], missing: [ids]}
        `missing` is only populated when `ids` was given.
    """
    if not any([ids, authors, exclude_authors, types, exclude_types, pages]):
        raise ValueError(
            "Refusing to delete every annotation. Pass at least one of: "
            "ids, authors, exclude_authors, types, exclude_types, pages."
        )
    read_path, write_path, mode = _resolve_write(file, in_place, output_path)
    doc = _open(read_path)
    try:
        id_set = {int(x) for x in ids} if ids else None
        keep_authors = set(authors) if authors else None
        ex_authors = set(exclude_authors or [])
        keep_types = set(types) if types else None
        ex_types = set(exclude_types or [])
        page_set = set(pages) if pages else None

        deleted: list[int] = []
        for i in range(doc.page_count):
            page_num = i + 1
            if page_set is not None and page_num not in page_set:
                continue
            pg = doc[i]
            for a in list(pg.annots() or []):
                if id_set is not None:
                    if a.xref not in id_set:
                        continue
                else:
                    author = (a.info.get("title") or "")
                    sub = (a.type[1] if a.type else "") or ""
                    if keep_authors is not None and author not in keep_authors:
                        continue
                    if author in ex_authors:
                        continue
                    if keep_types is not None and sub not in keep_types:
                        continue
                    if sub in ex_types:
                        continue
                xref = a.xref
                pg.delete_annot(a)
                deleted.append(xref)
        missing = sorted(id_set - set(deleted)) if id_set is not None else []
        saved = _save(doc, write_path, mode)
        return {"saved_to": saved, "deleted": deleted, "missing": missing}
    finally:
        doc.close()


# --- Export / flatten ------------------------------------------------------

@mcp.tool()
def export_annotations(
    file: str,
    format: str = "json",
    output_path: Optional[str] = None,
    group_by: str = "page",
    only_review: bool = False,
    authors: Optional[list[str]] = None,
    exclude_authors: Optional[list[str]] = None,
) -> dict:
    """
    Export annotations from the PDF in the requested format.

    Formats:
      - "json" / "csv" — machine-readable, one row per annotation.
      - "markdown" / "text" — human-readable summary suitable for emailing
        a redline list to a drafter. Groups by page or author and
        resolves IRT reply chains.

    Args:
        format: json | csv | markdown | text.
        output_path: If given, write to file and return its path. Otherwise
            return the rendered string inline.
        group_by: page | author | none (markdown/text only).
        only_review: Hide imported CAD text (AutoCAD SHX Text).
        authors / exclude_authors: Whitelist / blacklist on the title field.
    """
    if format in ("markdown", "text"):
        s = summarize_annotations(file, group_by=group_by, format=format,
                                  only_review=only_review,
                                  authors=authors,
                                  exclude_authors=exclude_authors)
        if output_path:
            out = Path(output_path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(s["text"])
            return {"saved_to": str(out), "count": s["count"],
                    "format": format}
        return s

    if format not in ("json", "csv"):
        raise ValueError(f"format must be json|csv|markdown|text, got {format!r}")

    path = _resolve_file(file)
    doc = _open(path)
    try:
        ex = set(exclude_authors or [])
        if only_review:
            ex.add("AutoCAD SHX Text")
        keep = set(authors) if authors else None
        rows = []
        for i in range(doc.page_count):
            for a in doc[i].annots() or []:
                d = _annot_to_dict(a)
                au = d.get("author") or ""
                if au in ex:
                    continue
                if keep is not None and au not in keep:
                    continue
                rows.append(d)
        if format == "csv":
            import csv
            import io
            keys = ["id", "page", "type", "author", "content",
                    "color", "fill", "rect", "in_reply_to"]
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(r.get(k))
                                if isinstance(r.get(k), (list, dict))
                                else r.get(k))
                            for k in keys})
            data = buf.getvalue()
        else:
            data = json.dumps(rows, indent=2)
        if output_path:
            out = Path(output_path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(data)
            return {"saved_to": str(out), "count": len(rows),
                    "format": format}
        return {"count": len(rows), "data": data, "format": format}
    finally:
        doc.close()


@mcp.tool()
def flatten_annotations(
    file: str,
    output_path: str,
    dpi: int = 200,
    keep_authors: Optional[list[str]] = None,
    drop_authors: Optional[list[str]] = None,
) -> dict:
    """
    Render every page with annotations baked into the content stream and
    save as a new PDF. Useful when sending markups to recipients who
    shouldn't be able to remove them.

    Optional author filtering produces a per-reviewer flatten — pass
    `keep_authors=["Tan"]` to bake only Tan's redlines (others are
    stripped before rendering); pass `drop_authors=["AutoCAD SHX Text"]`
    to suppress CAD-text noise from the flattened output. The source
    file is never modified — filtering happens on an in-memory copy.
    """
    import io
    src = _resolve_file(file)
    if keep_authors or drop_authors:
        # Work on an in-memory copy so the source isn't touched
        with src.open("rb") as f:
            buf = io.BytesIO(f.read())
        doc = fitz.open(stream=buf.getvalue(), filetype="pdf")
        keep = set(keep_authors) if keep_authors else None
        drop = set(drop_authors or [])
        for i in range(doc.page_count):
            pg = doc[i]
            for a in list(pg.annots() or []):
                author = a.info.get("title") or ""
                if (keep is not None and author not in keep) or author in drop:
                    pg.delete_annot(a)
    else:
        doc = _open(src)
    try:
        out_doc = fitz.open()
        for i in range(doc.page_count):
            pg = doc[i]
            pix = pg.get_pixmap(dpi=dpi, annots=True)
            new_page = out_doc.new_page(width=pg.rect.width,
                                        height=pg.rect.height)
            new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(out))
        out_doc.close()
        return {"saved_to": str(out), "pages": doc.page_count,
                "filtered": bool(keep_authors or drop_authors)}
    finally:
        doc.close()


@_internal
def summarize_annotations(
    file: str,
    group_by: str = "page",
    format: str = "markdown",
    only_review: bool = True,
    authors: Optional[list[str]] = None,
    exclude_authors: Optional[list[str]] = None,
) -> dict:
    """
    Return a human-readable summary of all annotations in the PDF.
    Useful for emailing a redline list to a drafter.

    Args:
        group_by: "page" (default) | "author" | "none".
        format: "markdown" (default) | "text".
        only_review: Hide imported CAD text (AutoCAD SHX Text). Default True.
        authors / exclude_authors: Whitelist / blacklist on the title field.

    Returns: {format, group_by, count, text} where `text` is the
        rendered summary. Each entry includes id, page, type, author,
        content, and any reply text (IRT chains).
    """
    if group_by not in ("page", "author", "none"):
        raise ValueError("group_by must be page|author|none")
    if format not in ("markdown", "text"):
        raise ValueError("format must be markdown|text")

    path = _resolve_file(file)
    doc = _open(path)
    try:
        ex = set(exclude_authors or [])
        if only_review:
            ex.add("AutoCAD SHX Text")
        keep = set(authors) if authors else None

        # Collect, then resolve reply chains
        items: list[dict] = []
        replies_by_parent: dict[int, list[dict]] = {}
        for i in range(doc.page_count):
            for a in doc[i].annots() or []:
                d = _annot_to_dict(a)
                if (d.get("author") or "") in ex:
                    continue
                if keep is not None and (d.get("author") or "") not in keep:
                    continue
                irt = d.get("in_reply_to")
                if irt:
                    replies_by_parent.setdefault(irt, []).append(d)
                else:
                    items.append(d)

        for p in items:
            p["replies"] = replies_by_parent.get(p["id"], [])

        # Render
        md = format == "markdown"
        out = []
        if group_by == "page":
            by_page: dict[int, list[dict]] = {}
            for it in items:
                by_page.setdefault(it["page"], []).append(it)
            for pn in sorted(by_page):
                hdr = f"## Page {pn}" if md else f"Page {pn}"
                out.append(hdr)
                for it in by_page[pn]:
                    out.append(_fmt_annot(it, md))
                out.append("")
        elif group_by == "author":
            by_auth: dict[str, list[dict]] = {}
            for it in items:
                by_auth.setdefault(it.get("author") or "(unknown)",
                                   []).append(it)
            for au in sorted(by_auth):
                hdr = f"## {au}" if md else f"{au}:"
                out.append(hdr)
                for it in sorted(by_auth[au], key=lambda x: x["page"]):
                    out.append(_fmt_annot(it, md, show_page=True))
                out.append("")
        else:
            for it in items:
                out.append(_fmt_annot(it, md, show_page=True))

        text = "\n".join(out).rstrip()
        return {"format": format, "group_by": group_by,
                "count": len(items), "text": text}
    finally:
        doc.close()


def _fmt_annot(it: dict, md: bool, show_page: bool = False) -> str:
    bullet = "- " if md else "  - "
    parts = []
    if show_page:
        parts.append(f"p.{it['page']}")
    parts.append(it.get("type") or "")
    author = it.get("author")
    if author:
        parts.append(f"by {author}")
    parts.append(f"#{it['id']}")
    head = f"{bullet}**{' | '.join(parts)}**" if md \
        else f"{bullet}{' | '.join(parts)}"
    body = it.get("content") or ""
    body = body.replace("\n", " ").strip()
    line = f"{head}: {body}" if body else head
    lines = [line]
    for r in it.get("replies") or []:
        ra = r.get("author") or ""
        rc = (r.get("content") or "").replace("\n", " ").strip()
        prefix = "    > " if md else "      > "
        lines.append(f"{prefix}**{ra}**: {rc}" if md
                     else f"{prefix}{ra}: {rc}")
    return "\n".join(lines)


# --- Binary I/O (for callers without filesystem access) -------------------

@mcp.tool()
def load_pdf(
    data: str,
    name: Optional[str] = None,
    append_to: Optional[str] = None,
) -> dict:
    """
    Decode base64 PDF data and write it to a temp file. Returns a path
    that other tools accept as the `file` argument.

    Use this when the caller can't reach a local file (e.g. a PDF attached
    in a Claude Desktop chat). Pair with `dump_pdf` to retrieve the
    annotated result as base64.

    For PDFs that don't fit in a single tool call, send the first chunk
    with no `append_to`, then subsequent chunks with `append_to=<the path
    returned by the previous call>`.

    Args:
        data: Base64-encoded PDF bytes (or one chunk thereof).
        name: Optional filename hint for the temp path.
        append_to: If set, append to this existing path instead of creating
            a new temp file. Use this for chunked uploads.

    Returns: {file, size_bytes, pages | None}
        `pages` is None if the upload appears incomplete.
    """
    import base64
    import tempfile

    raw = base64.b64decode(data, validate=False)
    if append_to:
        out = Path(append_to).expanduser().resolve()
        if not out.exists():
            raise FileNotFoundError(f"append_to: {out} does not exist")
        with out.open("ab") as f:
            f.write(raw)
    else:
        if raw[:5] not in (b"%PDF-", b"%PDF\r"):
            raise ValueError(
                "Decoded data does not start with a %PDF header. "
                "Pass the first chunk without append_to, then subsequent "
                "chunks with append_to set."
            )
        stem = Path(name).stem if name else "uploaded"
        fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix=f"{stem}-")
        os.close(fd)
        out = Path(tmp)
        out.write_bytes(raw)

    pages: Optional[int]
    try:
        d = fitz.open(str(out))
        pages = d.page_count
        d.close()
    except Exception:
        pages = None  # likely an incomplete upload — caller should chunk more
    return {
        "file": str(out),
        "size_bytes": out.stat().st_size,
        "pages": pages,
    }


@mcp.tool()
def dump_pdf(
    file: str,
    chunk: int = 0,
    chunk_size_kb: int = 600,
) -> dict:
    """
    Return the file's contents as base64. Pair with `load_pdf` when the
    caller needs the annotated result as data instead of a path.

    Most PDFs exceed Claude Desktop's 1 MB tool-result cap once base64
    inflation is applied (~33%). Default `chunk_size_kb=600` keeps each
    response well under the cap; call repeatedly with chunk=0,1,...,
    until `chunk == total_chunks - 1`, then concatenate the `data` fields
    in order and base64-decode.

    Args:
        chunk: 0-indexed chunk number to return.
        chunk_size_kb: Raw bytes per chunk before base64 encoding.

    Returns: {file, size_bytes, chunk, total_chunks, data}
    """
    import base64
    path = _resolve_file(file)
    raw = path.read_bytes()
    chunk_size = max(1, chunk_size_kb) * 1024
    total = max(1, (len(raw) + chunk_size - 1) // chunk_size)
    if chunk < 0 or chunk >= total:
        raise ValueError(
            f"chunk {chunk} out of range (valid: 0..{total - 1})"
        )
    start = chunk * chunk_size
    piece = raw[start:start + chunk_size]
    return {
        "file": str(path),
        "size_bytes": len(raw),
        "chunk": chunk,
        "total_chunks": total,
        "data": base64.b64encode(piece).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# Unified dispatch tools (the public MCP surface)
# ---------------------------------------------------------------------------

@mcp.tool()
def inspect(
    target: Optional[str] = None,
    kind: str = "annotations",
    page: Optional[int] = None,
    query: Optional[str] = None,
    quads: bool = False,
    authors: Optional[list[str]] = None,
    exclude_authors: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
    only_review: bool = False,
    limit: Optional[int] = None,
) -> Any:
    """
    Read tool — discovery, page info, annotation listing, text search.
    `kind` selects what to inspect:

    - "projects" — return configured project shortcuts. `target` ignored.
    - "pdfs" (target=folder) — list PDFs under a project name or folder.
    - "pages" (target=file) — page count + dimensions for every page.
    - "annotations" (target=file) — read annotations with filters
        (page, authors, exclude_authors, types, exclude_types,
        only_review, limit). only_review hides "AutoCAD SHX Text"
        imported from CAD drawings.
    - "text" (target=file, query=...) — find_text occurrences. page
        optional; quads=True returns multi-line quads instead of rects.
    - "blocks" (target=file, page=...) — every text block with bbox
        and content. Useful for structural reasoning.

    Returns shape varies by kind: list[dict] for most; dict for
    "projects".
    """
    if kind == "projects":
        return list_projects()
    if kind == "pdfs":
        if not target:
            raise ValueError("kind='pdfs' requires target=<folder or project>")
        return list_pdfs(target)
    if not target:
        raise ValueError(f"kind={kind!r} requires target=<file path>")
    if kind == "pages":
        return page_info(target)
    if kind == "annotations":
        return list_annotations(
            target, page=page, authors=authors,
            exclude_authors=exclude_authors, types=types,
            exclude_types=exclude_types, only_review=only_review,
            limit=limit,
        )
    if kind == "text":
        if not query:
            raise ValueError("kind='text' requires query=<string>")
        if page is None:
            raise ValueError("kind='text' requires page=<1-indexed>")
        return find_text(target, page=page, query=query, quads=quads)
    if kind == "blocks":
        if page is None:
            raise ValueError("kind='blocks' requires page=<1-indexed>")
        return get_text_blocks(target, page=page)
    raise ValueError(
        f"kind must be projects|pdfs|pages|annotations|text|blocks, got {kind!r}"
    )


@mcp.tool()
def render(
    file: str,
    page: int,
    mode: str = "page",
    dpi: int = 120,
    with_annotations: bool = True,
    overlay_grid: bool = False,
    rect: Optional[list[float]] = None,
    grid_coords: bool = False,
    row: int = 0,
    col: int = 0,
    rows: int = 2,
    cols: int = 2,
    overlap: float = 20.0,
    max_bytes: int = MAX_IMAGE_BYTES,
    format: str = "auto",
) -> list:
    """
    Render a page (or part of one) as an image. `mode` selects the
    flavour:

    - "page" — full page. Set overlay_grid=True for a 50-pt coordinate
        grid (labels: "x=100pt (g2)" — both PDF-pt and grid-cell index).
        Auto-downsamples (PNG → JPEG → drop DPI) to stay under
        max_bytes (default 900 KB).
    - "region" (rect=[x0,y0,x1,y1]) — zoom into one region. Use when
        verifying placement on a large sheet without hitting the byte
        cap. dpi default 120 here too; bump to 200+ for fine detail.
    - "tile" (row, col, rows, cols) — render one cell of a grid covering
        the page. Use on A1/A0 sheets where a full render must drop
        DPI below readability. Tiles overlap by `overlap` pt so text
        cut at a tile edge is still readable in at least one tile.

    Common args:
        dpi: Render resolution. Coordinate space is page-relative
            (DPI-agnostic) — only the pixel size of the image changes.
        format: "auto" (PNG, fall back to JPEG) | "png" | "jpeg".
        max_bytes: Override the size budget. Renderer keeps shrinking
            until output fits.
        grid_coords: For mode="region", interpret rect in 50-pt grid
            units instead of PDF points.

    Returns [Image, info] where info reports actual_dpi, requested_dpi,
    format, size_bytes, downscaled, page_size_pt, plus mode-specific
    fields (clip_pt for region, tile + grid for tile).
    """
    if mode == "page":
        return render_page(
            file, page, dpi=dpi, with_annotations=with_annotations,
            overlay_grid=overlay_grid, max_bytes=max_bytes, format=format,
        )
    if mode == "region":
        if rect is None:
            raise ValueError("mode='region' requires rect=[x0,y0,x1,y1]")
        return render_region(
            file, page, rect, dpi=dpi if dpi != 120 else 200,
            with_annotations=with_annotations, grid_coords=grid_coords,
            max_bytes=max_bytes, format=format,
        )
    if mode == "tile":
        return render_tile(
            file, page, row=row, col=col, rows=rows, cols=cols,
            dpi=dpi if dpi != 120 else 150,
            with_annotations=with_annotations, overlap=overlap,
            max_bytes=max_bytes, format=format,
        )
    raise ValueError(f"mode must be page|region|tile, got {mode!r}")


@mcp.tool()
def annotate(
    file: str,
    items: list[dict],
    output_path: Optional[str] = None,
    in_place: bool = False,
    grid_coords: bool = False,
) -> dict:
    """
    Create annotations. The same tool covers every annotation type, in
    single or batch form, with location given either by coordinates or
    by text query.

    Each item is a dict. Common fields:
        type:  rectangle | circle | polygon | line | arrow | highlight |
               underline | strikeout | squiggly | sticky_note | freetext |
               callout | ink | stamp
        page:  1-indexed page (required for coord-anchored items;
               optional for text-anchored — defaults to all pages)

    Two ways to specify location:

    A) **Coordinates** (PDF points, top-left origin; or 50-pt grid
       units when grid_coords=True at call level):
         rectangle/circle/stamp:  rect=[x0,y0,x1,y1]
         polygon:                 points=[[x,y],...]
         line/arrow:              start=[x,y], end=[x,y]
         highlight/underline/strikeout/squiggly:  rect=[x0,y0,x1,y1]
                                                  or quads=[[x0,y0,x1,y1],...]
         sticky_note:             point=[x,y], text="..."
         freetext:                rect=[x0,y0,x1,y1], text="..."
         callout:                 callout_point=[x,y], knee_point=[x,y],
                                  text_rect=[x0,y0,x1,y1], text="..."
         ink:                     strokes=[[[x,y],...], ...]

    B) **Text-anchored** (no coordinates needed; available for markup +
       sticky_note + callout). Add `query` to the item:
         {type:"highlight", query:"VACCUM", content:"typo"}
         {type:"sticky_note", query:"Clause 7.4", text:"see HPCL spec",
          placement:"after"}            # after|before|above|below|overlay
         {type:"callout", query:"V-anchor", text:"check spacing",
          placement:"auto"}             # auto|right|left|above|below
       Optional `occurrence` (0-indexed) selects a specific match;
       omit to mark every match (highlight) or first match (sticky/callout).

    Common per-item options: content, color, fill, opacity, author,
    border_width (shapes), cloud_intensity (shapes), line_start/line_end,
    icon (sticky), font_size/align (freetext), stamp (stamp).

    Save options (call-level): output_path, in_place, grid_coords.

    Returns: {saved_to, count, annotations: [{id, type, page, ...}, ...]}.
        Each result includes the xref `id` for later modify() calls.
    """
    if not items:
        raise ValueError("items must contain at least one annotation spec")
    # Split text-anchored from coord-anchored
    text_items = []
    coord_items = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"items[{i}] must be a dict")
        if "query" in it:
            text_items.append(it)
        else:
            coord_items.append(it)

    saved = None
    out_results: list[dict] = []

    if coord_items:
        r = add_annotations(file, items=coord_items,
                            output_path=output_path, in_place=in_place,
                            grid_coords=grid_coords)
        saved = r["saved_to"]
        out_results.extend(r["annotations"])

    # After the first call, subsequent saves should accumulate on the
    # derived/in-place file already established above.
    next_inplace = in_place
    next_output = output_path
    if coord_items and saved and saved != str(_resolve_file(file)):
        # Switch to editing the derived working file in place
        file_for_text = saved
        next_output = None
        next_inplace = False  # _resolve_write detects .annotated.pdf and goes incremental
    else:
        file_for_text = file

    for it in text_items:
        kind = it["type"]
        query = it["query"]
        page = it.get("page")
        occurrence = it.get("occurrence",
                            None if kind in ("highlight", "underline",
                                             "strikeout", "squiggly") else 0)
        author = it.get("author")
        content = it.get("content")
        color = it.get("color")

        if kind in ("highlight", "underline", "strikeout", "squiggly"):
            r = highlight_text(
                file_for_text, query, page=page, occurrence=occurrence,
                kind=kind, color=color, content=content, author=author,
                output_path=next_output, in_place=next_inplace,
            )
            saved = r["saved_to"]
            for m in r["matches"]:
                out_results.append({
                    "id": m["id"], "type": kind, "page": m["page"],
                    "rect": m["rect"], "anchor": "text",
                })
        elif kind == "sticky_note":
            text = it.get("text") or content
            if text is None:
                raise ValueError("sticky_note item needs 'text' or 'content'")
            r = add_sticky_note_at_text(
                file_for_text, query, text, page=page, occurrence=occurrence,
                icon=it.get("icon", "Note"),
                placement=it.get("placement", "after"),
                offset=it.get("offset", 6),
                color=color or "#ffff00",
                author=author,
                output_path=next_output, in_place=next_inplace,
            )
            saved = r["saved_to"]
            for n in r["notes"]:
                out_results.append({
                    "id": n["id"], "type": kind, "page": n["page"],
                    "anchor_rect": n["anchor_rect"], "point": n["point"],
                    "anchor": "text",
                })
        elif kind == "callout":
            text = it.get("text") or content
            if text is None:
                raise ValueError("callout item needs 'text' or 'content'")
            r = add_callout_at_text(
                file_for_text, query, text, page=page, occurrence=occurrence,
                placement=it.get("placement", "auto"),
                color=color or "#ff0000",
                fill=it.get("fill", "#ffffe0"),
                text_color=it.get("text_color", "#000000"),
                font_size=it.get("font_size", 10),
                box_width=it.get("box_width", 220),
                box_height=it.get("box_height", 60),
                author=author,
                output_path=next_output, in_place=next_inplace,
            )
            saved = r["saved_to"]
            for c in r["callouts"]:
                out_results.append({
                    "id": c["id"], "type": kind, "page": c["page"],
                    "anchor_rect": c["anchor_rect"],
                    "callout_point": c["callout_point"],
                    "knee_point": c["knee_point"],
                    "text_rect": c["text_rect"],
                    "anchor": "text",
                })
        else:
            raise ValueError(
                f"text-anchored placement (with `query`) is supported for "
                f"highlight, underline, strikeout, squiggly, sticky_note, "
                f"and callout. Got type={kind!r}; pass coordinates instead."
            )
        # After the first text-anchored save, keep editing the derived file
        if saved:
            file_for_text = saved
            next_output = None
            next_inplace = False

    if saved is None:
        saved = str(_resolve_file(file))
    return {"saved_to": saved, "count": len(out_results),
            "annotations": out_results}


@mcp.tool()
def modify(
    file: str,
    action: str,
    # update
    updates: Optional[list[dict]] = None,
    # delete
    ids: Optional[list[int]] = None,
    authors: Optional[list[str]] = None,
    exclude_authors: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
    pages: Optional[list[int]] = None,
    # reply
    annot_id: Optional[int] = None,
    text: Optional[str] = None,
    author: Optional[str] = None,
    # save
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> dict:
    """
    Edit existing annotations. `action` selects the operation:

    - "update" — change properties of one or many annotations.
        updates=[{id, content?, color?, fill?, opacity?, author?}, ...].
        Returns {saved_to, updated:[ids], missing:[ids]}.

    - "delete" — remove annotations either by `ids` (precise) or by
        criteria (`authors`, `exclude_authors`, `types`, `exclude_types`,
        `pages`; AND-combined). Refuses to run with no filters at all.
        Returns {saved_to, deleted:[ids], missing:[ids]}.

    - "reply" — add an Acrobat-style reply (IRT thread) to an existing
        annotation. annot_id=<xref>, text="...", author="..." optional.
        Returns {saved_to, id, in_reply_to}.

    Save options: output_path, in_place. By default a sibling
    `<source>.annotated.pdf` is updated incrementally.
    """
    if action == "update":
        if not updates:
            raise ValueError("action='update' requires updates=[{id, ...}]")
        return update_annotations(file, updates=updates,
                                   output_path=output_path,
                                   in_place=in_place)
    if action == "delete":
        return delete_annotations(
            file, ids=ids, authors=authors, exclude_authors=exclude_authors,
            types=types, exclude_types=exclude_types, pages=pages,
            output_path=output_path, in_place=in_place,
        )
    if action == "reply":
        if annot_id is None or text is None:
            raise ValueError(
                "action='reply' requires annot_id=<xref> and text=<string>"
            )
        return reply_to_annotation(file, annot_id=annot_id, text=text,
                                    author=author, output_path=output_path,
                                    in_place=in_place)
    raise ValueError(f"action must be update|delete|reply, got {action!r}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
