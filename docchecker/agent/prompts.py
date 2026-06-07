"""System-prompt construction for the checking agent.

The stable base instructions live in an editable file (``CHECKER_PROMPT_FILE``,
default ``checker_prompt.md`` in the project root) so they can be tuned without
touching code. ``build_system_prompt`` returns the base followed by the run's
metadata and free-text guiding prompt.

For prompt caching, the base instructions are the stable prefix; the per-run
metadata + guiding prompt are appended after and kept small.
"""
from __future__ import annotations

import os

from . import schema  # noqa: F401 (kept for type clarity)
from .schema import CheckContext

# Built-in fallback if the editable file is missing.
_DEFAULT_BASE = """You are a senior EPC document checker. Compare a SUBMITTED document
against REFERENCE documents (tender / PO / PR) and report concrete, defensible findings.
Cite every finding to a reference requirement (doc + page + verbatim quote) and to an exact
submitted-doc location (page + a verbatim anchor copied from that page). Never speculate.
Categories: compliance, completeness, consistency, correctness, bom, dimension, deviation.
Severity: critical, major, minor, observation.

# Context and applicability are decisive
Before you raise a finding, establish WHAT a reference requirement actually governs, and apply
it ONLY to the items it covers. Engineering requirements are scoped, not universal:
- A requirement stated for one item, system or service does NOT automatically apply to another.
  Stringent tests or materials specified for, say, heater-coil pipes inside a fired heater do not
  govern external utility piping; a grade required for dampers does not govern on-off valves;
  a thickness for one line class does not govern another. Check the clause's stated scope first.
- BUT honour genuinely general requirements: when the tender states a requirement for "all
  piping", "every component of this type", or a whole class/service, it DOES apply across that
  class unless a more specific clause overrides it. Apply the stated order of precedence; if a
  general and a specific clause conflict, cite both and flag it rather than silently choosing.
- Read clause and document context, not keywords. The same term means different things by
  context: a pipe "bend" (a fitting) is not a road "bend", and neither is a "bend test" on a
  plate or weld coupon. Match a submitted item to a requirement only when the subject, service,
  size/class and component type genuinely correspond.
- State what a requirement governs before asserting non-compliance. If you are not sure the
  requirement applies to the submitted item, say so and lower the finding's confidence rather
  than asserting a deviation that may be out of scope."""

_PROMPT_FILE_ENV = "CHECKER_PROMPT_FILE"


def base_instructions() -> str:
    path = os.environ.get(_PROMPT_FILE_ENV) or _default_prompt_path()
    try:
        if path and os.path.exists(path):
            text = open(path, encoding="utf-8").read().strip()
            if text:
                return text
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_BASE


def _default_prompt_path() -> str:
    # checker/ -> project root /checker_prompt.md
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "checker_prompt.md")


def build_system_prompt(ctx: CheckContext) -> str:
    """Base (stable, cacheable) + run metadata + guiding prompt (variable)."""
    parts = [base_instructions()]

    meta = ctx.metadata or {}
    parts.append(
        "# This run\n"
        f"- Project number: {meta.get('project_number')}\n"
        f"- Document type: {meta.get('document_type')}\n"
        f"- Prepared by: {meta.get('originator')}"
    )

    if ctx.guiding_prompt:
        parts.append(
            "# Reviewer instructions for this run (take precedence on conflict)\n"
            + ctx.guiding_prompt.strip()
        )

    return "\n\n".join(parts)
