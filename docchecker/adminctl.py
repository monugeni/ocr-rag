"""CLI: manage document-checker admins (who sees Documents / Ingest / Jobs).

Run from the ocr-rag dir:

  ./venv/bin/python -m docchecker.adminctl grant alice@esteem.co.in bob@esteem.co.in
  ./venv/bin/python -m docchecker.adminctl revoke alice@esteem.co.in
  ./venv/bin/python -m docchecker.adminctl list

Grants take effect on the user's next page load (no restart). For Byom, the OAuth
subject is the email, so granting by email works whether or not they've logged in yet.
"""
from __future__ import annotations

import sqlite3
import sys

from . import config


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.CHECKER_DB, timeout=30.0)
    c.row_factory = sqlite3.Row
    return c


def _set(emails: list[str], value: int) -> None:
    c = _conn()
    try:
        for raw in emails:
            e = raw.strip().lower()
            if not e:
                continue
            c.execute(
                "INSERT INTO users (oidc_sub, email, is_admin) VALUES (?, ?, ?) "
                "ON CONFLICT(oidc_sub) DO UPDATE SET is_admin=excluded.is_admin, email=excluded.email",
                (e, e, value),
            )
            print(("granted admin:" if value else "revoked admin:"), e)
        c.commit()
    finally:
        c.close()


def _list() -> None:
    c = _conn()
    try:
        rows = c.execute(
            "SELECT email, oidc_sub FROM users WHERE is_admin = 1 ORDER BY email"
        ).fetchall()
        if not rows:
            print("(no admins)")
        for r in rows:
            print("  admin:", r["email"] or r["oidc_sub"])
    finally:
        c.close()


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("grant", "revoke", "list"):
        print(__doc__)
        return 1
    if argv[1] == "list":
        _list()
        return 0
    emails = argv[2:]
    if not emails:
        print("Provide at least one email.")
        return 1
    _set(emails, 1 if argv[1] == "grant" else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
