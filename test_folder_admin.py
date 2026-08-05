"""Authorization regression tests for destructive folder deletion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web
from docchecker import auth


class FolderDeletionAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.old_db_path = web.DB_PATH
        self.old_uploads_dir = web.UPLOADS_DIR
        web.DB_PATH = str(root / "docs.db")
        web.UPLOADS_DIR = str(root / "uploads")
        Path(web.UPLOADS_DIR, "Tender A").mkdir(parents=True)
        Path(web.UPLOADS_DIR, "Tender A", "source.pdf").write_bytes(b"test")
        conn = web.init_db(web.DB_PATH)
        conn.close()
        self.client = TestClient(web.app)
        self.user = {
            "id": 1,
            "oidc_sub": "alice@example.test",
            "email": "alice@example.test",
        }

    def tearDown(self):
        self.client.close()
        web.DB_PATH = self.old_db_path
        web.UPLOADS_DIR = self.old_uploads_dir
        self.temp_dir.cleanup()

    def test_non_admin_cannot_delete_folder(self):
        with (
            patch.object(auth, "require_user", return_value=self.user),
            patch.object(auth, "is_admin", return_value=False) as is_admin,
        ):
            response = self.client.delete("/api/folders/Tender%20A")

        self.assertEqual(response.status_code, 403)
        self.assertTrue(Path(web.UPLOADS_DIR, "Tender A", "source.pdf").exists())
        is_admin.assert_called_once_with("alice@example.test")

    def test_admin_can_delete_folder(self):
        with (
            patch.object(auth, "require_user", return_value=self.user),
            patch.object(auth, "is_admin", return_value=True) as is_admin,
        ):
            response = self.client.delete("/api/folders/Tender%20A")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "deleted")
        self.assertFalse(Path(web.UPLOADS_DIR, "Tender A").exists())
        is_admin.assert_called_once_with("alice@example.test")

    def test_non_admin_is_blocked_from_all_management_endpoints(self):
        cases = [
            ("post", "/api/folders", {"json": {"name": "Blocked"}}),
            ("patch", "/api/folders/Tender%20A", {"json": {"name": "Blocked"}}),
            ("delete", "/api/folders/Tender%20A", {}),
            ("get", "/api/folders/Tender%20A/documents", {}),
            ("post", "/api/folders/Tender%20A/upload", {
                "files": {"files": ("blocked.pdf", b"blocked", "application/pdf")},
            }),
            ("post", "/api/folders/Tender%20A/pending/source.pdf/discard", {"json": {}}),
            ("patch", "/api/documents/1", {"json": {"title": "Blocked"}}),
            ("post", "/api/documents/bulk-move", {
                "json": {"doc_ids": [1], "target_folder": "Tender A"},
            }),
            ("post", "/api/documents/bulk-delete", {"json": {"doc_ids": [1]}}),
            ("delete", "/api/documents/1", {}),
            ("post", "/api/folders/Tender%20A/ingest", {"json": {}}),
            ("post", "/api/folders/Tender%20A/ingest/source.pdf", {"json": {}}),
            ("get", "/api/ingestion/jobs", {}),
            ("post", "/api/ingestion/jobs/job-1/retry", {"json": {}}),
            ("post", "/api/ingestion/jobs/job-1/cancel", {"json": {}}),
            ("get", "/api/folders/Tender%20A/quality", {}),
            ("patch", "/api/quality/1/resolve", {"json": {}}),
        ]

        with (
            patch.object(auth, "require_user", return_value=self.user),
            patch.object(auth, "is_admin", return_value=False) as is_admin,
        ):
            for method, path, kwargs in cases:
                with self.subTest(method=method, path=path):
                    response = getattr(self.client, method)(path, **kwargs)
                    self.assertEqual(response.status_code, 403, response.text)

        self.assertEqual(is_admin.call_count, len(cases))
        self.assertTrue(Path(web.UPLOADS_DIR, "Tender A", "source.pdf").exists())
        self.assertFalse(Path(web.UPLOADS_DIR, "Blocked").exists())

    def test_non_admin_retains_read_only_knowledge_access(self):
        with (
            patch.object(auth, "require_user", return_value=self.user),
            patch.object(auth, "is_admin", return_value=False),
        ):
            folders = self.client.get("/api/folders")
            document = self.client.get("/api/documents/999999")

        self.assertEqual(folders.status_code, 200)
        # The read reached the document handler (and correctly found no record),
        # rather than being rejected by the management admin gate.
        self.assertEqual(document.status_code, 404)


if __name__ == "__main__":
    unittest.main()
