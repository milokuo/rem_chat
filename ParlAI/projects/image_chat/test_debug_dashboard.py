# -*- coding: utf-8 -*-
"""
Tests for debug_dashboard.py — Phase 2 TDD

Covers:
  POST /api/trace     — ingests trace payload, ring-buffer semantics
  GET  /api/turns     — returns stored turns newest-first
  GET  /              — serves the HTML page
  GET  /api/photo/<path:photo_path>
                      — serves real photos; blocks path traversal
"""

import json
import os
import sys
import tempfile
import unittest

# debug_dashboard is in the same directory
sys.path.insert(0, os.path.dirname(__file__))


class TestDebugDashboard(unittest.TestCase):

    def setUp(self):
        """Import module fresh (reloads ring buffer) and configure test uploads dir."""
        # We need a fresh app each test to isolate ring-buffer state.
        # The module exports `create_app(uploads_root)` for testability.
        import importlib
        import debug_dashboard as dd
        importlib.reload(dd)
        self.dd = dd

        self.tmpdir = tempfile.mkdtemp()
        self.app = dd.create_app(uploads_root=self.tmpdir)
        self.client = self.app.test_client()

    # ------------------------------------------------------------------
    # POST /api/trace
    # ------------------------------------------------------------------

    def test_post_trace_returns_200(self):
        payload = {"user_input": "hello", "final_response": "hi"}
        resp = self.client.post("/api/trace",
                                data=json.dumps(payload),
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200)

    def test_post_trace_non_json_returns_400(self):
        resp = self.client.post("/api/trace",
                                data="not-json",
                                content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_post_trace_stores_payload(self):
        payload = {"user_input": "hello", "final_response": "world"}
        self.client.post("/api/trace",
                         data=json.dumps(payload),
                         content_type="application/json")
        turns = json.loads(self.client.get("/api/turns").data)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["user_input"], "hello")

    def test_post_trace_ring_buffer_max_50(self):
        """Buffer must not exceed 50 entries."""
        for i in range(55):
            self.client.post("/api/trace",
                             data=json.dumps({"seq": i}),
                             content_type="application/json")
        turns = json.loads(self.client.get("/api/turns").data)
        self.assertEqual(len(turns), 50)

    def test_post_trace_oldest_dropped_when_full(self):
        """After 51 inserts the very first entry is gone."""
        for i in range(51):
            self.client.post("/api/trace",
                             data=json.dumps({"seq": i}),
                             content_type="application/json")
        turns = json.loads(self.client.get("/api/turns").data)
        seqs = [t["seq"] for t in turns]
        self.assertNotIn(0, seqs)   # first entry dropped
        self.assertIn(50, seqs)      # last entry present

    # ------------------------------------------------------------------
    # GET /api/turns
    # ------------------------------------------------------------------

    def test_get_turns_empty_returns_empty_list(self):
        resp = self.client.get("/api/turns")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(json.loads(resp.data), [])

    def test_get_turns_newest_first(self):
        """Most-recent trace must appear at index 0."""
        for i in range(3):
            self.client.post("/api/trace",
                             data=json.dumps({"seq": i}),
                             content_type="application/json")
        turns = json.loads(self.client.get("/api/turns").data)
        seqs = [t["seq"] for t in turns]
        self.assertEqual(seqs, [2, 1, 0])

    def test_get_turns_content_type_json(self):
        resp = self.client.get("/api/turns")
        self.assertIn("application/json", resp.content_type)

    # ------------------------------------------------------------------
    # GET /
    # ------------------------------------------------------------------

    def test_root_returns_200(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_root_returns_html(self):
        resp = self.client.get("/")
        self.assertIn(b"<html", resp.data.lower())

    # ------------------------------------------------------------------
    # GET /api/photo/<path:photo_path>
    # ------------------------------------------------------------------

    def _write_test_photo(self, relative_path: str, content: bytes = b"FAKEJPEG") -> str:
        """Write a fake photo inside self.tmpdir and return relative path."""
        full_path = os.path.join(self.tmpdir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(content)
        return relative_path

    def test_photo_serves_existing_file(self):
        rel = self._write_test_photo("P001/birthday.jpg")
        resp = self.client.get(f"/api/photo/{rel}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, b"FAKEJPEG")

    def test_photo_path_with_subdirectory(self):
        """Patient/filename paths (containing /) must work — requires <path:> converter."""
        rel = self._write_test_photo("patient_abc/event/photo.jpg")
        resp = self.client.get(f"/api/photo/{rel}")
        self.assertEqual(resp.status_code, 200)

    def test_photo_missing_file_returns_404(self):
        resp = self.client.get("/api/photo/P001/nonexistent.jpg")
        self.assertEqual(resp.status_code, 404)

    def test_photo_traversal_blocked_dotdot(self):
        """Path traversal via .. must be blocked with 403."""
        # Try to escape uploads dir
        resp = self.client.get("/api/photo/../../../etc/passwd")
        self.assertIn(resp.status_code, (403, 404))

    def test_photo_traversal_blocked_absolute(self):
        """Absolute paths (encoded) must not escape uploads root.
        Flask normalises %2F-encoded slashes, which may result in a 308 redirect
        (double-slash normalisation) before our handler runs — that is also safe."""
        resp = self.client.get("/api/photo/%2Fetc%2Fpasswd")
        self.assertIn(resp.status_code, (308, 403, 404))

    def test_photo_traversal_stays_within_uploads(self):
        """A legitimate file IS served, confirming the guard only blocks escapes."""
        rel = self._write_test_photo("P001/safe.jpg")
        resp = self.client.get(f"/api/photo/{rel}")
        self.assertEqual(resp.status_code, 200)

    # ------------------------------------------------------------------
    # Thread safety smoke test
    # ------------------------------------------------------------------

    def test_concurrent_posts_do_not_corrupt_buffer(self):
        """Multiple threads posting simultaneously must not raise exceptions."""
        import threading
        errors = []

        def post_trace(i):
            try:
                self.client.post("/api/trace",
                                 data=json.dumps({"seq": i}),
                                 content_type="application/json")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=post_trace, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        turns = json.loads(self.client.get("/api/turns").data)
        self.assertLessEqual(len(turns), 20)


if __name__ == "__main__":
    unittest.main()
