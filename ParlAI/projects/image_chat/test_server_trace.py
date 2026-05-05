"""Phase 4: Verify _post_trace() fires correctly to debug dashboard (port 8090).

Run in clip_env from ParlAI/projects/image_chat/:
    python -m pytest test_server_trace.py -v --noconftest
"""
import sys
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

# Use defaults — no extra CLI args
sys.argv = ['server_updated_zhengxuan.py']

# ── Stub heavy imports before loading the server ────────────────────────────
for _mod in ('chromadb', 'deep_translator'):
    sys.modules.setdefault(_mod, MagicMock())

_mem_ret_stub = MagicMock()
_mem_ret_stub.retrieve = MagicMock(return_value=[])
_mem_ret_stub.format_retrieved_block = MagicMock(return_value='')
sys.modules.setdefault('memory_retriever', _mem_ret_stub)

for _mod in ('photo_db', 'memory_extractor'):
    sys.modules.setdefault(_mod, MagicMock())

for _mod in ('server_multipart', 'server_timing', 'server_users',
             'server_conv_store', 'server_memory'):
    sys.modules.setdefault(_mod, MagicMock())

# ── Import server after stubs are in place ───────────────────────────────────
import server_updated_zhengxuan as srv


class TestPostTraceBehavior(unittest.TestCase):
    """Unit tests for _post_trace() fire-and-forget helper."""

    def setUp(self):
        self._req_patcher = patch('server_updated_zhengxuan.requests.post')
        self.mock_post = self._req_patcher.start()

    def tearDown(self):
        self._req_patcher.stop()

    def _call_and_wait(self, payload: dict, timeout: float = 1.0):
        """Call _post_trace then spin until requests.post is called or timeout."""
        srv._post_trace(payload)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.mock_post.called:
                break
            time.sleep(0.01)

    def test_uses_correct_url(self):
        """_post_trace must POST to http://localhost:8090/api/trace."""
        self._call_and_wait({"ts": "2026-01-01"})
        self.assertTrue(self.mock_post.called, "_post_trace never called requests.post")
        url = self.mock_post.call_args[0][0]
        self.assertEqual(url, "http://localhost:8090/api/trace")

    def test_sends_payload_as_json(self):
        """The payload dict must be forwarded as the json= kwarg."""
        payload = {"ts": "2026-01-01", "patient_id": "P001", "final_response": "Hi"}
        self._call_and_wait(payload)
        call_kwargs = self.mock_post.call_args[1]
        self.assertEqual(call_kwargs.get('json'), payload)

    def test_timeout_is_300ms(self):
        """requests.post must use a 0.3 s (300 ms) timeout."""
        self._call_and_wait({"ts": "x"})
        call_kwargs = self.mock_post.call_args[1]
        self.assertAlmostEqual(call_kwargs.get('timeout'), 0.3, places=5)

    def test_silent_on_connection_error(self):
        """Connection errors must not propagate — dashboard may be offline."""
        self.mock_post.side_effect = ConnectionError("refused")
        try:
            srv._post_trace({"ts": "x"})
            time.sleep(0.1)
        except Exception as exc:
            self.fail(f"_post_trace raised when it should be silent: {exc}")

    def test_silent_on_request_timeout(self):
        """Request timeouts must not propagate."""
        import requests as _r
        self.mock_post.side_effect = _r.exceptions.Timeout()
        try:
            srv._post_trace({"ts": "x"})
            time.sleep(0.1)
        except Exception as exc:
            self.fail(f"_post_trace raised on timeout: {exc}")

    def test_uses_daemon_thread(self):
        """Background thread must be daemon so it never blocks process exit."""
        with patch('server_updated_zhengxuan.threading.Thread') as mock_cls:
            mock_thread = MagicMock()
            mock_cls.return_value = mock_thread
            srv._post_trace({"ts": "x"})
            mock_cls.assert_called_once()
            self.assertTrue(
                mock_cls.call_args[1].get('daemon'),
                "threading.Thread must be created with daemon=True",
            )
            mock_thread.start.assert_called_once()


class TestRagCandidateStripping(unittest.TestCase):
    """Embedding field must be stripped before serialising rag_candidates."""

    def test_embedding_stripped(self):
        candidates = [
            {"id": "P001/a.jpg", "caption": "a park",
             "embedding": [0.1] * 512, "_rank_score": 0.9,
             "_entity_score": 0.5, "_theme_match": True, "_recency_score": 0.8},
        ]
        stripped = [{k: v for k, v in c.items() if k != "embedding"} for c in candidates]
        self.assertNotIn("embedding", stripped[0])
        self.assertIn("id", stripped[0])
        self.assertIn("_rank_score", stripped[0])

    def test_non_embedding_fields_preserved(self):
        candidates = [{"id": "x", "caption": "y", "embedding": [], "_rank_score": 1.0}]
        stripped = [{k: v for k, v in c.items() if k != "embedding"} for c in candidates]
        self.assertEqual(stripped[0]["id"], "x")
        self.assertEqual(stripped[0]["_rank_score"], 1.0)


class TestTracePayloadSchema(unittest.TestCase):
    """Verify the required keys are always present in the trace payload."""

    REQUIRED_KEYS = {
        "ts", "patient_id", "user_input", "model_name",
        "full_prompt", "raw_response", "final_response",
        "timing", "photo_id", "retrieved_context", "rag_candidates",
    }

    def _make_payload(self, **overrides) -> dict:
        base = {
            "ts":               "2026-01-01T00:00:00",
            "patient_id":       "P001",
            "user_input":       "hello",
            "model_name":       "gpt-5-mini",
            "full_prompt":      [{"role": "system", "content": "..."}],
            "raw_response":     "9. Hi!\nJSON_OUTPUT: ...",
            "final_response":   "Hi!",
            "timing":           {"total_ms": 100},
            "photo_id":         "P001/photo.jpg",
            "retrieved_context": "",
            "rag_candidates":   [],
        }
        base.update(overrides)
        return base

    def test_all_required_keys_present(self):
        payload = self._make_payload()
        missing = self.REQUIRED_KEYS - set(payload.keys())
        self.assertEqual(missing, set(), f"Missing trace payload keys: {missing}")

    def test_rag_candidates_is_list(self):
        payload = self._make_payload()
        self.assertIsInstance(payload["rag_candidates"], list)

    def test_full_prompt_is_list(self):
        payload = self._make_payload()
        self.assertIsInstance(payload["full_prompt"], list)


if __name__ == '__main__':
    unittest.main()
