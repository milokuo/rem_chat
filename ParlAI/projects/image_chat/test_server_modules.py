# -*- coding: utf-8 -*-
"""
TDD tests for Phase 2 extracted server modules:
  - server_timing.py
  - server_users.py
  - server_conv_store.py

Run: python -m pytest test_server_modules.py -v
"""

import json
import os
import time
import tempfile
import pytest

# ---------------------------------------------------------------------------
# server_timing tests
# ---------------------------------------------------------------------------

class TestTimedWorker:
    def test_returns_result_and_elapsed(self):
        from server_timing import _timed_worker
        result, elapsed = _timed_worker(lambda: 42)
        assert result == 42
        assert isinstance(elapsed, int)
        assert elapsed >= 0

    def test_elapsed_reflects_sleep(self):
        from server_timing import _timed_worker
        _, elapsed = _timed_worker(time.sleep, 0.05)
        assert elapsed >= 40  # at least 40 ms

    def test_forwards_args(self):
        from server_timing import _timed_worker
        result, _ = _timed_worker(lambda a, b: a + b, 3, 4)
        assert result == 7


class TestLogTiming:
    def test_appends_json_line(self, tmp_path):
        from server_timing import _log_timing
        log_file = str(tmp_path / "timing.jsonl")
        _log_timing({"service": "clip", "ms": 120}, log_path=log_file)
        with open(log_file, encoding="utf-8") as f:
            record = json.loads(f.read().strip())
        assert record["service"] == "clip"
        assert record["ms"] == 120
        assert "ts" in record

    def test_appends_multiple_lines(self, tmp_path):
        from server_timing import _log_timing
        log_file = str(tmp_path / "timing.jsonl")
        _log_timing({"n": 1}, log_path=log_file)
        _log_timing({"n": 2}, log_path=log_file)
        lines = open(log_file, encoding="utf-8").read().strip().splitlines()
        assert len(lines) == 2

    def test_does_not_raise_on_bad_path(self):
        """If log_path is unwritable, silently swallow the error."""
        from server_timing import _log_timing
        _log_timing({"x": 1}, log_path="/nonexistent_dir/timing.jsonl")  # must not raise


# ---------------------------------------------------------------------------
# server_users tests
# ---------------------------------------------------------------------------

class TestUsersFile:
    def test_path_under_uploads_dir(self, tmp_path):
        from server_users import _users_file
        result = _users_file(str(tmp_path))
        assert result == str(tmp_path / "users.json")


class TestLoadUsers:
    def test_returns_default_when_no_file(self, tmp_path):
        from server_users import _load_users
        result = _load_users(str(tmp_path), default_user="default")
        assert result == ["default"]

    def test_loads_existing_file(self, tmp_path):
        from server_users import _load_users
        users_path = tmp_path / "users.json"
        users_path.write_text(json.dumps(["alice", "bob"]), encoding="utf-8")
        result = _load_users(str(tmp_path), default_user="default")
        assert result == ["alice", "bob"]

    def test_returns_default_on_corrupt_file(self, tmp_path):
        from server_users import _load_users
        (tmp_path / "users.json").write_text("not json", encoding="utf-8")
        result = _load_users(str(tmp_path), default_user="fallback")
        assert result == ["fallback"]


class TestPersistUser:
    def test_creates_file_with_user(self, tmp_path):
        from server_users import _persist_user, _load_users
        _persist_user("alice", str(tmp_path))
        users = _load_users(str(tmp_path), default_user="default")
        assert "alice" in users

    def test_does_not_duplicate_user(self, tmp_path):
        from server_users import _persist_user, _load_users
        _persist_user("alice", str(tmp_path))
        _persist_user("alice", str(tmp_path))
        users = _load_users(str(tmp_path), default_user="default")
        assert users.count("alice") == 1

    def test_stores_sorted(self, tmp_path):
        from server_users import _persist_user, _load_users
        _persist_user("charlie", str(tmp_path))
        _persist_user("alice", str(tmp_path))
        users = _load_users(str(tmp_path), default_user="default")
        assert users == sorted(users)


# ---------------------------------------------------------------------------
# server_conv_store tests
# ---------------------------------------------------------------------------

class TestGetConvFile:
    def test_path_structure(self, tmp_path):
        from server_conv_store import _get_conv_file
        path = _get_conv_file("patient1", "photo.jpg", str(tmp_path))
        assert path.endswith("photo.jpg.json")
        assert "conversations" in path
        assert "patient1" in path

    def test_creates_dir(self, tmp_path):
        from server_conv_store import _get_conv_file
        _get_conv_file("patient1", "photo.jpg", str(tmp_path))
        assert os.path.isdir(str(tmp_path / "patient1" / "conversations"))

    def test_strips_subdirectory_from_photo_id(self, tmp_path):
        from server_conv_store import _get_conv_file
        # photo_id might be "Jack/2026.jpg"
        path = _get_conv_file("p1", "Jack/2026.jpg", str(tmp_path))
        assert os.path.basename(path) == "2026.jpg.json"


class TestSaveAndLoadConvTurns:
    def test_save_creates_file(self, tmp_path):
        from server_conv_store import _save_conv_turn, _get_conv_file
        _save_conv_turn("photo.jpg", "p1", "hello", "hi there", str(tmp_path))
        path = _get_conv_file("p1", "photo.jpg", str(tmp_path))
        assert os.path.exists(path)

    def test_load_returns_saved_turns(self, tmp_path):
        from server_conv_store import _save_conv_turn, _load_conv_turns
        _save_conv_turn("photo.jpg", "p1", "hello", "hi there", str(tmp_path))
        turns = _load_conv_turns("photo.jpg", "p1", str(tmp_path))
        assert len(turns) == 1
        assert turns[0]["user"] == "hello"
        assert turns[0]["assistant"] == "hi there"

    def test_max_turns_limit(self, tmp_path):
        from server_conv_store import _save_conv_turn, _load_conv_turns
        for i in range(10):
            _save_conv_turn("photo.jpg", "p1", f"q{i}", f"a{i}", str(tmp_path))
        turns = _load_conv_turns("photo.jpg", "p1", str(tmp_path), max_turns=3)
        assert len(turns) == 3
        assert turns[-1]["user"] == "q9"  # last turn

    def test_load_nonexistent_returns_empty(self, tmp_path):
        from server_conv_store import _load_conv_turns
        result = _load_conv_turns("no_such.jpg", "p1", str(tmp_path))
        assert result == []

    def test_appends_across_calls(self, tmp_path):
        from server_conv_store import _save_conv_turn, _load_conv_turns
        _save_conv_turn("photo.jpg", "p1", "q1", "a1", str(tmp_path))
        _save_conv_turn("photo.jpg", "p1", "q2", "a2", str(tmp_path))
        turns = _load_conv_turns("photo.jpg", "p1", str(tmp_path), max_turns=10)
        assert len(turns) == 2
