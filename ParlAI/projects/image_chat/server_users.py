# -*- coding: utf-8 -*-
"""
server_users.py

User-list persistence helpers for rem_chat server.
Extracted from server_updated_zhengxuan.py (Phase 2 refactor).
"""

import json
import os


def _users_file(uploads_dir: str) -> str:
    """Return path to users.json inside uploads_dir."""
    return os.path.join(uploads_dir, "users.json")


def _load_users(uploads_dir: str, default_user: str = "default") -> list:
    """Load the user list from users.json, or return [default_user] if absent/corrupt."""
    path = _users_file(uploads_dir)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [default_user]


def _persist_user(user_id: str, uploads_dir: str) -> None:
    """Add user_id to the persistent user list if not already present."""
    users = _load_users(uploads_dir)
    if user_id not in users:
        users.append(user_id)
        os.makedirs(uploads_dir, exist_ok=True)
        with open(_users_file(uploads_dir), "w", encoding="utf-8") as f:
            json.dump(sorted(users), f, ensure_ascii=False)
