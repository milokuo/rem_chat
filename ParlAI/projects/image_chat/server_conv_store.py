# -*- coding: utf-8 -*-
"""
server_conv_store.py

Per-photo conversation storage helpers for rem_chat server.
Extracted from server_updated_zhengxuan.py (Phase 2 refactor).
"""

import datetime
import json
import os


def _get_conv_file(patient_id: str, photo_id: str, uploads_dir: str) -> str:
    """Return path to the JSON file storing conversation turns for a photo."""
    # photo_id may contain subdirectories (e.g. "Jack/2026.jpg"); use basename only.
    base = os.path.basename(photo_id)
    conv_dir = os.path.join(uploads_dir, patient_id, "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    return os.path.join(conv_dir, base + ".json")


def _save_conv_turn(
    photo_id: str,
    patient_id: str,
    user_msg: str,
    assistant_msg: str,
    uploads_dir: str,
) -> None:
    """Append one conversation turn to the photo's conversation file."""
    path = _get_conv_file(patient_id, photo_id, uploads_dir)
    turns = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                turns = json.load(f)
        except Exception:
            pass
    turns.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user_msg,
        "assistant": assistant_msg,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)
    print(f"[ConvStore] Saved turn for {photo_id} (total turns: {len(turns)})")


def _load_conv_turns(
    photo_id: str,
    patient_id: str,
    uploads_dir: str,
    max_turns: int = 5,
) -> list:
    """Load the most recent N conversation turns for a photo."""
    path = _get_conv_file(patient_id, photo_id, uploads_dir)
    if not os.path.exists(path):
        # Fallback: legacy naming scheme used patient_id prefix
        base = os.path.basename(photo_id)
        old_path = os.path.join(uploads_dir, patient_id, "conversations",
                                f"{patient_id}_{base}.json")
        if os.path.exists(old_path):
            path = old_path
        else:
            return []
    try:
        with open(path, encoding="utf-8") as f:
            turns = json.load(f)
        return turns[-max_turns:]
    except Exception:
        return []
