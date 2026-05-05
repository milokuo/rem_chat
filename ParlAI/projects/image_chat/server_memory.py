# -*- coding: utf-8 -*-
"""
server_memory.py

Conversation-memory finalization helper for rem_chat server.
Extracted from server_updated_zhengxuan.py (Phase 2 refactor).

Depends on:
  - server_conv_store._load_conv_turns
  - memory_extractor.extract_session_memory  (predictors/clip_iu/)
"""

import datetime
import json
import os
import sys
import threading

from server_conv_store import _load_conv_turns

# Ensure predictors/clip_iu is importable (same path as the server itself).
_CLIP_IU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "predictors", "clip_iu")
if _CLIP_IU_DIR not in sys.path:
    sys.path.insert(0, _CLIP_IU_DIR)

from memory_extractor import extract_session_memory  # noqa: E402


def _finalize_conversation_memory(
    photo_id: str,
    patient_id: str,
    uploads_dir: str,
    *,
    photo_db,
    openai_client,
    model: str,
    bg_semaphore,
) -> None:
    """Finalize memory for a photo after the user switches away.

    Reads conversation turns for *photo_id*, calls GPT to extract a session
    summary and named entities, then writes the result back to ChromaDB.
    Runs in a background daemon thread to avoid blocking the main request.
    """
    if not photo_id or not openai_client:
        return

    def _run() -> None:
        try:
            conv_turns = _load_conv_turns(photo_id, patient_id, uploads_dir, max_turns=50)
            if not conv_turns:
                return

            existing = photo_db.query_by_patient(patient_id, n_results=200)
            photo_record = next((p for p in existing if p["id"] == photo_id), None)
            caption = photo_record.get("caption", "") if photo_record else ""

            mem = extract_session_memory(conv_turns, caption, openai_client, model=model)
            if not mem.get("summary") and not any(
                mem[k] for k in ("people", "activities", "locations", "objects")
            ):
                return

            updates = {
                "conv_summary": mem["summary"],
                "last_chatted": datetime.datetime.now().isoformat(),
            }
            if photo_record:
                for field, key in [
                    ("entities_people",     "people"),
                    ("entities_activities", "activities"),
                    ("entities_locations",  "locations"),
                    ("entities_objects",    "objects"),
                ]:
                    existing_list = json.loads(photo_record.get(field, "[]") or "[]")
                    merged = list(dict.fromkeys(existing_list + mem.get(key, [])))
                    updates[field] = json.dumps(merged, ensure_ascii=False)

            photo_db.update_metadata(photo_id, updates)
            print(f"[Memory] Finalized memory for {photo_id}: {mem['summary'][:80]}...")
        except Exception as exc:
            print(f"[Memory] finalize failed for {photo_id}: {exc}")

    def _run_sem() -> None:
        with bg_semaphore:
            _run()

    threading.Thread(target=_run_sem, daemon=True).start()
