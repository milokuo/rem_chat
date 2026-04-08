# -*- coding: utf-8 -*-
"""
album_indexer.py

Offline batch script: pre-processes a patient's photo album and writes
all results into ChromaDB via PhotoDB.

Usage:
    source predictors/clip_iu/clip_env/bin/activate
    cd predictors/clip_iu
    python album_indexer.py --album_dir ../../albums/patient_01 --patient_id patient_01

    # Optionally enrich with GPT theme/entity extraction (separate pass, can be interrupted):
    python album_indexer.py --album_dir ../../albums/patient_01 --patient_id patient_01 --enrich

Flags:
    --album_dir   Path to the folder containing the patient's photos
    --patient_id  Unique patient identifier (used for filtering in DB)
    --overwrite   Re-index photos that are already in the DB (default: skip)
    --enrich      Run GPT theme/entity enrichment after basic indexing
    --db_dir      Path to ChromaDB persist directory (default: ./photo_index)

Services required (must be running):
    9205 — CLIP predictor  (event / place / relationship + embedding)
    9206 — DETR detector   (objects)
    9207 — BLIP captioner  (caption)

For --enrich, also set OPENAI_API_KEY in the environment or rely on config.py.
"""

import argparse
import datetime
import json
import os
import time

import requests

from photo_db import PhotoDB

# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLIP_URL    = "http://127.0.0.1:9205/"
DETR_URL    = "http://127.0.0.1:9206/"
CAPTION_URL = "http://127.0.0.1:9207/"


def call_service(url: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    payload["post_time"] = str(time.time())
    try:
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  [ERROR] {url} → {e}")
    return {}


def index_album(album_dir: str, patient_id: str, overwrite: bool, db: PhotoDB):
    """Phase 1: basic indexing — CLIP + DETR + BLIP only (fast, no GPT)."""
    photos = [
        f for f in os.listdir(album_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    if not photos:
        print(f"No supported images found in {album_dir}")
        return

    print(f"Found {len(photos)} photo(s) in {album_dir}")
    indexed = 0
    skipped = 0

    for filename in sorted(photos):
        photo_id = f"{patient_id}/{filename}"
        full_path = os.path.join(album_dir, filename)

        # Skip already-indexed photos unless --overwrite
        if not overwrite:
            existing = db.collection.get(ids=[photo_id])
            if existing["ids"]:
                print(f"  [SKIP] {filename} (already indexed)")
                skipped += 1
                continue

        print(f"  [INDEX] {filename}")

        # 1. CLIP: classification + embedding
        clip_res = call_service(CLIP_URL, {"cate": "clip", "img_id": filename, "full_path": full_path})
        event        = clip_res.get("event", {}).get("label", "unknown")
        place        = clip_res.get("place", {}).get("label", "unknown")
        relationship = clip_res.get("relationship", {}).get("label", "unknown")
        embedding    = clip_res.get("embedding", [])

        if not embedding:
            print(f"    [WARN] No embedding returned for {filename}, skipping.")
            continue

        # 2. DETR: objects
        detr_res = call_service(DETR_URL, {"cate": "detr", "img_id": filename, "full_path": full_path})
        objects = detr_res.get("objects", "")

        # 3. BLIP: caption
        cap_res = call_service(CAPTION_URL, {"cate": "caption", "img_id": filename, "full_path": full_path})
        caption = cap_res.get("caption", "")

        metadata = {
            "caption":          caption,
            "objects":          objects,
            "event":            event,
            "place":            place,
            "relationship":     relationship,
            "patient_id":       patient_id,
            "filename":         filename,
            "upload_timestamp": datetime.datetime.now().isoformat(),
        }

        db.add_photo(photo_id, embedding, metadata)
        print(f"    caption: {caption}")
        print(f"    event: {event} | place: {place} | relationship: {relationship}")
        print(f"    objects: {objects}")
        indexed += 1

    print(f"\nDone. Indexed: {indexed} | Skipped: {skipped} | Total in DB: {db.count()}")


def enrich_album(patient_id: str, db: PhotoDB, model: str = "gpt-5-mini"):
    """Phase 2: GPT enrichment — theme + entity extraction for photos lacking them.

    Skips photos that already have a non-empty 'theme' field (idempotent).
    Can be interrupted and re-run safely.
    """
    try:
        import openai
        from memory_extractor import classify_theme_and_entities
    except ImportError as exc:
        print(f"[Enrich] Cannot import required modules: {exc}")
        return

    # Load OpenAI key from config.py if available
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_cfg", os.path.join(os.path.dirname(__file__), "config.py"))
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
            api_key = cfg.parse_args().openai_key
            model = cfg.parse_args().model_name
        except Exception as e:
            print(f"[Enrich] Could not load API key from config.py: {e}")
            return

    client = openai.OpenAI(api_key=api_key)

    photos = db.query_by_patient(patient_id, n_results=1000)
    need_enrich = [p for p in photos if not p.get("theme")]
    print(f"\n[Enrich] {len(need_enrich)} photo(s) need enrichment (theme missing).")

    enriched = 0
    for p in need_enrich:
        photo_id = p["id"]
        caption  = p.get("caption", "")
        objects  = p.get("objects", "")
        print(f"  [ENRICH] {photo_id}")
        info = classify_theme_and_entities(caption, objects, client, model=model)
        updates = {
            "theme":                info["theme"],
            "entities_people":      json.dumps(info["people"],     ensure_ascii=False),
            "entities_activities":  json.dumps(info["activities"], ensure_ascii=False),
            "entities_locations":   json.dumps(info["locations"],  ensure_ascii=False),
            "entities_objects":     json.dumps(info["objects"],    ensure_ascii=False),
        }
        db.update_metadata(photo_id, updates)
        print(f"    theme={info['theme']} | people={info['people']}")
        enriched += 1
        time.sleep(0.3)  # gentle rate limit

    print(f"\n[Enrich] Done. Enriched: {enriched} photo(s).")


def main():
    parser = argparse.ArgumentParser(description="Batch-index a photo album into ChromaDB.")
    parser.add_argument("--album_dir",  required=True, help="Path to album folder")
    parser.add_argument("--patient_id", required=True, help="Patient identifier")
    parser.add_argument("--overwrite",  action="store_true", help="Re-index already-indexed photos")
    parser.add_argument("--enrich",     action="store_true", help="Run GPT theme/entity enrichment after indexing")
    parser.add_argument("--db_dir",     default="./photo_index", help="ChromaDB persist directory")
    args = parser.parse_args()

    if not os.path.isdir(args.album_dir):
        print(f"[ERROR] album_dir not found: {args.album_dir}")
        return

    db = PhotoDB(persist_dir=args.db_dir)
    print(f"ChromaDB at: {args.db_dir} (currently {db.count()} photo(s))")

    index_album(args.album_dir, args.patient_id, args.overwrite, db)

    if args.enrich:
        enrich_album(args.patient_id, db)


if __name__ == "__main__":
    main()
