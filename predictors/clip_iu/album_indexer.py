# -*- coding: utf-8 -*-
"""
album_indexer.py

Offline batch script: pre-processes a patient's photo album and writes
all results into ChromaDB via PhotoDB.

Usage:
    source predictors/clip_iu/clip_iu/bin/activate
    cd predictors/clip_iu
    python album_indexer.py --album_dir ../../albums/patient_01 --patient_id patient_01

Flags:
    --album_dir   Path to the folder containing the patient's photos
    --patient_id  Unique patient identifier (used for filtering in DB)
    --overwrite   Re-index photos that are already in the DB (default: skip)
    --db_dir      Path to ChromaDB persist directory (default: ./photo_index)

Services required (must be running):
    9205 — CLIP predictor  (event / place / relationship + embedding)
    9206 — DETR detector   (objects)
    9207 — BLIP captioner  (caption)
"""

import argparse
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
            "caption":      caption,
            "objects":      objects,
            "event":        event,
            "place":        place,
            "relationship": relationship,
            "patient_id":   patient_id,
            "filename":     filename,
        }

        db.add_photo(photo_id, embedding, metadata)
        print(f"    caption: {caption}")
        print(f"    event: {event} | place: {place} | relationship: {relationship}")
        print(f"    objects: {objects}")
        indexed += 1

    print(f"\nDone. Indexed: {indexed} | Skipped: {skipped} | Total in DB: {db.count()}")


def main():
    parser = argparse.ArgumentParser(description="Batch-index a photo album into ChromaDB.")
    parser.add_argument("--album_dir",  required=True, help="Path to album folder")
    parser.add_argument("--patient_id", required=True, help="Patient identifier")
    parser.add_argument("--overwrite",  action="store_true", help="Re-index already-indexed photos")
    parser.add_argument("--db_dir",     default="./photo_index", help="ChromaDB persist directory")
    args = parser.parse_args()

    if not os.path.isdir(args.album_dir):
        print(f"[ERROR] album_dir not found: {args.album_dir}")
        return

    db = PhotoDB(persist_dir=args.db_dir)
    print(f"ChromaDB at: {args.db_dir} (currently {db.count()} photo(s))")

    index_album(args.album_dir, args.patient_id, args.overwrite, db)


if __name__ == "__main__":
    main()
