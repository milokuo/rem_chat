# -*- coding: utf-8 -*-
"""
debug_dashboard.py

Lightweight read-only observability dashboard for rem_chat (port 8090).
Receives fire-and-forget trace payloads from the main server and
makes them available to a browser UI for debugging.

Routes
------
GET  /                           Serve the dashboard HTML page.
GET  /api/turns                  Return last 50 traces, newest-first, as JSON.
POST /api/trace                  Ingest one trace payload (from server).
GET  /api/photo/<path:p>         Serve a photo thumbnail from the uploads dir.
GET  /api/memory/patients        List all patient_ids stored in ChromaDB.
GET  /api/memory/<patient_id>    List all photos/memories for a patient.
"""

import os
import threading
from collections import deque
from flask import Flask, jsonify, request, send_file, abort, render_template

# ---- Ring buffer (module-level so module reload in tests resets it) ----------
_MAX_TURNS = 50
_buffer: deque = deque(maxlen=_MAX_TURNS)
_lock = threading.Lock()

# Default uploads root — same directory as this file + "uploads/"
_DEFAULT_UPLOADS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

# Default ChromaDB dir — relative to this file
_DEFAULT_DB_DIR = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../../predictors/clip_iu/photo_index")
)


def _open_collection(db_dir: str):
    """Open the ChromaDB 'photos' collection, or return None on failure."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_dir)
        return client.get_or_create_collection(
            name="photos",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        print(f"[dashboard] ChromaDB open failed: {exc}")
        return None


def create_app(uploads_root: str = _DEFAULT_UPLOADS_ROOT,
               db_dir: str = _DEFAULT_DB_DIR) -> Flask:
    """Application factory — allows test injection of uploads_root and db_dir."""
    app = Flask(__name__, template_folder="templates")
    app.config["UPLOADS_ROOT"] = os.path.realpath(uploads_root)
    app.config["DB_DIR"] = os.path.realpath(db_dir)

    # ------------------------------------------------------------------ traces
    @app.route("/")
    def index():
        return render_template("debug_dashboard.html")

    @app.route("/api/turns", methods=["GET"])
    def get_turns():
        with _lock:
            turns = list(_buffer)
        turns.reverse()
        return jsonify(turns)

    @app.route("/api/trace", methods=["POST"])
    def post_trace():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "invalid JSON"}), 400
        with _lock:
            _buffer.append(data)
        return jsonify({"ok": True})

    @app.route("/api/photo/<path:photo_path>")
    def serve_photo(photo_path: str):
        uploads_root = app.config["UPLOADS_ROOT"]
        photo_path = photo_path.lstrip("/")
        candidate = os.path.realpath(os.path.join(uploads_root, photo_path))
        if not candidate.startswith(uploads_root + os.sep) and candidate != uploads_root:
            abort(403)
        if not os.path.isfile(candidate):
            abort(404)
        return send_file(candidate)

    # ------------------------------------------------------------------ memory
    @app.route("/api/memory/patients", methods=["GET"])
    def get_memory_patients():
        """Return list of {patient_id, count} for all patients in ChromaDB."""
        col = _open_collection(app.config["DB_DIR"])
        if col is None or col.count() == 0:
            return jsonify([])
        results = col.get(include=["metadatas"])
        patients: dict[str, int] = {}
        for meta in results["metadatas"]:
            pid = meta.get("patient_id", "")
            if pid:
                patients[pid] = patients.get(pid, 0) + 1
        return jsonify([{"patient_id": p, "count": c}
                        for p, c in sorted(patients.items())])

    @app.route("/api/memory/<patient_id>", methods=["GET"])
    def get_memory(patient_id: str):
        """Return all photos/memories stored for a patient (no embeddings)."""
        col = _open_collection(app.config["DB_DIR"])
        if col is None or col.count() == 0:
            return jsonify([])
        results = col.get(
            where={"patient_id": patient_id},
            include=["metadatas"],
        )
        items = []
        for i, photo_id in enumerate(results["ids"]):
            item = {"id": photo_id}
            item.update(results["metadatas"][i])
            # Drop raw embedding-like fields if somehow present
            item.pop("embedding", None)
            items.append(item)
        # Sort: photos with conv_summary (i.e. discussed) first, then by last_chatted desc
        items.sort(key=lambda x: (
            0 if x.get("conv_summary") else 1,
            -(len(x.get("last_chatted") or "")),
        ))
        return jsonify(items)

    return app


# ---- Stand-alone entry point -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8090))
    app = create_app()
    print(f"[debug_dashboard] listening on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
