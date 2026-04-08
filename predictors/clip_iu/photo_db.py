# -*- coding: utf-8 -*-
import logging

"""
photo_db.py

ChromaDB wrapper for the rem_chat album RAG system.
All other modules interact with the vector database through this class.

Schema per document:
  id         : filename (unique)
  embedding  : CLIP image embedding (512-dim)
  metadata   : caption, objects, event, place, relationship, patient_id
"""

import chromadb

# Suppress ChromaDB's verbose debug logs (e.g. raw embedding values printed during upsert).
# Using a root Filter covers all loggers regardless of their name or lazy initialisation.
class _SuppressChromaVerbose(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "in upsert" not in msg and "in add" not in msg

logging.root.addFilter(_SuppressChromaVerbose())
# Also set level in case some handlers have their own level check
logging.getLogger("chromadb").setLevel(logging.WARNING)


class PhotoDB:
    def __init__(self, persist_dir="./photo_index"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="photos",
            metadata={"hnsw:space": "cosine"},
        )

    def add_photo(self, photo_id: str, image_embedding: list, metadata: dict):
        """Insert or update a photo record."""
        self.collection.upsert(
            ids=[photo_id],
            embeddings=[image_embedding],
            metadatas=[metadata],
        )

    def query(self, query_embedding: list, n_results: int = 3, patient_id: str = None) -> list[dict]:
        """Return Top-K most similar photos. If patient_id is given, only search within that patient's photos."""
        if self.collection.count() == 0:
            return []

        if patient_id:
            # Count available docs for this patient to avoid n_results > available error
            patient_docs = self.collection.get(where={"patient_id": patient_id})
            available = len(patient_docs["ids"])
            if available == 0:
                return []
            n = min(n_results, available)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where={"patient_id": patient_id},
            )
        else:
            n = min(n_results, self.collection.count())
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
            )

        items = []
        for i, photo_id in enumerate(results["ids"][0]):
            item = {"id": photo_id, "distance": results["distances"][0][i]}
            item.update(results["metadatas"][0][i])
            items.append(item)
        return items

    def query_by_patient(self, patient_id: str, n_results: int = 10) -> list[dict]:
        """Return all indexed photos for a specific patient (no embedding needed)."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.get(
            where={"patient_id": patient_id},
        )
        items = []
        for i, photo_id in enumerate(results["ids"]):
            item = {"id": photo_id}
            item.update(results["metadatas"][i])
            items.append(item)
        return items[:n_results]

    def reset(self):
        """Drop and recreate the collection."""
        self.client.delete_collection("photos")
        self.collection = self.client.get_or_create_collection(
            name="photos",
            metadata={"hnsw:space": "cosine"},
        )

    def list_patients(self) -> list[str]:
        """Return sorted list of unique patient_ids stored in the DB."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get()
        patients = {m.get("patient_id", "") for m in results["metadatas"] if m.get("patient_id")}
        return sorted(patients)

    def count(self) -> int:
        return self.collection.count()
