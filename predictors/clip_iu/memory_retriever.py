# -*- coding: utf-8 -*-
"""
memory_retriever.py

Three-way retrieval + weighted rerank for the Photo-Anchored Autobiographical Memory system.

Retrieval paths (parallel, then merged):
  1. Visual    — CLIP cosine similarity  (weight α = 0.50)
  2. Theme     — same-theme candidates   (weight γ = 0.15, binary)
  3. Entity    — Jaccard entity overlap  (weight β = 0.25)
  4. Recency   — Ebbinghaus-inspired     (weight δ = 0.10)

Lazy fallback rules (empty fields → skip that path, visual score carries full weight):
  - theme empty  → skip theme matching
  - entities all empty → entity_score = 0
  - last_chatted empty → recency_score = 0
"""

import json
import math
import datetime
import logging
from typing import Optional

from photo_db import PhotoDB

logger = logging.getLogger(__name__)

# Retrieval weights (must sum to 1.0)
_W_VISUAL   = 0.50
_W_ENTITY   = 0.25
_W_THEME    = 0.15
_W_RECENCY  = 0.10

# Recency decay: how many days until the score halves
_RECENCY_HALF_LIFE_DAYS = 30.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (returns 0 on error)."""
    try:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets. Returns 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _entity_score(candidate: dict, query_entities: dict) -> float:
    """Average Jaccard across the four entity types."""
    total = 0.0
    count = 0
    for field, key in [
        ("entities_people",     "people"),
        ("entities_activities", "activities"),
        ("entities_locations",  "locations"),
        ("entities_objects",    "objects"),
    ]:
        raw = candidate.get(field, "")
        if not raw:
            continue
        try:
            cand_set = set(json.loads(raw))
        except Exception:
            cand_set = set()
        query_set = set(query_entities.get(key, []))
        if cand_set or query_set:
            total += _jaccard(cand_set, query_set)
            count += 1
    return total / count if count > 0 else 0.0


def _recency_score(last_chatted: Optional[str]) -> float:
    """Ebbinghaus-inspired decay: score = exp(-λ * days_since_last_chat).

    Returns 0 if last_chatted is empty (never discussed).
    """
    if not last_chatted:
        return 0.0
    try:
        dt = datetime.datetime.fromisoformat(last_chatted)
        days_ago = max((datetime.datetime.now() - dt).total_seconds() / 86400, 0)
        lam = math.log(2) / _RECENCY_HALF_LIFE_DAYS
        return math.exp(-lam * days_ago)
    except Exception:
        return 0.0


def _rank_score(
    visual: float,
    entity: float,
    theme_match: bool,
    recency: float,
    has_entities: bool,
    has_theme: bool,
    has_recency: bool,
) -> float:
    """Compute weighted rank score, redistributing weight from missing signals."""
    # Determine effective weights given what signals are available
    w_v = _W_VISUAL
    w_e = _W_ENTITY  if has_entities else 0.0
    w_t = _W_THEME   if has_theme    else 0.0
    w_r = _W_RECENCY if has_recency  else 0.0
    total_w = w_v + w_e + w_t + w_r
    if total_w == 0:
        return 0.0
    # Normalise so weights still sum to 1
    scale = 1.0 / total_w
    return (
        w_v * visual              * scale +
        w_e * entity              * scale +
        w_t * (1.0 if theme_match else 0.0) * scale +
        w_r * recency             * scale
    )


def retrieve(
    photo_db: PhotoDB,
    query_embedding: list[float],
    query_theme: str,
    query_entities: dict,
    patient_id: str,
    current_photo_id: str,
    n_results: int = 3,
) -> list[dict]:
    """Run three-way retrieval + rerank.

    Args:
        photo_db:          PhotoDB instance.
        query_embedding:   CLIP embedding of the current photo.
        query_theme:       Theme of the current photo (may be empty → lazy fallback).
        query_entities:    {"people": [...], "activities": [...], "locations": [...], "objects": [...]}.
        patient_id:        Restrict search to this patient.
        current_photo_id:  Exclude the current photo from results.
        n_results:         How many results to return.

    Returns:
        List of candidate dicts (photo metadata + rank_score), sorted descending.
    """
    # Fetch all photos for this patient (with embeddings for visual scoring)
    all_photos = photo_db.query_by_patient(patient_id, n_results=500, with_embeddings=True)
    candidates = [p for p in all_photos if p.get("id") != current_photo_id]

    if not candidates:
        return []

    # Determine which signals are globally available (at least one candidate has them)
    has_theme_signal   = query_theme and any(p.get("theme") for p in candidates)
    has_entity_signal  = any(
        p.get(f) for p in candidates
        for f in ("entities_people", "entities_activities", "entities_locations", "entities_objects")
    )
    has_recency_signal = any(p.get("last_chatted") for p in candidates)

    scored = []
    for p in candidates:
        emb = p.get("embedding")
        visual = _cosine_similarity(query_embedding, emb) if emb else 0.0

        entity = _entity_score(p, query_entities) if has_entity_signal else 0.0
        theme_match = bool(query_theme and p.get("theme") == query_theme)
        recency = _recency_score(p.get("last_chatted")) if has_recency_signal else 0.0

        score = _rank_score(
            visual=visual,
            entity=entity,
            theme_match=theme_match,
            recency=recency,
            has_entities=has_entity_signal,
            has_theme=has_theme_signal,
            has_recency=has_recency_signal,
        )
        p["_rank_score"]   = score
        p["_visual_score"] = visual
        scored.append(p)

    scored.sort(key=lambda x: x["_rank_score"], reverse=True)
    return scored[:n_results]


def format_retrieved_block(
    candidates: list[dict],
    load_conv_turns_fn,
    patient_id: str,
) -> str:
    """Format retrieved candidates into a structured GPT prompt block.

    Args:
        candidates:          Output of retrieve().
        load_conv_turns_fn:  Callable(photo_id, patient_id, max_turns) → list of turn dicts.
        patient_id:          Used for loading conversation turns.

    Returns:
        Formatted string for injection into the system prompt.
    """
    if not candidates:
        return ""

    lines = []
    for i, c in enumerate(candidates, 1):
        theme   = c.get("theme", "")
        caption = c.get("caption", "")
        summary = c.get("conv_summary", "")
        last_dt = c.get("last_chatted", "")
        photo_id = c["id"]
        r_patient = c.get("patient_id", patient_id)

        # Entity lines — only include non-empty ones
        entity_parts = []
        for label, field in [
            ("People",     "entities_people"),
            ("Activities", "entities_activities"),
            ("Location",   "entities_locations"),
        ]:
            raw = c.get(field, "")
            if raw:
                try:
                    items = json.loads(raw)
                    if items:
                        entity_parts.append(f"  {label}: {', '.join(items)}")
                except Exception:
                    pass

        lines.append(f"[Memory {i} — related past photo]")
        if theme:
            lines.append(f"  Theme: {theme}")
        lines.extend(entity_parts)
        if last_dt:
            try:
                dt = datetime.datetime.fromisoformat(last_dt)
                lines.append(f"  Last discussed: {dt.strftime('%Y-%m-%d')}")
            except Exception:
                pass
        if summary:
            lines.append(f"  Summary: {summary}")
        else:
            # Fall back to caption if no summary yet
            if caption:
                lines.append(f"  Caption: {caption}")

        # For top-1: include raw past conversation turns
        if i == 1:
            past_turns = load_conv_turns_fn(photo_id, r_patient, 5)
            if past_turns:
                lines.append("  [Past conversation (most recent turns):]")
                for t in past_turns:
                    lines.append(f"    User said: {t.get('user', '')}")
                    lines.append(f"    Assistant replied: {t.get('assistant', '')}")
                lines.append("  [End past conversation]")

        lines.append("")  # blank line between memories

    return "\n".join(lines)
