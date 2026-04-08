# -*- coding: utf-8 -*-
"""
memory_extractor.py

GPT-based extraction helpers for the autobiographical memory system.

Two responsibilities:
  1. classify_theme_and_entities()  — called once at upload time to annotate a photo
  2. extract_session_memory()       — called at end-of-session (换图) to summarise a conversation
"""

import json
import logging
import time
import openai

logger = logging.getLogger(__name__)

# 20 themes from Jung-Min paper Table II
THEMES = [
    "family", "friends", "romance", "work", "education",
    "travel", "sports", "hobbies", "food", "celebration",
    "childhood", "health", "religion", "community", "nature",
    "pets", "home", "music", "art", "other",
]

_THEME_LIST_STR = ", ".join(THEMES)


def classify_theme_and_entities(
    caption: str,
    objects: str,
    client: openai.OpenAI,
    model: str = "gpt-5-mini",
) -> dict:
    """One GPT call: classify theme + extract entities from a photo's caption/objects.

    Returns:
        {
            "theme": str,               # one of THEMES
            "people": list[str],
            "activities": list[str],
            "locations": list[str],
            "objects": list[str],
        }

    On any error, returns {"theme": "", "people": [], "activities": [], "locations": [], "objects": []}.
    """
    empty = {"theme": "", "people": [], "activities": [], "locations": [], "objects": []}
    if not caption and not objects:
        return empty

    prompt = f"""Given a photo description, do two things:
1. Classify the photo into ONE theme from this list: {_THEME_LIST_STR}
2. Extract named entities of four types: people, activities, locations, objects.

Photo description: {caption}
Detected objects: {objects}

Respond ONLY with JSON (no extra text):
{{
  "theme": "<one theme>",
  "people": ["<person>", ...],
  "activities": ["<activity>", ...],
  "locations": ["<location>", ...],
  "objects": ["<object>", ...]
}}"""

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        raw = response.choices[0].message.content.strip()
        logger.debug("classify_theme_and_entities took %dms", elapsed)

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        theme = parsed.get("theme", "").strip().lower()
        if theme not in THEMES:
            theme = "other"
        return {
            "theme": theme,
            "people": [str(x) for x in parsed.get("people", [])],
            "activities": [str(x) for x in parsed.get("activities", [])],
            "locations": [str(x) for x in parsed.get("locations", [])],
            "objects": [str(x) for x in parsed.get("objects", [])],
        }
    except Exception as exc:
        logger.warning("classify_theme_and_entities failed: %s", exc)
        return empty


def extract_session_memory(
    conversation: list[dict],
    photo_caption: str,
    client: openai.OpenAI,
    model: str = "gpt-5-mini",
) -> dict:
    """Summarise a completed conversation session for long-term memory storage.

    Args:
        conversation: List of {"user": str, "assistant": str} dicts.
        photo_caption: BLIP caption for the current photo (context anchor).
        client: Authenticated openai.OpenAI client.
        model: Model name.

    Returns:
        {
            "summary": str,             # ≤300 chars natural-language summary
            "people": list[str],        # additional people mentioned in conversation
            "activities": list[str],
            "locations": list[str],
            "objects": list[str],
        }

    On any error, returns {"summary": "", "people": [], ...}.
    """
    empty = {"summary": "", "people": [], "activities": [], "locations": [], "objects": []}
    if not conversation:
        return empty

    # Format conversation for the prompt (last 10 turns max to stay under context)
    recent = conversation[-10:]
    conv_text = "\n".join(
        f"User: {t.get('user', '')}\nAssistant: {t.get('assistant', '')}"
        for t in recent
    )

    prompt = f"""A therapy conversation just ended. The photo shows: {photo_caption}

Conversation:
{conv_text}

Extract a memory record for long-term storage. Respond ONLY with JSON:
{{
  "summary": "<one-paragraph summary of what the user recalled, ≤300 chars>",
  "people": ["<person mentioned>", ...],
  "activities": ["<activity mentioned>", ...],
  "locations": ["<location mentioned>", ...],
  "objects": ["<object mentioned>", ...]
}}"""

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        raw = response.choices[0].message.content.strip()
        logger.debug("extract_session_memory took %dms", elapsed)

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        summary = str(parsed.get("summary", ""))[:300]
        return {
            "summary": summary,
            "people": [str(x) for x in parsed.get("people", [])],
            "activities": [str(x) for x in parsed.get("activities", [])],
            "locations": [str(x) for x in parsed.get("locations", [])],
            "objects": [str(x) for x in parsed.get("objects", [])],
        }
    except Exception as exc:
        logger.warning("extract_session_memory failed: %s", exc)
        return empty
