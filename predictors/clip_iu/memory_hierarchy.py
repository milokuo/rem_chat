# -*- coding: utf-8 -*-
"""
Helpers for representing Jung-Min-style autobiographical memory hierarchy.

The paper stores utterances in a four-layer graph:
theme -> lifetime period -> general event -> episodic memory.

rem_chat keeps ChromaDB as the storage backend, so we encode the graph nodes
as explicit metadata fields on each episodic-memory document. Retrieval can
then use these fields without introducing Elasticsearch as a second database.
"""

import json


ENTITY_KEYS = ("people", "activities", "locations", "objects")


def _node_part(value: str) -> str:
    """Return a stable, single-line node-id component."""
    return str(value or "").strip().replace("::", "/").replace("\n", " ")


def _dedupe(items: list) -> list[str]:
    seen = set()
    result = []
    for item in items or []:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def event_names_from_entities(entities: dict) -> list[str]:
    """Return typed general-event names, e.g. ``people:Alice``.

    Typed names preserve the paper's people/activity/location/object split and
    make event matching less ambiguous than matching raw surface strings alone.
    """
    names = []
    for key in ENTITY_KEYS:
        for value in _dedupe(entities.get(key, [])):
            names.append(f"{key}:{value}")
    return names


def build_episode_hierarchy_metadata(
    episode_id: str,
    patient_id: str,
    timestamp: str,
    theme: str,
    entities: dict,
) -> dict:
    """Build graph-like metadata for one episodic memory.

    ChromaDB metadata values must be scalar, so node collections are JSON
    strings. The original simple fields (theme, entities_*) remain stored by
    the caller for backwards compatibility.
    """
    lifetime_period = (timestamp or "")[:10]
    theme_value = _node_part(theme) or "other"
    patient = _node_part(patient_id) or "default"
    date_part = _node_part(lifetime_period) or "unknown-date"

    theme_node_id = f"theme::{theme_value}"
    lifetime_node_id = f"lifetime::{patient}::{date_part}::{theme_value}"
    episodic_node_id = f"episode::{_node_part(episode_id)}"

    general_event_names = event_names_from_entities(entities)
    general_event_nodes = []
    for event_name in general_event_names:
        event_type, event_value = event_name.split(":", 1)
        general_event_nodes.append({
            "id": f"event::{patient}::{date_part}::{theme_value}::{event_type}::{_node_part(event_value)}",
            "type": event_type,
            "name": event_value,
            "lifetime_node_id": lifetime_node_id,
            "theme_node_id": theme_node_id,
        })

    virtual_event_node_id = ""
    if not general_event_nodes:
        virtual_event_node_id = f"event::{patient}::{date_part}::{theme_value}::virtual::{_node_part(episode_id)}"
        general_event_nodes.append({
            "id": virtual_event_node_id,
            "type": "virtual",
            "name": "virtual",
            "lifetime_node_id": lifetime_node_id,
            "theme_node_id": theme_node_id,
        })

    return {
        "lifetime_period": lifetime_period,
        "theme_node_id": theme_node_id,
        "lifetime_node_id": lifetime_node_id,
        "general_event_nodes": json.dumps(general_event_nodes, ensure_ascii=False),
        "general_event_names": json.dumps(general_event_names, ensure_ascii=False),
        "virtual_event_node_id": virtual_event_node_id,
        "episodic_node_id": episodic_node_id,
        "has_event_entities": bool(general_event_names),
        "autobiographical_layers": json.dumps(
            ["theme", "lifetime_period", "general_event", "episodic"],
            ensure_ascii=False,
        ),
    }

