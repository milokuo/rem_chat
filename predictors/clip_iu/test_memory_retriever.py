# -*- coding: utf-8 -*-
"""
Tests for memory_retriever.retrieve() — Phase 1 TDD

Verifies that each returned candidate dict carries all five score fields
needed by the debug dashboard:
  _visual_score, _entity_score, _theme_match, _recency_score, _rank_score
"""

import json
import datetime
import sys
import os
import unittest
from unittest.mock import MagicMock

# Allow import without activating clip_env
sys.path.insert(0, os.path.dirname(__file__))

import memory_retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(value: float = 0.5, dim: int = 512) -> list:
    """Unit vector in one direction for deterministic cosine similarity."""
    v = [0.0] * dim
    v[0] = value
    return v


def _make_photo(
    photo_id: str,
    theme: str = "",
    people: list = None,
    activities: list = None,
    locations: list = None,
    objects: list = None,
    last_chatted: str = "",
    caption: str = "test photo",
) -> dict:
    return {
        "id": photo_id,
        "patient_id": "P001",
        "theme": theme,
        "entities_people":     json.dumps(people or []),
        "entities_activities": json.dumps(activities or []),
        "entities_locations":  json.dumps(locations or []),
        "entities_objects":    json.dumps(objects or []),
        "last_chatted": last_chatted,
        "caption": caption,
        "embedding": _make_embedding(0.9),
    }


def _make_photo_db(photos: list) -> MagicMock:
    """Return a mock PhotoDB whose query_by_patient returns `photos`."""
    db = MagicMock()
    db.query_by_patient.return_value = photos
    return db


def _make_episode(
    episode_id: str,
    theme: str = "",
    people: list = None,
    activities: list = None,
    locations: list = None,
    objects: list = None,
    timestamp: str = "",
    user_utterance: str = "hello",
    assistant_reply: str = "hi",
) -> dict:
    return {
        "id": episode_id,
        "patient_id": "P001",
        "photo_id": "P001/photo.jpg",
        "timestamp": timestamp,
        "theme": theme,
        "entities_people": json.dumps(people or []),
        "entities_activities": json.dumps(activities or []),
        "entities_locations": json.dumps(locations or []),
        "entities_objects": json.dumps(objects or []),
        "user_utterance": user_utterance,
        "assistant_reply": assistant_reply,
        "embedding": _make_embedding(0.9),
    }


def _make_episode_db(episodes: list) -> MagicMock:
    db = MagicMock()
    db.query_episodes_by_patient.return_value = episodes
    db.query_episodes.return_value = episodes
    db.query_episodes_by_theme.side_effect = lambda theme, *_args, **_kwargs: [
        e for e in episodes if e.get("theme") == theme
    ]
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrieveScoreFields(unittest.TestCase):

    def _run_retrieve(self, candidates, query_theme="", query_entities=None):
        """Helper: build a fake DB with candidates and call retrieve()."""
        db = _make_photo_db(candidates)
        return memory_retriever.retrieve(
            photo_db=db,
            query_embedding=_make_embedding(1.0),
            query_theme=query_theme,
            query_entities=query_entities or {"people": [], "activities": [], "locations": [], "objects": []},
            patient_id="P001",
            current_photo_id="P001/current.jpg",
            n_results=10,
        )

    # --- Field presence ---

    def test_result_has_visual_score(self):
        """Each candidate must expose _visual_score."""
        photos = [_make_photo("P001/a.jpg")]
        results = self._run_retrieve(photos)
        self.assertEqual(len(results), 1)
        self.assertIn("_visual_score", results[0])

    def test_result_has_entity_score(self):
        """Each candidate must expose _entity_score."""
        photos = [_make_photo("P001/a.jpg", people=["Alice"])]
        results = self._run_retrieve(photos, query_entities={"people": ["Alice"], "activities": [], "locations": [], "objects": []})
        self.assertEqual(len(results), 1)
        self.assertIn("_entity_score", results[0])

    def test_result_has_theme_match(self):
        """Each candidate must expose _theme_match."""
        photos = [_make_photo("P001/a.jpg", theme="family")]
        results = self._run_retrieve(photos, query_theme="family")
        self.assertEqual(len(results), 1)
        self.assertIn("_theme_match", results[0])

    def test_result_has_recency_score(self):
        """Each candidate must expose _recency_score."""
        ts = (datetime.datetime.now() - datetime.timedelta(days=10)).isoformat()
        photos = [_make_photo("P001/a.jpg", last_chatted=ts)]
        results = self._run_retrieve(photos)
        self.assertEqual(len(results), 1)
        self.assertIn("_recency_score", results[0])

    def test_result_has_rank_score(self):
        """Each candidate must expose _rank_score (already present in original code)."""
        photos = [_make_photo("P001/a.jpg")]
        results = self._run_retrieve(photos)
        self.assertEqual(len(results), 1)
        self.assertIn("_rank_score", results[0])

    # --- Field types / ranges ---

    def test_theme_match_is_bool(self):
        """_theme_match must be a bool."""
        photos = [_make_photo("P001/a.jpg", theme="family")]
        results = self._run_retrieve(photos, query_theme="family")
        self.assertIsInstance(results[0]["_theme_match"], bool)

    def test_theme_match_true_when_theme_matches(self):
        photos = [_make_photo("P001/a.jpg", theme="family")]
        results = self._run_retrieve(photos, query_theme="family")
        self.assertTrue(results[0]["_theme_match"])

    def test_theme_match_false_when_theme_differs(self):
        photos = [_make_photo("P001/a.jpg", theme="travel")]
        results = self._run_retrieve(photos, query_theme="family")
        self.assertFalse(results[0]["_theme_match"])

    def test_visual_score_between_0_and_1(self):
        photos = [_make_photo("P001/a.jpg")]
        results = self._run_retrieve(photos)
        score = results[0]["_visual_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_entity_score_between_0_and_1(self):
        photos = [_make_photo("P001/a.jpg", people=["Alice", "Bob"])]
        results = self._run_retrieve(photos, query_entities={"people": ["Alice"], "activities": [], "locations": [], "objects": []})
        score = results[0]["_entity_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_recency_score_between_0_and_1(self):
        ts = (datetime.datetime.now() - datetime.timedelta(days=5)).isoformat()
        photos = [_make_photo("P001/a.jpg", last_chatted=ts)]
        results = self._run_retrieve(photos)
        score = results[0]["_recency_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_recency_score_zero_when_never_chatted(self):
        photos = [_make_photo("P001/a.jpg", last_chatted="")]
        results = self._run_retrieve(photos)
        self.assertEqual(results[0]["_recency_score"], 0.0)

    # --- Multiple candidates all have the fields ---

    def test_all_candidates_have_all_score_fields(self):
        """Every returned candidate (not just top-1) must carry all score fields."""
        ts = (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
        photos = [
            _make_photo("P001/a.jpg", theme="family", people=["Alice"], last_chatted=ts),
            _make_photo("P001/b.jpg", theme="travel"),
            _make_photo("P001/c.jpg"),
        ]
        results = self._run_retrieve(photos, query_theme="family",
                                     query_entities={"people": ["Alice"], "activities": [], "locations": [], "objects": []})
        required = {"_visual_score", "_entity_score", "_theme_match", "_recency_score", "_rank_score"}
        for r in results:
            for field in required:
                self.assertIn(field, r, msg=f"Missing {field!r} in candidate {r.get('id')}")

    # --- Results sorted by rank_score descending ---

    def test_results_sorted_by_rank_score_descending(self):
        """retrieve() must return candidates in descending rank_score order."""
        photos = [
            _make_photo("P001/a.jpg"),
            _make_photo("P001/b.jpg"),
            _make_photo("P001/c.jpg"),
        ]
        results = self._run_retrieve(photos)
        scores = [r["_rank_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # --- Empty DB ---

    def test_empty_db_returns_empty_list(self):
        db = _make_photo_db([])
        results = memory_retriever.retrieve(
            photo_db=db,
            query_embedding=_make_embedding(),
            query_theme="",
            query_entities={},
            patient_id="P001",
            current_photo_id="P001/current.jpg",
        )
        self.assertEqual(results, [])

    # --- Current photo excluded ---

    def test_current_photo_excluded_from_results(self):
        photos = [
            _make_photo("P001/current.jpg"),
            _make_photo("P001/other.jpg"),
        ]
        results = self._run_retrieve(photos)
        ids = [r["id"] for r in results]
        self.assertNotIn("P001/current.jpg", ids)
        self.assertIn("P001/other.jpg", ids)


class TestRetrieveEpisodes(unittest.TestCase):

    def _run_retrieve(self, episodes, query_theme="", query_entities=None):
        db = _make_episode_db(episodes)
        return memory_retriever.retrieve_episodes(
            photo_db=db,
            query_embedding=_make_embedding(1.0),
            query_theme=query_theme,
            query_entities=query_entities or {"people": [], "activities": [], "locations": [], "objects": []},
            patient_id="P001",
            current_episode_id="P001/episode/current",
            n_results=10,
        )

    def test_episode_result_has_score_fields(self):
        ts = datetime.datetime.now().isoformat()
        episodes = [_make_episode("P001/episode/1", theme="family", people=["Alice"], timestamp=ts)]
        results = self._run_retrieve(
            episodes,
            query_theme="family",
            query_entities={"people": ["Alice"], "activities": [], "locations": [], "objects": []},
        )
        self.assertEqual(len(results), 1)
        required = {"_semantic_score", "_visual_score", "_entity_score",
                    "_theme_match", "_recency_score", "_rank_score"}
        for field in required:
            self.assertIn(field, results[0])

    def test_current_episode_excluded(self):
        episodes = [
            _make_episode("P001/episode/current"),
            _make_episode("P001/episode/old"),
        ]
        results = self._run_retrieve(episodes)
        ids = [r["id"] for r in results]
        self.assertNotIn("P001/episode/current", ids)
        self.assertIn("P001/episode/old", ids)

    def test_format_episode_block_contains_turn_text(self):
        episodes = [_make_episode(
            "P001/episode/1",
            theme="family",
            people=["Alice"],
            user_utterance="I went with Alice.",
            assistant_reply="That sounds meaningful.",
        )]
        block = memory_retriever.format_episode_block(episodes)
        self.assertIn("Related episodic memories", block)
        self.assertIn("Theme: family", block)
        self.assertIn("People: Alice", block)
        self.assertIn("User said: I went with Alice.", block)
        self.assertIn("Assistant replied: That sounds meaningful.", block)

    def test_episode_records_matching_paths(self):
        ts = datetime.datetime.now().isoformat()
        episodes = [_make_episode("P001/episode/1", theme="family", people=["Alice"], timestamp=ts)]
        results = self._run_retrieve(
            episodes,
            query_theme="family",
            query_entities={"people": ["Alice"], "activities": [], "locations": [], "objects": []},
        )
        self.assertIn("_match_paths", results[0])
        self.assertIn("semantic", results[0]["_match_paths"])
        self.assertIn("theme", results[0]["_match_paths"])
        self.assertIn("event", results[0]["_match_paths"])

    def test_theme_matching_recovers_episode_without_semantic_query(self):
        db = _make_episode_db([
            _make_episode("P001/episode/family", theme="family"),
            _make_episode("P001/episode/travel", theme="travel"),
        ])
        results = memory_retriever.retrieve_episodes(
            photo_db=db,
            query_embedding=[],
            query_theme="family",
            query_entities={"people": [], "activities": [], "locations": [], "objects": []},
            patient_id="P001",
            n_results=10,
        )
        ids = [r["id"] for r in results]
        self.assertIn("P001/episode/family", ids)
        self.assertNotIn("P001/episode/travel", ids)
        self.assertEqual(results[0]["_match_paths"], ["theme"])

    def test_event_matching_recovers_episode_without_semantic_query(self):
        db = _make_episode_db([
            _make_episode("P001/episode/alice", people=["Alice"]),
            _make_episode("P001/episode/bob", people=["Bob"]),
        ])
        results = memory_retriever.retrieve_episodes(
            photo_db=db,
            query_embedding=[],
            query_theme="",
            query_entities={"people": ["Alice"], "activities": [], "locations": [], "objects": []},
            patient_id="P001",
            n_results=10,
        )
        ids = [r["id"] for r in results]
        self.assertIn("P001/episode/alice", ids)
        self.assertNotIn("P001/episode/bob", ids)
        self.assertEqual(results[0]["_match_paths"], ["event"])


if __name__ == "__main__":
    unittest.main()
