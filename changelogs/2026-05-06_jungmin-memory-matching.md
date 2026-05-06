# 2026-05-06 — Jung-Min Feature Matching 對齊

## 狀態：已實作

---

## 背景

Jung-Min 論文的 autobiographical memory retrieval 不是直接把全部 memory 一起打分，
而是先做三路 feature matching：

1. semantic matching
2. theme matching
3. event entity matching

三路候選 union 後，才進入 memory ranking。rem_chat 先前的 text episodic retrieval
已具備 theme/entity/semantic/recency 分數，但流程上是「取出全部 episodes 後直接 rerank」，
和論文流程仍有落差。

---

## 核心變更

### 1. `retrieve_episodes()` 改成 matching → union → rerank

`predictors/clip_iu/memory_retriever.py` 的 text-turn episodic retrieval 現在分成三路：

- Semantic：優先使用 ChromaDB episode vector query；失敗時 fallback 為 Python cosine sort
- Theme：使用 `theme` metadata 取同主題 episodes，再以 semantic score 排序
- Event：使用 people / activities / locations / objects 的 Jaccard overlap 取候選

三路結果會 union 去重，保留 `_match_paths` debug 欄位，例如：

```
["_match_paths": ["semantic", "theme", "event"]]
```

接著沿用原本 weighted rerank 欄位：

```
_semantic_score, _visual_score, _entity_score,
_theme_match, _recency_score, _rank_score
```

### 2. PhotoDB 新增 episode 查詢 helper

`predictors/clip_iu/photo_db.py` 新增：

- `query_episodes(query_embedding, n_results, patient_id, with_embeddings)`
- `query_episodes_by_theme(theme, patient_id, n_results, with_embeddings)`

讓 semantic/theme matching 可以透過 ChromaDB helper 執行，而不是把資料庫細節散在 retriever。

### 3. Episode metadata 補 lifetime/general-event 對齊欄位

`server_updated_zhengxuan.py` 儲存 text-turn episodic memory 時新增：

- `lifetime_period`：由 timestamp 取 `YYYY-MM-DD`
- `has_event_entities`：該 utterance 是否抽到 people/activity/location/object

這讓 ChromaDB metadata 更明確對應論文的 lifetime period layer 與 general event layer。

---

## 設計取捨

- 不引入 Elasticsearch；保留目前 ChromaDB 架構。
- 不照搬論文的 MacBERT cross-encoder ranking；目前仍使用既有 heuristic weighted rerank。
- `_match_paths` 只作為 observability/debug 欄位，不影響 prompt 文字。
- Theme/event matching 對 ChromaDB metadata 做輕量查詢與 Python side scoring，避免大型 schema migration。

---

## 驗證

- `python -m py_compile predictors\clip_iu\memory_retriever.py predictors\clip_iu\photo_db.py ParlAI\projects\image_chat\server_updated_zhengxuan.py`
- `python predictors\clip_iu\test_memory_retriever.py`
- `python predictors\clip_iu\test_chat_engine_trace_fields.py`
- `python ParlAI\projects\image_chat\test_server_trace.py`

---

## 影響檔案

- `predictors/clip_iu/memory_retriever.py`
- `predictors/clip_iu/photo_db.py`
- `predictors/clip_iu/test_memory_retriever.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`
- `changelogs/2026-05-06_jungmin-memory-matching.md`
