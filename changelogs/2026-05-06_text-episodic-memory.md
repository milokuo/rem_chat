# 2026-05-06 — 文字對話回合 Episodic Memory

## 狀態：已實作

---

## 背景

Jung-Min 論文的 autobiographical memory 以每個使用者 utterance 作為 episodic memory，
再以 theme、time、event entities、semantic embedding 進行多路檢索與 ranking。

本專案先前的記憶實作偏向「照片級」：每張照片有 caption / objects / theme / entities /
conv_summary，文字輪次只存 JSON conversation log，不會在每輪回覆前重新做 episodic memory
retrieval。

---

## 核心變更

### 1. Theme taxonomy 對齊論文 Table II

`memory_extractor.py` 的 20 個 theme 改為 Jung-Min 論文 Table II：

```
arts & culture, business & entrepreneurs, celebrity & pop culture,
diaries & daily life, family, fashion & style, film tv & video,
fitness & health, food & dining, gaming, learning & educational,
music, news & social concern, other hobbies, relationships,
science & technology, sports, travel & adventure,
youth & student life, other
```

不保留舊 theme alias；現有 ChromaDB 記憶預期會在功能更新後清空重建。

### 2. 新增 text-turn feature extraction

`memory_extractor.py` 新增：

- `classify_utterance_memory()`：從目前 user utterance + recent conversation context 抽出 theme / people / activities / locations / objects
- `embed_memory_text()`：用 `text-embedding-3-small` 產生文字 semantic embedding

### 3. 新增 episodic memory collection

`photo_db.py` 新增 ChromaDB collection：

- `conversation_episodes`

新增方法：

- `add_episode(episode_id, text_embedding, metadata)`
- `query_episodes_by_patient(patient_id, with_embeddings=True)`
- `count_episodes()`

每筆 episode metadata 包含：

```
patient_id, photo_id, timestamp, theme,
entities_people, entities_activities, entities_locations, entities_objects,
user_utterance, assistant_reply, content
```

### 4. 新增 episode retrieval + prompt formatting

`memory_retriever.py` 新增：

- `retrieve_episodes()`：以 semantic embedding + theme + entity + recency 做 weighted rerank
- `format_episode_block()`：格式化成 GPT 可讀的 episodic memory block

episode score 欄位：

```
_semantic_score, _visual_score, _entity_score,
_theme_match, _recency_score, _rank_score
```

`_visual_score` 保留為 dashboard 相容欄位，值等於 text semantic score。

### 5. 8082 text / streaming 路徑接入

`server_updated_zhengxuan.py` 的兩條文字路徑都改為：

```
user utterance
  -> classify_utterance_memory()
  -> embed_memory_text()
  -> retrieve_episodes()
  -> format_episode_block()
  -> combine(photo-level memory, episode-level memory)
  -> chat_engine
  -> save conversation JSON
  -> add_episode()
```

也新增 `photo_retrieved_context` / `turn_retrieved_context`，避免每輪文字 retrieval 把舊 episode block
累積進 `SHARED['retrieved_context']`。

### 6. Debug trace

文字路徑 trace 的 `rag_candidates` 現在會包含 episode candidates，並去除 embedding；
同時補上 dashboard 友善欄位：

```
visual, semantic, entity, recency, rank_score, theme_match
```

---

## 驗證

- `python -m py_compile predictors\clip_iu\memory_extractor.py predictors\clip_iu\memory_retriever.py predictors\clip_iu\photo_db.py ParlAI\projects\image_chat\server_updated_zhengxuan.py`
- `python predictors\clip_iu\test_memory_retriever.py`
- `python predictors\clip_iu\test_chat_engine_trace_fields.py`
- `python ParlAI\projects\image_chat\test_server_trace.py`

---

## 影響檔案

- `predictors/clip_iu/memory_extractor.py`
- `predictors/clip_iu/memory_retriever.py`
- `predictors/clip_iu/photo_db.py`
- `predictors/clip_iu/test_memory_retriever.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`
- `ParlAI/projects/image_chat/test_server_trace.py`
