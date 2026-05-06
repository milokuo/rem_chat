# 2026-05-06 — Jung-Min 四層 Episodic Memory Metadata

## 狀態：已實作

---

## 背景

讀取 `reference_code/JungMinPaper/jung_min_journal_new_edition.pdf` 與對應 code 後，確認論文的
autobiographical memory 不是只有 embedding retrieval，而是四層 graph：

1. Theme layer
2. Lifetime period layer
3. General event layer（people / activity / location / object，無事件時建立 virtual event node）
4. Episodic memory layer（user utterance）

參考實作以 Elasticsearch 建立 `lp`、`ge`、`ep` 三個 index；本專案保留現有 ChromaDB 架構，將 graph node
編碼成 episode metadata，避免引入第二套資料庫。

---

## 核心變更

### 1. 新增 hierarchy metadata builder

新增 `predictors/clip_iu/memory_hierarchy.py`：

- `build_episode_hierarchy_metadata()`：為每個 text-turn episode 產生
  - `theme_node_id`
  - `lifetime_node_id`
  - `general_event_nodes`
  - `general_event_names`
  - `virtual_event_node_id`
  - `episodic_node_id`
  - `autobiographical_layers`
- `event_names_from_entities()`：把 people/activity/location/object 轉成 typed event names，例如 `people:Alice`

ChromaDB metadata 只能保存 scalar，因此 node list 以 JSON string 保存。

### 2. 文字 episode 寫入四層 metadata

`server_updated_zhengxuan.py` 的 `_save_text_episode_memory()` 現在會在原本欄位之外，加上論文四層結構欄位。

原有欄位仍保留：

```
theme, lifetime_period, entities_people, entities_activities,
entities_locations, entities_objects, has_event_entities
```

因此舊 retrieval、dashboard trace 與既有測試保持相容。

### 3. Event matching 改用 general-event layer

`memory_retriever.py` 的 event matching 現在優先讀取 `general_event_names` 做 Jaccard overlap；
若舊資料沒有此欄位，則 fallback 到原本的 `entities_*` 欄位。

這讓 current dialogue features 可以和 paper-style general event layer 對齊，同時不需要重建既有 ChromaDB。

### 4. Prompt block 顯示 hierarchy 訊號

`format_episode_block()` 現在會在 episodic memory block 中加入：

- `Lifetime period`
- `General events`

讓注入 `chat_engine` 的 retrieved context 更接近論文 prompt 的七屬性記憶格式。

---

## 驗證

- `python -m py_compile predictors\clip_iu\memory_hierarchy.py predictors\clip_iu\memory_retriever.py predictors\clip_iu\photo_db.py ParlAI\projects\image_chat\server_updated_zhengxuan.py`
- `python predictors\clip_iu\test_memory_retriever.py`（26 tests）
- `python ParlAI\projects\image_chat\test_server_trace.py`（11 tests）
- `python predictors\clip_iu\test_chat_engine_trace_fields.py`（15 tests）

---

## 影響檔案

- `predictors/clip_iu/memory_hierarchy.py`（新增）
- `predictors/clip_iu/memory_retriever.py`
- `predictors/clip_iu/photo_db.py`
- `predictors/clip_iu/test_memory_retriever.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`
- `changelogs/2026-05-06_hierarchical-episode-memory.md`

