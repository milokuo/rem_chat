# 實作：Photo-Anchored Autobiographical Memory

**Date:** 2026-04-09  
**Status:** IMPLEMENTED  
**Plan ref:** `2026-04-09_photo-anchored-memory-plan.md` (修訂版 v2，採納 Codex 回饋)

---

## 核心目標

將 Jung-Min 論文的 Hierarchical Autobiographical Memory 以照片為錨點移植進系統：
- 4 層記憶結構（Theme / Lifetime Period / General Event / Episodic）
- 三路檢索（visual + theme + entity）+ Recency rerank
- 換圖時自動寫回記憶摘要；上傳時背景補 GPT enrichment

---

## Codex 回饋採納說明

| 回饋 | 處理 |
|---|---|
| [P1] last=True 觸發不可靠 | 改為「換圖時強制 finalize（背景 thread）」，不依賴 last=True |
| [P1] 缺舊資料遷移策略 | memory_retriever 加 lazy fallback：空欄位退回 visual-only |
| [P2] 上傳即做 GPT 拖慢批次索引 | album_indexer 改兩段式：基礎快速 + `--enrich` 選擇性補充 |
| [P2] get_all_by_patient 重複 | 改擴充現有 `query_by_patient(with_embeddings=)` 參數 |
| [P2] 完全拿掉 response rating 風險 | 保留輕量 evidence check（預設 off，`--evidence_check` flag 控制）|

---

## 新增/修改檔案

### `predictors/clip_iu/photo_db.py` — 修改

新增三個方法：
- `update_metadata(photo_id, updates)` — 部分更新 ChromaDB metadata（fetch→merge→upsert）
- `query_by_theme(theme, patient_id, n)` — 同 theme 候選查詢
- `query_by_patient(..., with_embeddings=False)` — 擴充原有方法，支援取回 embedding 向量

Schema 備注（無需 migration，ChromaDB 新欄位自動支援）：
```
theme, entities_people, entities_activities, entities_locations, entities_objects
upload_timestamp, conv_summary, last_chatted
```

---

### `predictors/clip_iu/memory_extractor.py` — 新建

兩個 GPT 功能：
1. `classify_theme_and_entities(caption, objects, client, model)` — 單次 GPT call，分類 theme（20 類）+ 抽 4 類 entity
2. `extract_session_memory(conversation, photo_caption, client, model)` — 對話結束後呼叫，生成 ≤300 字摘要 + 對話中提及的 entities

---

### `predictors/clip_iu/memory_retriever.py` — 新建

三路並行檢索 + Weighted rerank：
```
rank_score = α*visual + β*entity + γ*theme_binary + δ*recency
           = 0.50     + 0.25     + 0.15           + 0.10
```

Lazy fallback：
- `theme` 空 → 跳過 theme 路徑，weight 重分配給 visual
- `entities_*` 全空 → entity_score = 0
- `last_chatted` 空 → recency_score = 0

`format_retrieved_block()` 生成結構化 GPT prompt：
```
[Memory 1 — related past photo]
  Theme: family
  People: 媽媽, 小美
  Last discussed: 2026-03-15
  Summary: 用戶說這是女兒去年生日...
  [Past conversation (most recent turns):]
    User said: ...
    Assistant replied: ...
```

---

### `predictors/clip_iu/album_indexer.py` — 修改

兩段式設計：
- **基礎索引**（Phase 1，預設，快）：CLIP + DETR + BLIP，與舊版相同速度
- **GPT Enrichment**（Phase 2，`--enrich` flag，可中斷重跑）：補 theme/entities

```bash
python album_indexer.py --album_dir ./photos --patient_id Jack       # 快速基礎
python album_indexer.py --album_dir ./photos --patient_id Jack --enrich  # 補 GPT
```

---

### `ParlAI/projects/image_chat/server_updated_zhengxuan.py` — 修改

**新增 imports：**
- `memory_extractor`, `memory_retriever`, `openai`

**新增 `_finalize_conversation_memory(photo_id, patient_id)`：**
- 換圖時在 background thread 呼叫
- 讀 conversation JSON → GPT session summary → ChromaDB write-back
- 合併並去重 entities（既有 + 本次對話新發現）

**新增 `_enrich_photo()` inline：**
- 每次上傳後在 background thread 補 theme + entities
- 不阻塞 opening turn 回傳

**換 retriever：**
- 舊 `_save_and_retrieve()` → 新 `_memory_retrieve()` + `format_retrieved_block()`
- 上傳時 query_theme='' / query_entities={} → 退化為 visual-only（enrichment 還沒跑完）

---

### `predictors/clip_iu/chat_engine.py` — 修改

**`_build_retrieved_block()`：** 加入 Rule 3（主動引用規則 + 不確定時改問句）

**`_check_response_grounding()`：** 新方法，輕量 evidence check
- 預設關閉（`args.evidence_check` 預設 False，不影響延遲）
- 只在 `retrieved_context` 非空時執行
- 讓 GPT 確認回應只引用了 memory block 中可驗證的細節

---

## 資料流總覽

```
上傳照片
  └─ CLIP/DETR/BLIP (parallel) ─────────────────── 取 embedding, caption, objects
  └─ ChromaDB upsert (基礎 metadata)
  └─ [background] classify_theme_and_entities → update_metadata
  └─ memory_retrieve (visual-only，等 enrichment) → format_retrieved_block
  └─ chat_engine opening turn

每輪對話
  └─ _save_conv_turn → JSON (每輪，已有)
  └─ chat_engine GPT response
  └─ [optional] _check_response_grounding

換圖時
  └─ [background] _finalize_conversation_memory
       └─ 讀 JSON → extract_session_memory → update_metadata (summary + entities)
  └─ 新圖流程...
```

---

## 已知狀況與降級行為

| 狀況 | 行為 |
|---|---|
| 舊 DB 無 theme/entity | Lazy fallback → visual-only retrieval，不崩潰 |
| GPT enrichment 尚未完成 | query_theme='' → 自動跳過 theme 路徑 |
| finalize 失敗 | 印 warning，不影響對話流程 |
| evidence_check 關閉（預設）| 跳過，不增加延遲 |
| ChromaDB update_metadata 找不到 ID | 靜默 return |
