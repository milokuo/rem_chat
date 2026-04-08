# Plan: Photo-Anchored Autobiographical Memory

**Date:** 2026-04-09  
**Status:** PLANNED — not yet implemented  
**Reference:** `reference_code/JungMinPaper/jung_min_journal_new_edition.pdf`

---

## Background

目前系統的 GPT 延遲問題促使重新思考方向。討論後決定：

1. **不改懷舊治療主題**（可選，非必要）
2. **主要賣點是記憶架構**：讓 AI 真的「認識」用戶、記得每張照片聊了什麼
3. 將 Jung-Min 論文的 Hierarchical Autobiographical Memory 移植並以照片為錨點強化

## 論文核心貢獻對照

| 論文元件 | 實作方式 |
|---|---|
| 4 層 hierarchical memory | Theme + Timestamp + Entities + Episodic JSON |
| 20 themes classification | GPT 從 caption 分類，存 ChromaDB |
| Event Entity Extraction（People/Activity/Location/Object）| GPT 從 caption+objects 抽取；對話後再補充 |
| Theme Matching retrieval | query_by_theme()，同 theme 候選 |
| Event Entity Matching | Jaccard overlap，entities 候選 |
| Semantic Matching | 現有 CLIP visual similarity（取代 ERNIE SBERT）|
| Memory Ranking（時間 + 語意）| 四維加權分數 + recency softmax（不訓練模型）|
| Prompt Management | 改寫 _build_retrieved_block() + 主動引用指引 |
| Response Rating | 跳過訓練模型；靠 prompt constraint 替代 |

## 不做的功能（決定排除）

- 臉部 clustering 作為記憶主幹（標註 UI 複雜、隱私風險、工程量大）
- EXIF 時間軸（資料常缺失，容易誤導）
- 情緒驅動主動選圖（目前 UI 沒有對應行為）

## 現狀分析

已有：
- ChromaDB：CLIP embedding + caption/objects/event/place/relationship
- JSON per photo：每輪對話存到 `conversations/<photo>.json`
- RAG 注入：Top-1 相似照片近 5 輪對話已注入 GPT 提示

缺少：
- ChromaDB 沒有 `theme`、`entities`、`conv_summary`、`last_chatted`
- 只有 visual 單路檢索
- 沒有 post-session 寫回（對話結束後不更新記憶）
- retrieved 格式未結構化

---

## 實作計畫

### Phase 0：ChromaDB Schema 擴充

**檔案：** `predictors/clip_iu/photo_db.py`

新增 metadata fields：
```
theme              str   # 論文 20 themes 之一
entities_people    str   # JSON list
entities_activities str  # JSON list
entities_locations str   # JSON list
entities_objects   str   # JSON list
upload_timestamp   str   # ISO datetime（Lifetime Period Layer）
conv_summary       str   # GPT 摘要，對話後寫入（≤500 字）
last_chatted       str   # 最後對話的 ISO datetime
```

新增方法：
- `update_metadata(photo_id, updates: dict)`
- `query_by_theme(theme, patient_id, n)`
- `get_all_by_patient(patient_id)`

---

### Phase 1：新建 `memory_extractor.py`

**檔案：** `predictors/clip_iu/memory_extractor.py`（新建）

```python
def classify_theme(caption: str, objects: str) -> str:
    # GPT 分類到論文 Table II 的 20 themes

def extract_photo_entities(caption: str, objects: str) -> dict:
    # 從照片描述抽 People/Activity/Location/Object
    # 回傳 {"people": [], "activities": [], "locations": [], "objects": []}

def extract_session_memory(conversation: list[dict], photo_caption: str) -> dict:
    # 對話結束後呼叫
    # 回傳 {"summary": str, "people": [], "activities": [], "locations": [], "objects": []}
```

`classify_theme` + `extract_photo_entities` 合併為一次 GPT call 以節省延遲。

---

### Phase 2：上傳豐富化 + 對話後寫回

**2a. 上傳時**（修改 `album_indexer.py` + `server_updated_zhengxuan.py`）

```
CLIP + DETR + BLIP → metadata + theme + entities + upload_timestamp → ChromaDB
```

**2b. 對話結束時**（修改 `server_updated_zhengxuan.py`）

觸發點：chat_engine 回傳 `"last": True`  
動作：從 JSON file 載入本次對話 → `extract_session_memory()` → `photo_db.update_metadata()`

---

### Phase 3：新建 `memory_retriever.py`

**檔案：** `predictors/clip_iu/memory_retriever.py`（新建）

三路並行（對應論文 Fig. 4）：
1. **Visual**：CLIP cosine similarity（現有）
2. **Theme Matching**：同 theme 候選，按 visual_score 排序
3. **Entity Matching**：Jaccard overlap 計算 entity_score

Rerank（對應論文 Eq. 8 簡化版）：
```python
rank_score = (
    α * visual_score +     # 0.50
    β * entity_score +     # 0.25
    γ * theme_score +      # 0.15  (1.0 if same theme else 0.0)
    δ * recency_score      # 0.10  (Ebbinghaus softmax)
)
```

取代 `_save_and_retrieve()` 中的單路 visual query。

---

### Phase 4：Prompt 策略改寫

**檔案：** `predictors/clip_iu/chat_engine.py`

改寫 `_build_retrieved_block()`，格式改為：
```
[Memory 1 — related past photo]
  Theme: family
  People: 媽媽, 小美
  Activity: 生日慶祝
  Location: 家裡
  Last discussed: 2026-03-15
  Summary: 用戶說這是女兒去年生日...
```

加入主動引用 + 低信心保護規則（論文 Memory CoT reasoning instruction 精簡版）：
```
Rule: If a memory is relevant, you MAY proactively say:
"This reminds me of [brief description from memory]..."
Constraint: Only cite a specific detail you can verify.
If uncertain, ask: "Did you also talk about...?"
```

---

## 檔案變動總表

| 檔案 | 動作 |
|---|---|
| `predictors/clip_iu/photo_db.py` | 修改 — 新 fields + update_metadata + query_by_theme |
| `predictors/clip_iu/memory_extractor.py` | **新建** — theme/entity 抽取 |
| `predictors/clip_iu/memory_retriever.py` | **新建** — 三路檢索 + rerank |
| `predictors/clip_iu/album_indexer.py` | 修改 — 上傳時加 theme/entity/timestamp |
| `ParlAI/projects/image_chat/server_updated_zhengxuan.py` | 修改 — 上傳豐富化 + write-back + 換 retriever |
| `predictors/clip_iu/chat_engine.py` | 修改 — 新 prompt template + 主動引用策略 |

---

## 已知風險

| 風險 | 評估 | 處理方式 |
|---|---|---|
| theme/entity 品質取決於 BLIP caption 長短 | 中等 | 空 entities 退化為純 visual，仍有效 |
| 多一次 GPT call 增加上傳延遲 | 可接受 | theme+entity 合一次 GPT call |
| ChromaDB metadata 單值大小限制 | 存在 | summary 限 500 字，raw 繼續存 JSON |
| Entity Jaccard 在資料少時訊號弱 | 初期幾乎無效 | visual 權重 0.50 保底 |
| Response Rating 未實作 | 論文最重的訓練需求 | 靠 prompt constraint 替代 |
