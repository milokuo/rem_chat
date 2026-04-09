# Debug Dashboard — 完整 RAG / GPT 可觀測性系統

**日期：** 2026-04-09
**狀態：** 已實作，13 + 17 + 11 = 41 tests 全部通過

---

## 問題背景

系統每次對話都呼叫 CLIP / DETR / BLIP / RAG / GPT，但開發者完全看不到：

1. RAG 為什麼選出這張照片而非另一張（各信號分數不透明）
2. GPT 實際收到的完整 prompt 是什麼
3. GPT 回了什麼原始 CoT 文字（postprocess 之前）
4. 各服務的延遲分佈

---

## 解決方案：四個 Phase 串接

### Phase 1 — `memory_retriever.py`：分數暴露

每個 RAG 候選物件新增三個欄位，讓下游（dashboard）可以顯示 per-signal breakdown：

```python
candidate["_entity_score"]   # float [0, 1] — 實體重疊比例
candidate["_theme_match"]    # bool — 主題是否吻合
candidate["_recency_score"]  # float [0, 1] — 近期度（上次聊天時間）
```

**測試：** `test_memory_retriever.py` — 16 tests，涵蓋欄位存在、型別、值域、sort order、空 DB、排除 current photo

---

### Phase 2 — `debug_dashboard.py` + `debug_dashboard.html`：可視化端點

Flask app 在 port **8090** 提供 debug 介面。

**架構：**
- `POST /api/trace` — 接收 trace payload，存入 `deque(maxlen=50)` ring buffer
- `GET /api/turns` — 回傳最近 50 筆（newest-first），瀏覽器每 3 秒 poll
- `GET /` — 回傳 `debug_dashboard.html`
- `GET /api/photo/<path:photo_path>` — 提供照片圖片（含 path traversal guard）

**安全設計：**
- `realpath` + `startswith(uploads_root)` 雙層 path traversal 防護
- `lstrip("/")` 防止絕對路徑覆蓋
- `escHtml()` 對 ts / patient_id / model_name 做 XSS escaping

**UI 功能：**
- 左側列表：最近 50 筆 turn（點擊展開）
- 右側詳情：
  - Photo preview + patient_id + user input
  - Timing bar（CLIP / DETR / BLIP / RAG / GPT ms）
  - **RAG Candidates 表格**：每個候選的 visual / entity / theme / recency score bar
  - Full GPT Prompt（可收合）
  - Raw GPT Response CoT（可收合）
  - Final Response

**測試：** `test_debug_dashboard.py` — 17 tests，涵蓋 POST/GET 所有 route、ring buffer 語意、path traversal 封鎖、concurrent writes

---

### Phase 3 — `chat_engine.py`：trace 欄位輸出

`SocialREMChat` 新增兩個 instance 屬性：

```python
self._last_full_prompt: list = []   # messages list 傳給 OpenAI API 的完整內容
self._last_raw_response: str = ""   # GPT 回的原始文字（postprocess 之前）
```

Flask route 在兩個路徑（`reset` 新圖 + 一般對話）的 response JSON 都新增：

```json
{
  "full_prompt":   [...],   // [{role, content}, ...]
  "raw_response":  "9. Sure!\nJSON_OUTPUT: ...",
  "model_name":    "gpt-5-mini"
}
```

**測試：** `test_chat_engine_trace_fields.py` — 13 tests，涵蓋欄位存在、型別、raw_response 等於 GPT 原文、reset 路徑、既有欄位不受影響

---

### Phase 4 — `server_updated_zhengxuan.py`：fire-and-forget trace

新增 module-level 函數：

```python
def _post_trace(payload: dict) -> None:
    """Fire-and-forget daemon thread → POST to http://localhost:8090/api/trace, timeout=300ms."""
```

- Daemon thread：不會阻塞 process 退出
- 300ms timeout：dashboard 沒開時不影響主流程
- Silent on exception：ConnectionError / Timeout 靜默丟棄

**兩個 call sites：**

1. `_handle_new_image()` 末尾 — 包含完整 `rag_candidates`（embedding 欄位已剝除）
2. `interactive_running()` 末尾 — `rag_candidates: []`（文字輪次不做 RAG）

**Trace payload schema：**
```json
{
  "ts":               "2026-04-09T11:30:00",
  "patient_id":       "P001",
  "user_input":       "...",
  "model_name":       "gpt-5-mini",
  "full_prompt":      [{...}],
  "raw_response":     "...",
  "final_response":   "...",
  "timing":           {"total_ms": 1200, "gpt_ms": 800, ...},
  "photo_id":         "P001/birthday.jpg",
  "retrieved_context": "...",
  "rag_candidates":   [{"id": "...", "caption": "...", "_rank_score": 0.9, ...}]
}
```

**注意：** `candidates = []` 初始化移到 RAG block 前，確保 embedding 缺失或例外時有預設值。

**測試：** `test_server_trace.py` — 11 tests，涵蓋 URL、payload 傳遞、300ms timeout、ConnectionError 靜默、daemon thread

---

## 啟動方式

```bash
# 額外啟動 debug dashboard（clip_env）
source predictors/clip_iu/clip_env/bin/activate
python ParlAI/projects/image_chat/debug_dashboard.py
# → http://localhost:8090
```

其他服務（9205 CLIP / 9206 DETR / 9207 BLIP / 8087 chat_engine / 8082 server）照舊啟動即可。Dashboard 不在線時 trace 靜默丟棄，不影響主系統。

---

## 影響的檔案

| 檔案 | 狀態 | Phase |
|------|------|-------|
| `predictors/clip_iu/memory_retriever.py` | 修改 | 1 |
| `predictors/clip_iu/test_memory_retriever.py` | NEW | 1 |
| `ParlAI/projects/image_chat/debug_dashboard.py` | NEW | 2 |
| `ParlAI/projects/image_chat/templates/debug_dashboard.html` | NEW | 2 |
| `ParlAI/projects/image_chat/test_debug_dashboard.py` | NEW | 2 |
| `predictors/clip_iu/chat_engine.py` | 修改 | 3 |
| `predictors/clip_iu/test_chat_engine_trace_fields.py` | NEW | 3 |
| `ParlAI/projects/image_chat/server_updated_zhengxuan.py` | 修改 | 4 |
| `ParlAI/projects/image_chat/test_server_trace.py` | NEW | 4 |
