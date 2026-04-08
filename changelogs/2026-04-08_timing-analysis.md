# 2026-04-08 — 部件時間消耗分析

## 核心變更

在不改變任何業務邏輯的前提下，於兩個主要服務加入計時儀器，記錄每個部件的耗時至 JSONL 日誌。

---

## 架構說明

### 計時點分布

**新圖流程（`_handle_new_image`）：**
| 欄位 | 說明 |
|------|------|
| `conv_over_ms` | 傳送 conversation over 至 chat_engine（條件性，僅前一對話存在時） |
| `sim_reset_ms` | reset_history 至 sim service (9110) |
| `clip_ms` | CLIP 預測服務 (9205) 個別耗時 |
| `detr_ms` | DETR 偵測服務 (9206) 個別耗時 |
| `blip_ms` | BLIP 字幕服務 (9207) 個別耗時 |
| `parallel_iu_ms` | 三服務並行區塊壁鐘時間 |
| `rag_ms` | ChromaDB upsert + query |
| `chat_engine_ms` | chat_engine 整體 round-trip (8087) |
| `gpt_ms` | GPT API 實際呼叫耗時（由 chat_engine 回傳） |
| `total_ms` | 整個 `_handle_new_image` 端到端時間 |

**文字流程（`interactive_running`）：**
| 欄位 | 說明 |
|------|------|
| `sim_pre_ms` | 請求前通知 sim service (9110) |
| `chat_engine_ms` | chat_engine 整體 round-trip (8087) |
| `gpt_ms` | GPT API 實際呼叫耗時（由 chat_engine 回傳） |
| `translate_ms` | zh 模式下的 Google Translate（僅 zh 有此欄位） |
| `sim_post_ms` | 請求後更新 sim service (9110) |
| `total_ms` | 整個 `interactive_running` 端到端時間 |

---

## 實作細節

### `_timed_worker` — 解決 ThreadPoolExecutor 計時問題

直接在 `executor.submit()` 外包時間只能得到壁鐘時間，無法取得各服務個別耗時。改用包裝函數：

```python
def _timed_worker(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round((time.perf_counter() - t0) * 1000)
```

各 future 回傳 `(result, elapsed_ms)`，解包後分別存入 timing dict。

### `gpt_ms` 跨服務傳遞

chat_engine 在 GPT call 前後加 `time.perf_counter()`，將 `gpt_ms` 包入回傳 JSON：
```json
{"return_message": "...", "last": false, "timing": {"gpt_ms": 1823}}
```
server 從 `res_chat.get("timing", {}).get("gpt_ms")` 取出，讓日誌能同時看到 round-trip 與 GPT 實際耗時（差值即網路 + Flask overhead）。

### 日誌路徑

使用 `__file__` 組絕對路徑，避免因啟動目錄不同寫錯位置：
```python
_TIMING_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_log.jsonl")
```
日誌位於 `ParlAI/projects/image_chat/timing_log.jsonl`。

### 日誌格式（JSON Lines）

```json
{"type": "new_image", "sim_reset_ms": 8, "clip_ms": 231, "detr_ms": 178, "blip_ms": 305, "parallel_iu_ms": 318, "rag_ms": 42, "chat_engine_ms": 1890, "gpt_ms": 1821, "total_ms": 2260, "ts": "2026-04-08T10:23:01"}
{"type": "text", "sim_pre_ms": 7, "chat_engine_ms": 1654, "gpt_ms": 1590, "sim_post_ms": 6, "total_ms": 1672, "ts": "2026-04-08T10:23:15"}
```

---

## 影響的檔案

- `predictors/clip_iu/chat_engine.py` — `import time`；`generate_opening` / `chatting` 加 GPT 計時並回傳 gpt_ms；`post_method` 回傳 JSON 加入 `timing` 欄位
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py` — 新增 `_TIMING_LOG`、`_timed_worker()`、`_log_timing()` helpers；`_handle_new_image` 與 `interactive_running` 加入全路徑計時

## 已知限制

- `image decode/save` 時間未計入（通常 < 50ms，訊噪比低）
- timing_log.jsonl 為 append-only，需手動或排程清理
