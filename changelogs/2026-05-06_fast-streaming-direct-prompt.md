# 2026-05-06 — Fast Streaming 恢復 Direct-Reply Prompt

## 狀態：已實作

---

## 問題描述

5/6 先前為了讓 streaming 與 accurate trace 對齊，將 `chat_engine.py /stream` 改成與 `/` 相同的 Demand Format + `JSON_OUTPUT` prompt。

這雖然讓 dashboard 看到的 prompt 格式一致，但因為 `/stream` 必須先收完整 raw response、postprocess 後才能把乾淨回覆送給前端，實際上變成「SSE 包裝的一次性回覆」，使用者體感與精確模式幾乎相同。

---

## 修正內容

### `chat_engine.py`

- `/stream` 改回使用 `preprocess_conversation_simple()`。
- fast streaming prompt 不包含 Demand Format / `JSON_OUTPUT` / CoT。
- fast streaming prompt 不再包含「選擇 Support Strategy 並解釋原因」的完整策略推理段落，避免模型把 `選用策略`、`原因` 等 debug 文字輸出給使用者。
- fast streaming prompt 明確要求：支援策略只作為內部風格使用，輸出只能是使用者看得到的一到兩句自然回覆，並依使用者最後一句話的語言回覆。
- OpenAI streamed chunk 一收到 token 就立刻以 SSE `token` event 送給前端。
- `raw_response` 保留完整 direct reply，`full_prompt` 仍寫入 done event 供 debug dashboard 使用。
- streaming 模式不做 evidence check，避免額外 GPT call 與 buffering 破壞即時體感。

### `test_chat_engine_trace_fields.py`

- 更新 streaming 測試語意：
  - prompt 應包含 direct-reply 指令。
  - prompt 不應包含舊的策略解釋段落（`Then select a Support Strategy...` / definitions）。
  - prompt 不應包含 `Demand Format` / `JSON_OUTPUT`。
  - token event 應分段出現，而不是最後一次整包送出。

### `server_updated_zhengxuan.py`

- streaming UI 分支也先顯示 `createThinkingRow()` 灰點佔位符。
- 第一個 token / error / done payload 到達時，將灰點 row 替換成 streaming bot row。
- 避免 fast streaming 在第一個 token 前呈現空白訊息列，讓使用者知道 GPT API 正在處理。

---

## 行為差異

| UI 模式 | 路由 | Prompt | 顯示方式 |
|---|---|---|---|
| 未勾精確模式 | `/interact_stream` → `chat_engine /stream` | simple direct reply | token 到就顯示 |
| 勾選精確模式 | `/interact` → `chat_engine /` | Demand Format + `JSON_OUTPUT` | 完整 postprocess 後一次顯示 |
| 圖片上傳 | `/interact` | opening prompt + Demand Format | 完整 postprocess 後一次顯示 |

---

## 驗證

- `python -m py_compile predictors/clip_iu/chat_engine.py`
- `python predictors/clip_iu/test_chat_engine_trace_fields.py`
- `python -m py_compile ParlAI/projects/image_chat/server_updated_zhengxuan.py`
- `python ParlAI/projects/image_chat/test_server_trace.py`

---

## 影響檔案

- `predictors/clip_iu/chat_engine.py`
- `predictors/clip_iu/test_chat_engine_trace_fields.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`
