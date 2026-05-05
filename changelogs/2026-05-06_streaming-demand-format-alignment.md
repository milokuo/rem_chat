# 2026-05-06 — Streaming 與 Accurate 路徑 Prompt 對齊

## 狀態：已實作

---

## 問題描述

同一張照片的多輪文字對話中：

- 第一輪（`/interact`）使用 Demand Format + `JSON_OUTPUT`。
- 第二輪若走 streaming（`/interact_stream` → `chat_engine /stream`），會切成 direct-reply prompt（`Do not output analysis steps or JSON`）。

結果是 dashboard 上可觀測到的 `full_prompt` / `raw_response` 格式與第一輪不一致，且行為上看起來像「模型沒遵守格式」。

---

## 根本原因

`chat_engine.py` 的 `/stream` endpoint 使用 `preprocess_conversation_simple()`，這條路徑刻意移除了 Demand Format / `JSON_OUTPUT` 指令。

---

## 修正內容

### 1) `chat_engine.py`：streaming 改走同一套 prompt 組裝

- `/stream` 由 `preprocess_conversation_simple()` 改為 `preprocess_conversation()`。
- streaming 仍保留 SSE 回傳，但改為：
  - 收完整 raw stream 文字（CoT + `JSON_OUTPUT`）。
  - 以 `postprocess_response_text()` 解析出最終回覆。
  - 前端 token 僅輸出乾淨回覆，不直接把 CoT 顯示給使用者。

### 2) `chat_engine.py`：抽出文字版後處理入口

- 新增 `postprocess_response_text(top_response)`，讓一般 `/` 與 `/stream` 共用同一解析邏輯。
- 原 `postprocess_response(response)` 保留，改為包裝呼叫文字版方法。

## 測試與驗證

- `python -m py_compile predictors/clip_iu/chat_engine.py`
- `python predictors/clip_iu/test_chat_engine_trace_fields.py`

其中 `test_chat_engine_trace_fields.py` 新增 streaming 專用測試，確認：

- streaming prompt 內含 `Demand Format` 與 `JSON_OUTPUT`
- 不再包含 direct-reply simple prompt 指令
- done event 同時保留 `raw_response` 與解析後 `full`

---

## 影響檔案

- `predictors/clip_iu/chat_engine.py`
- `predictors/clip_iu/test_chat_engine_trace_fields.py`
