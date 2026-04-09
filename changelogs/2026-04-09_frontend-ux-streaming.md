# Frontend UX 改善 + GPT Streaming（Phases 1–7）

**日期：** 2026-04-09  
**狀態：** 已實作

---

## 需求背景

使用者反映兩個明顯的 UX 問題：
1. 送出訊息後要等 GPT 回應完才看到自己的輸入，有延遲感
2. GPT 回應期間畫面無任何反饋，不知道是否有在處理

同時希望利用 streaming 模式加速回應顯示速度，並讓精確度可選。

---

## 實作概覽（Phases 1–7）

### Phase 1 — 使用者訊息即時顯示
- 修改 `fetchResult()`：送出 fetch **之前**就把使用者訊息 append 到畫面
- 之前是 fetch resolve 後才顯示，現在改為即時

### Phase 2 — Bot thinking indicator + submit 鎖
- 新增 `createThinkingRow()`：三個跳動灰點（CSS `typing-bounce` animation）
- fetch 發出後插入 thinking row，收到回應後 replace 成實際訊息
- 新增 `_submitting` flag + `respond` button disabled，防止重複送出
- `finally` block 確保 indicator 在任何情況（成功/失敗/中斷）下都移除

### Phase 3 — ThreadingHTTPServer + HTTP/1.1
- 把 `HTTPServer` 換成 `ThreadingHTTPServer`
- `Handler.protocol_version = "HTTP/1.1"`
- 避免 streaming 連線開著時把其他 request 都擋住

### Phase 4 — GPT 直接輸出繁體中文
- `preprocess_conversation()` 和 `generate_opening()` 的 lang_suffix 調整
- zh 模式在 prompt 末尾加 `"請以繁體中文（台灣用語）回覆使用者。"`
- 移除 `trans_en_zh.translate()` 呼叫（省 latency，GPT-4o 中文品質好過翻譯層）

### Phase 5 — chat_engine `/stream` SSE 端點（`chat_engine.py`）
新增兩個方法 / 端點：

**`preprocess_conversation_simple()`**
- 同 `preprocess_conversation` 的 system task + strategies + observations + retrieved block
- 移除 CoT `instruct_prompt`（9 步驟 + JSON_OUTPUT）
- 改用一句簡短指令：「回覆使用者（最多2句話），並選擇一個合適的支援策略。」

**`@app.route("/stream")`**
- 接收同樣的 JSON payload（`user_message`, `caption_str`, `obj_str`, `retrieved_context`, `lang`）
- 呼叫 OpenAI API `stream=True`
- 逐 token 以 SSE 格式輸出：`data: {"token": "..."}\n\n`
- 結束時送 `data: {"done": true, "full": "..."}\n\n`
- 只有 `full_text` 非空時才 append 到 `_socialREMChat.context`（避免例外時插入空 turn）
- 保留 end_trigger 檢查（遇到結束指令直接送 closing token）

### Phase 6 — server `/interact_stream` SSE proxy（`server_updated_zhengxuan.py`）
新增兩個 handler 方法：

**`_write_chunked(data: bytes)`**
- 封裝 HTTP chunked-transfer-encoding 格式（hex_len + CRLF + data + CRLF）

**`_stream_interact(postvars)`**
- 副作用完整保留，順序：sim_pre → proxy stream → sim_post + conv_save + trace
- 送出 SSE headers（`Content-Type: text/event-stream`, `Transfer-Encoding: chunked`）
- 以 `requests.post(..., stream=True)` 接收 chat_engine 的 SSE 並逐行轉發至瀏覽器
- HTTP status 非 200 時 raise，進入 except 送 error 事件
- `got_done` flag：確保不管成功/失敗都會送 `{"done": true}` 給前端，不讓前端永遠 pending
- 呼叫 `_log_timing(timing)` 補全 timing_log.jsonl 覆蓋（之前只有 print + trace）

**`do_POST` 新增 `/interact_stream` 路由**
- 只接受 text-only 請求（圖片上傳仍走 `/interact`）
- 無 text 時回 400

### Phase 7 — 前端 streaming 接收 + format-align toggle

**UI 新增**：精確模式 checkbox（`#formatAlignToggle`），預設**未勾選**（= streaming 模式），顯示在 Submit 按鈕下方。

**`createStreamingBotRow()`**：建立帶 `textSpan` 引用的 bot 訊息列，token 進來直接 `textContent +=`。

**`fetchResult()` 分支邏輯**：
```
useStream = (image_data === "") && !formatAlignToggle.checked
```
- 有圖片：永遠走 `/interact`（accurate mode）
- 純文字 + 精確模式勾選：走 `/interact`
- 純文字 + 精確模式未勾選（預設）：走 `/interact_stream`

**Streaming 接收**：
- `response.ok` 檢查，非 200 立即顯示錯誤碼
- `payload.error` 收到時在 bot row 顯示錯誤文字
- `TextDecoder` final flush：`result.done` 時呼叫 `decoder.decode()`（無參數）排空剩餘 bytes

---

## Codex Code Review 修正（同日）

Codex 審查後發現的問題，於同 session 修正：

| 嚴重度 | 問題 | 修正 |
|--------|------|------|
| P1 | streaming 錯誤靜默失敗 | HTTP status check + 保證 done 事件 + 前端顯示 error |
| P2 | 例外後空字串寫入 context | `if full_text:` 才 append |
| P2 | stream 路徑缺 `_log_timing` | 補加 `_log_timing(timing)` |
| P3 | TextDecoder 未 final flush | `result.done` 時 `decoder.decode()` flush |

---

## 新增使用者按鈕 bug 修正（同日）

**症狀**：在文字框輸入名稱後點「新增」，下拉清單未更新。

**根因分析（Codex）與確認修正**：

| 問題 | 修正位置 | 修正內容 |
|------|----------|----------|
| `GET /users` 瀏覽器快取 | `do_GET /users` | 加 `Cache-Control: no-store` header |
| `loadUsers()` fetch 快取 | JS `loadUsers()` | `fetch('/users', {cache: 'no-store'})` |
| `addUserBtn` 無錯誤處理 | JS event listener | 加 `response.ok` 檢查 + `.catch()` + alert |
| `_photo_db.list_patients()` 無保護 | `do_GET /users` | 包進 try/except，DB 異常不影響 UI |

---

## 影響的檔案

| 檔案 | 修改類型 |
|------|----------|
| `predictors/clip_iu/chat_engine.py` | 新增 `preprocess_conversation_simple()`、`/stream` 端點、Flask Response import |
| `ParlAI/projects/image_chat/server_updated_zhengxuan.py` | Phase 1–4 前端 JS、ThreadingHTTPServer、`_write_chunked()`、`_stream_interact()`、`/interact_stream` 路由、精確模式 toggle UI、bug 修正 |

---

## 架構說明

```
瀏覽器 (streaming 模式)
  │ POST /interact_stream (multipart)
  ▼
server_updated_zhengxuan.py (port 8082)
  │ _stream_interact()
  │ sim_pre
  │ POST http://127.0.0.1:8087/stream (JSON, stream=True)
  ▼
chat_engine.py (port 8087)
  │ post_stream()
  │ preprocess_conversation_simple()
  │ openai.chat.completions.create(stream=True)
  │ yield SSE tokens
  ▼
server 逐行轉發 → 瀏覽器
  sim_post + conv_save + _post_trace + _log_timing (stream 結束後)
```

## 已知限制

- `_socialREMChat.context` 仍為全域共享物件；同時有多個請求時可能發生 context 污染。在單使用者療癒場景下風險低，但若未來需要多 session 並行需重構為 per-request 物件。
- Streaming 模式下沒有 CoT 步驟，無法做 evidence_check grounding 驗證。
