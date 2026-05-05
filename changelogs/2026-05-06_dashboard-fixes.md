# 2026-05-06 — Debug Dashboard 修正與 Memory Browser 功能

## 狀態：已實作

---

## 問題一：Streaming 模式 full_prompt 為空陣列

### 症狀
Debug dashboard 第二次對話（streaming 模式）顯示 `"full_prompt": []`，即看不到 GPT 實際收到的 prompt。

### 根本原因
兩處 bug 同時存在：

1. **`chat_engine.py` `/stream` endpoint**：`preprocess_conversation_simple()` 建構好 prompt 後，沒有存進 `_last_full_prompt`，也沒有把 prompt 放進 SSE done event。
2. **`server_updated_zhengxuan.py` streaming trace**：直接寫死 `"full_prompt": []`，不管 chat_engine 回傳什麼。

### 修正
- `chat_engine.py`：`/stream` handler 加上 `_socialREMChat._last_full_prompt = processed_context`；SSE done event 加入 `full_prompt` 欄位。
- `server_updated_zhengxuan.py`：從 SSE done payload 解析 `full_prompt`，傳給 `_post_trace`。

---

## 功能新增：Debug Dashboard Memory Browser 分頁

### 背景
Dashboard 原本只能看 in-memory trace buffer（最多 50 筆，8090 重啟即清空）。如果 8090 沒有在對話期間運行，那段對話的 trace 就永久遺失。此外，無法直接觀察 ChromaDB 裡存了哪些記憶。

### 新增功能

**後端（`debug_dashboard.py`）：**
- `GET /api/memory/patients` — 列出 ChromaDB 內所有 `patient_id` 及各自的照片數量
- `GET /api/memory/<patient_id>` — 回傳該病患所有已索引照片的 metadata（不含 embedding）；已討論（有 `conv_summary`）的排前面

**前端（`debug_dashboard.html`）：**
- Sidebar 加入 **Traces / Memory** 分頁切換
- Memory 分頁：左側顯示病患清單（附照片計數徽章），右側顯示該病患所有照片卡片
- 每張卡片顯示：縮圖、photo_id、caption、theme、entities（people/activities/locations）、event、place、上傳時間、最後對話時間
- 已討論的照片（有 `conv_summary`）綠色左邊框 + 顯示摘要；未討論的灰色 + 顯示「Not yet discussed」

**重要：** Memory 分頁直接讀取磁碟上的 ChromaDB，**不需要 8082 運行**，8090 自己就能查詢所有病患記憶。

---

## 影響的檔案

| 檔案 | 變更 |
|------|------|
| `predictors/clip_iu/chat_engine.py` | 修正：`/stream` 儲存 `_last_full_prompt` + done event 含 `full_prompt` |
| `ParlAI/projects/image_chat/server_updated_zhengxuan.py` | 修正：streaming trace 從 SSE done event 讀取 `full_prompt` |
| `ParlAI/projects/image_chat/debug_dashboard.py` | 新增：`/api/memory/patients`、`/api/memory/<patient_id>` 路由；`_open_collection()` helper |
| `ParlAI/projects/image_chat/templates/debug_dashboard.html` | 新增：Memory Browser 分頁（tab switcher + patient list + photo cards） |
| `CLAUDE.md` | 文件：補充 port 8090 說明及啟動指令 |
