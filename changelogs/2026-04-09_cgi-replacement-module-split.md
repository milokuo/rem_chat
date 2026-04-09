# 2026-04-09 — cgi 移除 + Server 模組拆分（Phase 2/3 品質修正）

## 背景

針對 `server_updated_zhengxuan.py` 的三項 HIGH 等級品質問題：

| ID | 問題 | 修正方式 |
|----|------|---------|
| H1 | `_save_and_retrieve()` 死碼（已被 `_memory_retrieve` 取代） | 上一 session 已刪除，929 行 |
| H3 | `import cgi`（Python 3.11+ DeprecationWarning，3.13 移除） | 以 `email.parser` stdlib 替換 |
| H2 | 檔案 929 行，超過 800 行上限 | 拆出 4 個輔助模組，降至 782 行 |

採用 TDD（RED → GREEN → REFACTOR）完成兩個 phase，共 28 個單元測試全部通過。

---

## Phase 3 — cgi 移除（H3）

### 問題

`cgi.FieldStorage` 自 Python 3.11 起標記為 deprecated，Python 3.13 已完全移除。
伺服器在 Python 3.11+ 啟動時會印出 `DeprecationWarning: 'cgi' is deprecated`。

### 解法

新增 `server_multipart.py`，以純 stdlib（`email.message_from_bytes`）實作等效的 multipart/form-data 解析器：

- `_parse_multipart(rfile, headers) -> dict[str, str]`
- `get_payload(decode=True)` 取得 raw bytes，再手動 UTF-8 decode，解決 `email.parser` 預設 latin-1 編碼造成的中文字元破碎問題
- 傳回 `dict`，呼叫端改用 `form.get("field", "")` 取代 `form["field"].value`

### 踩到的坑

- `get_payload(decode=False)` 傳回 latin-1 編碼的 str，UTF-8 中文字元被錯誤解讀 → 改用 `decode=True` 拿 bytes 再 decode
- Python 3.9 不支援 `str | None` PEP 604 union type → 改用 `Optional[str]` from `typing`

---

## Phase 2 — 模組拆分（H2）

### 拆出的模組

| 模組 | 內容 | 行數 |
|------|------|------|
| `server_multipart.py` | `_parse_multipart`、`_extract_name` | 74 |
| `server_timing.py` | `_TIMING_LOG`、`_timed_worker`、`_log_timing` | 33 |
| `server_users.py` | `_users_file`、`_load_users`、`_persist_user` | 37 |
| `server_conv_store.py` | `_get_conv_file`、`_save_conv_turn`、`_load_conv_turns` | 71 |
| `server_memory.py` | `_finalize_conversation_memory` | 89 |

主檔行數：929 → **782 行**（目標 ≤ 800）。

### 架構決策

**`uploads_dir` 改為參數傳入（而非讀全域變數）**
使四個輔助模組完全無全域依賴，測試可用 `tmpdir` 隔離，不需 mock 全域狀態。

**`_finalize_conversation_memory` 改為 keyword-only 依賴注入**
```python
_finalize_conversation_memory(
    photo_id, patient_id, uploads_dir,
    photo_db=_photo_db, openai_client=_openai_client,
    model=_MEMORY_MODEL, bg_semaphore=_bg_semaphore,
)
```
避免 server_memory.py 引入循環依賴或大量全域狀態。

**`format_retrieved_block` callback 以 lambda 適配**
`memory_retriever.format_retrieved_block` 期望簽名 `fn(photo_id, patient_id, n)`，而新的
`_load_conv_turns` 簽名多了 `uploads_dir`。在 call site 用 lambda 包裝，不改動外部模組：
```python
_conv_loader = lambda pid, phid, n=5: _load_conv_turns(pid, phid, SERVER_IMAGE_LOCATION, max_turns=n)
```

### 程式碼審查後修正（同次 commit）

- **M1**：移除 `server_updated_zhengxuan.py` 中多餘的 `_users_file` import（該函式僅在 `server_users.py` 內部使用）
- **M2**：`_load_users(SERVER_IMAGE_LOCATION)` 改為 `_load_users(SERVER_IMAGE_LOCATION, default_user=PATIENT_ID)`，確保無 `users.json` 時不產生 phantom `"default"` 使用者

---

## 測試覆蓋

### `test_parse_multipart.py`（7 tests）

| 測試 | 驗證內容 |
|------|---------|
| `test_single_text_field` | 單欄位解析 |
| `test_multiple_fields` | 多欄位解析 |
| `test_all_server_fields` | text/image_name/image/metadata/img_id/cate 全欄位 |
| `test_empty_body_returns_empty_dict` | body 為空時傳回 `{}` |
| `test_field_with_empty_value` | 空值欄位 |
| `test_unicode_value` | UTF-8 中文字元（「你好世界」）正確解析 |
| `test_missing_content_length_reads_zero` | 無 Content-Length header |

### `test_server_modules.py`（21 tests）

涵蓋 `server_timing`（6）、`server_users`（7）、`server_conv_store`（8）三個模組的完整功能。

**全套：28/28 通過，Python 3.9.13，pytest 8.4.1**

---

## 影響的檔案

**新增：**
- `ParlAI/projects/image_chat/server_multipart.py`
- `ParlAI/projects/image_chat/server_timing.py`
- `ParlAI/projects/image_chat/server_users.py`
- `ParlAI/projects/image_chat/server_conv_store.py`
- `ParlAI/projects/image_chat/server_memory.py`
- `ParlAI/projects/image_chat/test_parse_multipart.py`
- `ParlAI/projects/image_chat/test_server_modules.py`

**修改：**
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`（929 → 782 行，移除 cgi，拆出模組）

---

## 已知限制

- `server_memory.py` 在模組層級將 `predictors/clip_iu` 插入 `sys.path`（與 server 行為一致），若獨立 import（非透過 server 啟動）且 clip_env 未啟動，則 `memory_extractor` import 會失敗。此為開發階段已知限制，不影響生產場景。
- Smoke test（真實 HTTP multipart 請求）尚未執行；理論上應正確，單元測試已涵蓋所有欄位。
