# Changelog 索引

快速查閱各次更新的摘要。需要了解特定改動時，先看這份索引確認對應的日誌檔案，再去閱讀詳細內容。

---

## 2026-05-06 — Streaming 與 Accurate 路徑 Prompt 對齊
**檔案：** `2026-05-06_streaming-demand-format-alignment.md`

**狀態：** 已實作

**核心變更：** 修正 `/stream` 使用了不同 prompt 的行為差異，讓 streaming 與非 streaming 路徑都使用 Demand Format + `JSON_OUTPUT`

| 類別 | 內容 |
|------|------|
| Bug fix | `chat_engine.py /stream` 改走 `preprocess_conversation()`，不再使用 simple direct-reply prompt |
| Bug fix | `chat_engine.py` 新增 `postprocess_response_text()`，streaming 與非 streaming 共用同一回覆解析邏輯 |
| Test | `test_chat_engine_trace_fields.py` 新增 streaming 對齊測試（prompt 內容與回覆欄位） |

**影響的檔案：**
- `predictors/clip_iu/chat_engine.py`
- `predictors/clip_iu/test_chat_engine_trace_fields.py`
- `changelogs/2026-05-06_streaming-demand-format-alignment.md`（新增）

---

## 2026-05-06 — Debug Dashboard 修正與 Memory Browser
**檔案：** `2026-05-06_dashboard-fixes.md`

**狀態：** 已實作

**核心變更：** 修正 streaming 模式 full_prompt 為空陣列；新增 Memory Browser 分頁，直接查詢 ChromaDB（不需 8082）

| 類別 | 內容 |
|------|------|
| Bug fix | `chat_engine.py /stream` 現在儲存 `_last_full_prompt` 並在 done event 攜帶 `full_prompt` |
| Bug fix | `server_updated_zhengxuan.py` streaming trace 改從 SSE done payload 讀 `full_prompt`，不再寫死 `[]` |
| 新功能 | `debug_dashboard.py` 新增 `/api/memory/patients`、`/api/memory/<patient_id>` 路由 |
| 新功能 | Dashboard HTML 新增 Memory Browser 分頁（病患清單 + 照片卡片 + conv_summary 顯示）|

**影響的檔案：**
- `predictors/clip_iu/chat_engine.py`（修正）
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`（修正）
- `ParlAI/projects/image_chat/debug_dashboard.py`（修改）
- `ParlAI/projects/image_chat/templates/debug_dashboard.html`（修改）
- `CLAUDE.md`（文件更新）

---

## 2026-04-09 — Frontend UX 改善 + GPT Streaming（Phases 1–7）
**檔案：** `2026-04-09_frontend-ux-streaming.md`

**狀態：** 已實作

**核心變更：** 使用者訊息即時顯示；Bot thinking indicator；ThreadingHTTPServer；GPT 直接輸出中文；chat_engine `/stream` SSE 端點；server `/interact_stream` proxy；精確模式 toggle；新增使用者 bug 修正（快取 + 錯誤處理）

| 類別 | 內容 |
|------|------|
| Phase 1–2 | 使用者訊息即時顯示 + typing indicator（純前端） |
| Phase 3–4 | ThreadingHTTPServer + HTTP/1.1；GPT 直接輸出繁體中文 |
| Phase 5 | `chat_engine.py` 新增 `preprocess_conversation_simple()` + `/stream` SSE 端點 |
| Phase 6 | `server_updated_zhengxuan.py` 新增 `_stream_interact()` + `/interact_stream` 路由 |
| Phase 7 | 前端 streaming 接收（SSE reader + TextDecoder flush + error 顯示）+ 精確模式 toggle |
| Bug fix | `GET /users` Cache-Control + addUserBtn 錯誤處理 + `list_patients` 保護 |

**影響的檔案：**
- `predictors/clip_iu/chat_engine.py`（修改）
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`（修改）

---

## 2026-04-09 — RAG 檢索失敗修正：NumPy Embedding Bool 判斷錯誤
**檔案：** `2026-04-09_numpy-embedding-bugfix.md`

**狀態：** 已修正

**核心變更：** `memory_retriever.py` 的 `if emb` → `if emb is not None`，修正 ChromaDB 回傳 NumPy array 時 RAG 靜默失敗導致跨照片記憶完全消失的 bug

| 類別 | 內容 |
|------|------|
| Bug | `ValueError: The truth value of an array with more than one element is ambiguous` |
| 原因 | ChromaDB embedding 為 NumPy array，`if emb:` 對 array 不合法 |
| 修正 | `memory_retriever.py` L171 + `photo_db.py` L102 改用 `is not None` |

**影響的檔案：**
- `predictors/clip_iu/memory_retriever.py`（修正）
- `predictors/clip_iu/photo_db.py`（修正）

---

## 2026-04-09 — Debug Dashboard 可觀測性系統（Phases 1–4）
**檔案：** `2026-04-09_debug-dashboard-observability.md`

**狀態：** 已實作，41 tests 全部通過

**核心變更：** port 8090 debug dashboard；RAG score breakdown（visual/entity/theme/recency）；chat_engine 暴露 full_prompt + raw_response；server fire-and-forget trace

| 類別 | 內容 |
|------|------|
| Phase 1 | `memory_retriever.py` 加 `_entity_score`、`_theme_match`、`_recency_score` |
| Phase 2 | `debug_dashboard.py` Flask app + `debug_dashboard.html` 雙欄 UI（16+17 tests）|
| Phase 3 | `chat_engine.py` response 加 `full_prompt`、`raw_response`、`model_name`（13 tests）|
| Phase 4 | `server_updated_zhengxuan.py` 加 `_post_trace()` daemon thread + 兩個 call sites（11 tests）|

**影響的檔案：**
- `predictors/clip_iu/memory_retriever.py`（修改）
- `predictors/clip_iu/test_memory_retriever.py` ← NEW
- `ParlAI/projects/image_chat/debug_dashboard.py` ← NEW
- `ParlAI/projects/image_chat/templates/debug_dashboard.html` ← NEW
- `ParlAI/projects/image_chat/test_debug_dashboard.py` ← NEW
- `predictors/clip_iu/chat_engine.py`（修改）
- `predictors/clip_iu/test_chat_engine_trace_fields.py` ← NEW
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`（修改）
- `ParlAI/projects/image_chat/test_server_trace.py` ← NEW

---

## 2026-04-09 — cgi 移除 + Server 模組拆分（H2/H3 品質修正）
**檔案：** `2026-04-09_cgi-replacement-module-split.md`

**狀態：** 已修正並提交

**核心變更：** 移除棄用的 `import cgi`，以 `email.parser` stdlib 替換；將 929 行主檔拆為 5 個獨立模組（782 行）；28/28 tests 通過

| 類別 | 內容 |
|------|------|
| H3 修正 | 新增 `server_multipart.py`，`_parse_multipart` 取代 `cgi.FieldStorage` |
| H2 修正 | 拆出 `server_timing/users/conv_store/memory.py` 四個模組 |
| 測試 | `test_parse_multipart.py`（7）+ `test_server_modules.py`（21）|

**影響的檔案：**
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`（修改）
- `ParlAI/projects/image_chat/server_multipart.py` ← NEW
- `ParlAI/projects/image_chat/server_timing.py` ← NEW
- `ParlAI/projects/image_chat/server_users.py` ← NEW
- `ParlAI/projects/image_chat/server_conv_store.py` ← NEW
- `ParlAI/projects/image_chat/server_memory.py` ← NEW
- `ParlAI/projects/image_chat/test_parse_multipart.py` ← NEW
- `ParlAI/projects/image_chat/test_server_modules.py` ← NEW

---

## 2026-04-09 — 三路記憶系統 Bug 修正（Codex 驗證）
**檔案：** `2026-04-09_memory-bugfix-codex.md`

**狀態：** 已修正並提交

**核心變更：** 修正三路檢索退化、evidence_check 無法啟用、re-upload 記憶覆蓋、背景 thread 無上限等四項問題

| 類別 | 內容 |
|------|------|
| P1 修正 | 上傳前先 fetch existing_meta，revisit 時使用已有 theme/entities 做三路檢索 |
| P2a 修正 | `config.py` 新增 `--evidence_check` flag，evidence check 現在可以啟用 |
| P2b 修正 | add_photo 前 merge 已有豐富化欄位，re-upload 不再覆蓋 theme/entities/conv_summary |
| P3 修正 | 新增 `_bg_semaphore = Semaphore(4)`，限制同時執行的背景 thread 上限 |

**影響的檔案：**
- `predictors/clip_iu/config.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

---

## 2026-04-09 — Photo-Anchored Autobiographical Memory 實作
**檔案：** `2026-04-09_autobiographical-memory-impl.md`

**狀態：** 已實作

**核心變更：** 三路記憶檢索（visual + theme + entity + recency rerank）、換圖時背景 finalize、上傳時背景 GPT enrichment、輕量 evidence check（預設關閉）

| 類別 | 內容 |
|------|------|
| 新增 | `memory_extractor.py`（GPT theme/entity 抽取 + session summary） |
| 新增 | `memory_retriever.py`（三路檢索 + rerank + lazy fallback + 結構化 prompt） |
| 修改 | `photo_db.py`（update_metadata / query_by_theme / 擴充 query_by_patient） |
| 修改 | `album_indexer.py`（兩段式：基礎 + `--enrich` GPT 補充） |
| 修改 | `server_updated_zhengxuan.py`（換圖 finalize、上傳 enrich、換三路 retriever） |
| 修改 | `chat_engine.py`（Rule 3 主動引用、evidence check 方法） |

**影響的檔案：**
- `predictors/clip_iu/photo_db.py`
- `predictors/clip_iu/memory_extractor.py` ← NEW
- `predictors/clip_iu/memory_retriever.py` ← NEW
- `predictors/clip_iu/album_indexer.py`
- `predictors/clip_iu/chat_engine.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

---

## 2026-04-09 — [PLAN] Photo-Anchored Autobiographical Memory 設計計畫
**檔案：** `2026-04-09_photo-anchored-memory-plan.md`

**狀態：** 計畫中，尚未實作

**核心設計：** 將 Jung-Min 論文的 Hierarchical Autobiographical Memory 以照片為錨點移植進系統。4 層記憶結構（Theme/Lifetime Period/General Event/Episodic）、三路檢索（visual+theme+entity）、recency rerank、主動引用 prompt。

| 類別 | 內容 |
|------|------|
| 新增 | `memory_extractor.py`（theme/entity 抽取）、`memory_retriever.py`（三路檢索 + rerank）|
| 修改 | `photo_db.py`（schema 擴充）、`album_indexer.py`（上傳豐富化）|
| 修改 | `server_updated_zhengxuan.py`（write-back + 換 retriever）、`chat_engine.py`（新 prompt）|
| 不做 | 臉部 clustering、EXIF 時間軸、情緒驅動選圖 |

**影響的檔案（預期）：**
- `predictors/clip_iu/photo_db.py`
- `predictors/clip_iu/memory_extractor.py` ← NEW
- `predictors/clip_iu/memory_retriever.py` ← NEW
- `predictors/clip_iu/album_indexer.py`
- `predictors/clip_iu/chat_engine.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

---

## 2026-04-08 — 部件時間消耗分析
**檔案：** `2026-04-08_timing-analysis.md`

**核心變更：** 在 server (8082) 與 chat_engine (8087) 加入全路徑計時儀器，記錄至 `timing_log.jsonl`

| 類別 | 內容 |
|------|------|
| 新增 | `_timed_worker()`、`_log_timing()` helpers；`_TIMING_LOG` 絕對路徑 |
| 計時 | 新圖流程：sim_reset / CLIP / DETR / BLIP（各別）/ RAG / chat_engine / GPT |
| 計時 | 文字流程：sim_pre / chat_engine / GPT / translate(zh) / sim_post |
| 傳遞 | chat_engine 回傳 `timing.gpt_ms`，server 整合至同一筆日誌 |

**影響的檔案：**
- `predictors/clip_iu/chat_engine.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

---

## 2026-04-08 — chat_engine 重構：消除重複程式碼 + 強化 CoT 解析
**檔案：** `2026-04-08_chat-engine-refactor.md`

**核心變更：** 抽出 `_build_retrieved_block()`；instruct_prompt 加 JSON_OUTPUT tag，postprocess_response 優先以 JSON 解析取代脆弱正則

| 類別 | 內容 |
|------|------|
| 重構 | `_build_retrieved_block()` 消除兩處 ~15 行重複程式碼 |
| 強化 | JSON_OUTPUT tag 作為解析第一優先，原 step-9 regex / fallback 自動降級保留相容 |

**影響的檔案：**
- `predictors/clip_iu/chat_engine.py`

---

## 2026-04-08 — 圖片分析並行化與上傳邏輯重構
**檔案：** `2026-04-08_parallel-analysis-refactor.md`

**核心變更：** CLIP/DETR/BLIP 三服務改為並行呼叫（ThreadPoolExecutor），上傳處理邏輯抽出共用方法消除重複

| 類別 | 內容 |
|------|------|
| 效能 | 圖片上傳延遲從 t1+t2+t3 降至 max(t1,t2,t3) |
| 重構 | 抽出 `_handle_new_image()`，消除 path-based / base64 兩個上傳分支的 ~65 行重複程式碼 |

**影響的檔案：**
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

---

## 2026-03-11 — GPT-5-mini 升級與對話 Prompt 修正
**檔案：** `2026-03-11_gpt5mini-prompt-fix.md`

**核心變更：** 升級預設模型至 gpt-5-mini，並修正三個因新模型指令遵循行為改變暴露的問題

| 類別 | 內容 |
|------|------|
| 模型 | `config.py` 預設 model 改為 `gpt-5-mini` |
| Prompt | Retrieved context block 改為 Rule 1/2，允許模型回答使用者對過去對話的提問 |
| 開場白 | `generate_opening` 改用 CoT + `postprocess_response`，避免分析內容外洩 |
| 相容性 | `generate_kwargs` 依模型名稱自動略過 gpt-5 不支援的 sampling 參數 |
| 解析 | `postprocess_response` 改為直接搜尋 step 9，對缺少換行的輸出格式更穩健 |

**影響的檔案：**
- `predictors/clip_iu/config.py`
- `predictors/clip_iu/chat_engine.py`

---

## 2026-03-10 — 多照片相冊 RAG 架構升級
**檔案：** `2026-03-10_rag-upgrade.md`

**核心變更：** 從單張照片即時分析，升級為支援相冊預處理＋跨照片語意檢索（RAG）

| 類別 | 內容 |
|------|------|
| 新增檔案 | `photo_db.py`（ChromaDB 封裝）、`album_indexer.py`（批次建索引腳本） |
| 影像服務 | `clip_predictor.py` 新增回傳 512-dim image embedding；三個服務均新增 `full_path` 支援 |
| 對話引擎 | `chat_engine.py` 接收 `retrieved_context`，插入 GPT system prompt |
| Web 伺服器 | `server_updated_zhengxuan.py` 新增使用者選擇 UI、`/users`、`/set_user` 端點；照片按病患分資料夾儲存；RAG 查詢邏輯 |
| 依賴 | `chromadb`（需在 clip_env 安裝） |

**影響的檔案：**
- `predictors/clip_iu/clip_predictor.py`
- `predictors/clip_iu/detr_detector.py`
- `predictors/clip_iu/image_caption.py`
- `predictors/clip_iu/chat_engine.py`
- `predictors/clip_iu/photo_db.py` ← NEW
- `predictors/clip_iu/album_indexer.py` ← NEW
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`
