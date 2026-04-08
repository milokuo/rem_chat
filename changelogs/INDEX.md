# Changelog 索引

快速查閱各次更新的摘要。需要了解特定改動時，先看這份索引確認對應的日誌檔案，再去閱讀詳細內容。

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
