# 三路記憶系統 Bug 修正（Codex 驗證）

**日期：** 2026-04-09
**類型：** Bug Fix
**Commit：** TBD

---

## 背景

Codex 對昨日實作的 Photo-Anchored Autobiographical Memory 系統做靜態分析，指出四個問題。本次修正針對所有問題逐一驗證並修正。

---

## 問題與修正

### [P1] 三路檢索幾乎退化為 visual-only

**問題：** `_memory_retrieve` 呼叫時固定傳 `query_theme=""` 和 `query_entities={}`。
對新圖這是正確設計（enrichment 為非同步，尚未完成），但對**重複上傳的舊圖**，
ChromaDB 中已有 theme/entities，程式碼卻完全忽略這些信號。

**修正（`server_updated_zhengxuan.py`）：**
在 `add_photo` 之前先呼叫 `query_by_patient` 拿取 `_existing_meta`，
抽出 `_query_theme` 和 `_query_entities`（四個欄位 JSON decode）傳進 `_memory_retrieve`。
新圖 first-upload 時 `_existing_meta` 為空 dict，行為與原本相同（visual-only fallback）。

---

### [P2a] `_check_response_grounding` 幾乎無法啟用

**問題：** `chat_engine.py` 讀 `args.evidence_check`，但 `config.py` 沒有該參數，
`getattr(args, 'evidence_check', False)` 永遠回傳 `False`。

**修正（`config.py`）：**
```python
parser.add_argument('--evidence_check', action='store_true', default=False,
                    help='Enable post-generation evidence grounding check (adds latency)')
```
啟用方式：`python chat_engine.py --evidence_check`。

---

### [P2b] 同 photo_id 重上傳時覆蓋記憶欄位

**問題：** `add_photo` 呼叫 ChromaDB `upsert`，傳入的 metadata dict 只含基礎欄位
（caption/objects/event/place/relationship/patient_id/filename）。ChromaDB upsert 會**完整取代** metadata，
導致先前 enrichment 寫入的 theme、entities_*、conv_summary、last_chatted 全部消失。

**修正（`server_updated_zhengxuan.py`）：**
在 upsert 前 fetch `_existing_meta`，然後把以下欄位（若非空）merge 進 `db_metadata`：
```
theme, entities_people, entities_activities, entities_locations,
entities_objects, conv_summary, last_chatted, upload_timestamp
```
這樣 re-upload 只更新 caption/objects/event/place（圖片重新分析），
記憶欄位保留不動。

---

### [P3] daemon fire-and-forget 無上限控制

**問題：** 每次換圖都新開 thread（`_finalize_conversation_memory` + `_enrich_photo`），
無佇列與數量上限。快速切圖理論上可以累積大量 background thread。

**修正（`server_updated_zhengxuan.py`）：**
新增模組層級 `_bg_semaphore = threading.Semaphore(4)`，
兩個背景 thread 皆透過 wrapper 函式 acquire semaphore：

```python
# _finalize_conversation_memory
def _run_sem():
    with _bg_semaphore:
        _run()
threading.Thread(target=_run_sem, daemon=True).start()

# _enrich_photo
def _enrich_photo_sem():
    with _bg_semaphore:
        _enrich_photo()
threading.Thread(target=_enrich_photo_sem, daemon=True).start()
```

`daemon=True` 保留：這是 Flask development server，背景任務不應阻止 shutdown。
丟資料風險（process exit 時）對單一患者療程場景屬可接受範圍，已知限制。

---

## 修改的檔案

| 檔案 | 類型 | 說明 |
|------|------|------|
| `predictors/clip_iu/config.py` | 修改 | 新增 `--evidence_check` flag |
| `ParlAI/projects/image_chat/server_updated_zhengxuan.py` | 修改 | P1+P2b 現有記憶保護、P3 semaphore |

---

## 已知限制（未修正）

- daemon thread 在 process exit 時仍有丟資料風險（需要 graceful shutdown handler 才能完全解決）
- 若 ChromaDB 版本 < 0.4，`query_by_theme` 的 `$and` filter 語法可能失敗（已有 try/except fallback 降級為空結果）
