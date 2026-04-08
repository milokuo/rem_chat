# 2026-03-10 — 多照片相冊 RAG 架構升級

## 背景

原系統每次只接受單張照片，即時跑影像分析後 GPT 僅有當前照片的資訊，
導致問題重複率高、容易偏離主題。本次升級引入 ChromaDB 向量資料庫，
實現跨照片語意檢索（RAG），並加入使用者管理功能。

---

## 新增檔案

### `predictors/clip_iu/photo_db.py`
ChromaDB 封裝層，所有模組透過此類別操作向量資料庫。

- `add_photo(photo_id, embedding, metadata)` — upsert 一筆照片記錄
- `query(embedding, n_results, patient_id)` — Top-K 相似度搜尋，可指定病患過濾
- `query_by_patient(patient_id)` — 列出某病患所有已索引照片
- `list_patients()` — 回傳 DB 中所有不重複的 patient_id
- `reset()` / `count()` — 工具方法

Collection schema（每筆）：
```
id           : "{patient_id}/{filename}"
embedding    : CLIP image embedding, 512-dim float list
caption      : BLIP 生成的描述
objects      : DETR 偵測到的物件字串
event        : CLIP 分類：事件標籤
place        : CLIP 分類：地點標籤
relationship : CLIP 分類：人物關係標籤
patient_id   : 病患 ID
filename     : 原始檔名
```

### `predictors/clip_iu/album_indexer.py`
離線批次腳本，預處理整個相冊並建立 ChromaDB 索引。

用法：
```bash
cd predictors/clip_iu
python album_indexer.py --album_dir ../../albums/patient_01 --patient_id patient_01
# --overwrite  重新索引已存在的照片（預設跳過）
# --db_dir     指定 ChromaDB 路徑（預設 ./photo_index）
```

流程：掃描資料夾 → 對每張照片呼叫 9205/9206/9207 → 取 embedding → 寫入 ChromaDB。
需要三個分析服務正在執行。

---

## 修改的現有檔案

### `predictors/clip_iu/clip_predictor.py`

1. 新增 `import torch`
2. 新增 `full_path` 支援：若 request 帶有 `full_path` 且檔案存在，直接使用該路徑，不再拼接 `img_dir`（供 album_indexer 批次處理任意路徑的照片使用）
3. 新增 512-dim image embedding 提取，加進回傳的 metadata JSON：
   ```python
   pixel_values = cp.processor(images=pil_image, return_tensors="pt")["pixel_values"]
   image_embedding = cp.model.get_image_features(pixel_values=pixel_values)
   image_embedding = image_embedding.squeeze(0).tolist()
   metadata['embedding'] = image_embedding
   ```
4. 修改 print：不再印出完整 metadata（避免 512-dim 向量洗掉 terminal log）

### `predictors/clip_iu/detr_detector.py`
新增 `full_path` 支援（同 clip_predictor.py，供 album_indexer 使用）。

### `predictors/clip_iu/image_caption.py`
新增 `full_path` 支援（同 clip_predictor.py，供 album_indexer 使用）。

### `predictors/clip_iu/chat_engine.py`

1. `SocialREMChat.__init__()` 新增 `self.retrieved_context = ""`
2. Flask `post_method()` 新增接收：
   ```python
   if 'retrieved_context' in data:
       _socialREMChat.retrieved_context = data['retrieved_context']
   ```
3. `preprocess_conversation()` 與 `generate_opening()` 兩處都插入 retrieved context block：
   ```
   [Related memories from the album]
   Photo 1 (event: birthday, place: home): a family celebrating...
   Photo 2 (event: travel, place: beach): people walking on...
   ```
   放在 observation_prompt 之後、Conversation History 之前。

### `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

#### 新增 import
```python
import sys
# PhotoDB path setup
from photo_db import PhotoDB
```

#### 新增全域變數
```python
PATIENT_ID = "default"   # 目前選擇的病患 ID
```

#### 新增 RAG helpers
- `_photo_db` — PhotoDB 實例，DB 路徑為 `predictors/clip_iu/photo_index/`
- `_users_file()` / `_load_users()` / `_persist_user()` — 管理 `uploads/users.json`，讓使用者清單在重啟後持久保存
- `_save_and_retrieve(photo_id, embedding, metadata, patient_id, n_results)` — 存入 DB 後，查詢同病患的 Top-K 相關照片（排除自身），回傳格式化字串給 GPT prompt

#### 圖片上傳流程修改（兩個路徑均更新）
- 存檔路徑：`uploads/{PATIENT_ID}/{filename}`（每個病患獨立資料夾，永久保留）
- 分析服務改傳 `full_path`（直接指向病患資料夾）
- RAG 區塊加 `try/except`，確保 ChromaDB 出錯時仍能回傳 GPT 回覆
- `msg_chat` 新增 `retrieved_context` 欄位傳給 chat_engine

#### 新增 Web UI — 使用者選擇列
在聊天區上方新增一列：下拉框（顯示所有使用者）＋ 新增輸入框 ＋「新增」按鈕。
JS 行為：
- 頁面載入時呼叫 `GET /users` 填入下拉框
- 選擇使用者 → `POST /set_user`
- 新增使用者 → `POST /set_user` → 重新載入清單

#### 新增 API 端點
- `GET /users` — 回傳 `{"users": [...], "current": "Jack"}`，合併 `users.json` 與 ChromaDB 中的病患清單
- `POST /set_user` — 更新 `PATIENT_ID`，並寫入 `users.json` 持久化

#### 初始化
```python
SHARED['image_embedding']   = []
SHARED['retrieved_context'] = ""
_persist_user(PATIENT_ID)   # 確保預設使用者寫入 users.json
```

---

## 資料夾結構（新增部分）

```
rem_chat/
  predictors/clip_iu/
    photo_db.py          ← NEW
    album_indexer.py     ← NEW
    photo_index/         ← ChromaDB 自動建立
  albums/
    <patient_id>/        ← 手動放置相冊（供 album_indexer 使用）
  ParlAI/projects/image_chat/
    uploads/
      users.json         ← 使用者清單持久化
      <patient_id>/      ← 各病患上傳的照片
```

---

## 依賴安裝

```bash
source predictors/clip_iu/clip_iu/bin/activate
pip install chromadb
```

---

## 已知問題與修復過程

| 問題 | 原因 | 修復 |
|------|------|------|
| `NameError: name 'image' is not defined` | `image` 只在 `predict()` 內部，外部取不到 | 改為 `Image.open(url)` 明確開啟 |
| embedding 格式錯誤 `[[[[...]]]]` | `**image_inputs` 帶入多餘 key 或 batch dim 未正確去除 | 改為明確傳 `pixel_values=`，用 `squeeze(0)` |
| ChromaDB 印出長向量 | ChromaDB 內部 debug logging | 加 `_SuppressChromaVerbose` logging Filter |
| 新使用者重整後消失 | `/users` 只查 ChromaDB，無照片的使用者查不到 | 加 `users.json` 持久化 |
| 上傳照片後無回覆 | `_save_and_retrieve` 出錯導致整個 `do_POST` 崩潰 | 加 `try/except`，失敗時 `retrieved_context = ''` 繼續執行 |
