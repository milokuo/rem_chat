# RAG 檢索失敗修正：NumPy Embedding Bool 判斷錯誤

**日期：** 2026-04-09  
**狀態：** 已修正並提交

---

## 問題描述

上傳第二張照片時，RAG 系統未能成功 retrieve 第一次聊天的記憶，導致 GPT 開場問題缺乏跨照片上下文。

Terminal 出現：
```
[RAG] Error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

## 根本原因

ChromaDB 的 `collection.get(include=["embeddings"])` 回傳的 embedding 向量為 **NumPy array**，而非 Python list。

在 `memory_retriever.py` 中，程式碼以 truthiness 判斷 embedding 是否存在：

```python
# 修正前（有問題）
visual = _cosine_similarity(query_embedding, emb) if emb else 0.0
```

`if emb` 對 NumPy array（長度 > 1）會 raise `ValueError: The truth value of an array with more than one element is ambiguous.`

這個 exception 被上層 `except Exception as e:` 捕捉，使 `SHARED['retrieved_context'] = ''` — **RAG 完全靜默失敗，photo1 的所有記憶被清空。**

## 修正內容

### `predictors/clip_iu/memory_retriever.py` — 主要 bug

```python
# 修正前
visual = _cosine_similarity(query_embedding, emb) if emb else 0.0

# 修正後
visual = _cosine_similarity(query_embedding, emb) if emb is not None else 0.0
```

用 `is not None` 取代 truthiness 判斷，正確處理 NumPy array。

### `predictors/clip_iu/photo_db.py` — 防禦性修正

```python
# 修正前
if with_embeddings and results.get("embeddings"):

# 修正後
if with_embeddings and results.get("embeddings") is not None:
```

同樣邏輯，防止 ChromaDB 回傳 numpy array list 時出現相同問題。

## 驗證

修正後，上傳第二張照片的 log 應出現：
```
[Retrieved context]:
[Memory 1 — related past photo]
  Caption: ...
  [Past conversation (most recent turns):]
    User said: ...
    Assistant replied: ...
  [End past conversation]
```

而非 `[RAG] Error: The truth value of an array...`。

## 影響的檔案

- `predictors/clip_iu/memory_retriever.py`（修正）
- `predictors/clip_iu/photo_db.py`（修正）
