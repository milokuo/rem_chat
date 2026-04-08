# 2026-04-08 — 圖片分析並行化與上傳邏輯重構

## 核心變更

### #1 圖片分析三服務並行化

**之前：** CLIP (9205)、DETR (9206)、BLIP (9207) 依序呼叫，總延遲 = t1 + t2 + t3。

**之後：** 使用 `concurrent.futures.ThreadPoolExecutor(max_workers=3)` 同時發出三個請求，總延遲 ≈ max(t1, t2, t3)，理論上縮短約 2/3。

```python
with ThreadPoolExecutor(max_workers=3) as executor:
    fut_clip    = executor.submit(...)  # 9205
    fut_detr    = executor.submit(...)  # 9206
    fut_caption = executor.submit(...)  # 9207
```

### #2 圖片上傳邏輯去重複

**之前：** `do_POST` 中 path-based 上傳（`image_name` 分支）與 base64 上傳（`image_interactive` 分支）各自包含完整的 74 行邏輯（呼叫三服務、RAG、chat_engine 開場、存對話），兩段幾乎完全相同。

**之後：** 抽出 `_handle_new_image(img_save_path, img_filename, user_text) -> dict` 方法，兩個分支各自只保留圖片存檔的差異部分（約 12 行），共用邏輯全部移至新方法。同時移除 `reset_system_status` 旗標（邏輯已內化至 `_handle_new_image`）。

## 影響的檔案

- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

## 已知問題 / 注意事項

- `ThreadPoolExecutor` 在 CPython 受 GIL 影響，但由於三個服務呼叫均為 I/O 阻塞（HTTP requests），GIL 會在等待期間釋放，並行效果正常。
- 若其中一個服務掛掉，`fut_xxx.result()` 會拋出例外，目前不會個別降級（failover）；沿用原本的行為（整個圖片上傳失敗）。
