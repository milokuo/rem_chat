# 2026-04-08 — chat_engine 重構：消除重複程式碼 + 強化 CoT 解析

## 動機

針對 `chat_engine.py` 中兩個已知問題進行精準重構，不改動任何業務邏輯。

---

## 變更 1：抽出 `_build_retrieved_block()`

### 問題

`preprocess_conversation`（原 lines 135–149）與 `generate_opening`（原 lines 213–227）各自獨立維護一份完全相同的 retrieved block 建構程式碼，包含 Rule 1、Rule 2 說明文字。

### 修法

新增 `_build_retrieved_block(self)` 方法，兩處呼叫端各簡化為一行：

```python
self.system_prompt = self.system_task + self.system_strategies + self.observation_prompt + self._build_retrieved_block()
```

### 影響

- 無行為改變，純重構
- 未來修改 Rule 1/2 文字只需改一處

---

## 變更 2：instruct_prompt 加入 JSON_OUTPUT tag + 更新解析優先序

### 問題

`postprocess_response` 依賴正則解析 GPT 輸出的 step 9。gpt-5-mini 常合併步驟或改變格式，現有 primary + fallback 雙層解析在格式再次改變時可能靜默返回錯誤內容。

### 修法

在 zh 與 en 的 `instruct_prompt` step 9 末尾加入一行：

```
JSON_OUTPUT: {"reply": "<回覆內容>", "strategy": "<所選策略>"}
```

`postprocess_response` 解析優先序改為三層：

1. **JSON_OUTPUT parse**（新，最優先）：搜尋 `JSON_OUTPUT:` tag，直接 `json.loads` 取 `reply` 欄位
2. **step 9 regex**（原 primary）：`r'(?<!\d)9\.[^:：\n]*[:：](.*)'`
3. **numbered-step split**（原 fallback）：分割全文後取最後一段

JSON parse 失敗（JSONDecodeError / 欄位缺失）自動降級到下一層，行為與舊版相同。

### 優點

- CoT 推理流程（steps 1–8）完整保留，不影響回覆品質
- gpt-5-mini 若遵循格式則直接用 JSON，解析零歧義
- 舊模型（gpt-3.5/4）不輸出 JSON_OUTPUT 時自動 fallback，向下相容
- `cot_response` logging 仍保留（route handler line 311 不受影響）

---

---

## 補丁：修補 P1 / P2（Codex 驗證後）

### P1 — JSON parse 失敗時，`JSON_OUTPUT` 雜訊外洩給使用者

step9 regex 使用 `re.DOTALL` 抓到結尾，若 JSON parse 失敗會把 `JSON_OUTPUT: {...}` 整段帶入 `assistant_response`。

**修法**：降級前先用 `re.sub(r'\nJSON_OUTPUT:[^\n]*', '', top_response)` 產生 `response_for_regex`，step9 與 numbered-split fallback 均改用此清理後的字串。

### P2 — JSON regex `[^}]+` 遇到 reply 含 `}` 提早截斷

**修法**：改為 `[^\n]+`，JSON_OUTPUT 固定在同一行，用換行邊界更可靠。

---

## 影響的檔案

- `predictors/clip_iu/chat_engine.py`
