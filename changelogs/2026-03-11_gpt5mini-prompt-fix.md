# 2026-03-11 — GPT-5-mini 升級與對話 Prompt 修正

## 背景

將預設模型從 gpt-3.5-turbo 升級至 gpt-5-mini，並修正因新模型指令遵循行為改變而暴露的三個問題：
1. Retrieved context 指令矛盾：模型被要求「記住」過去對話但又被禁止回答
2. `generate_opening` 把 CoT 分析內容直接輸出給使用者
3. `postprocess_response` 的 regex 在步驟間缺少換行時，錯誤地把 step 8 + step 9 合成一段後只切 step 8 的 colon

---

## 變更內容

### `predictors/clip_iu/config.py`
- `--model_name` 預設值從 `gpt-3.5-turbo` 改為 `gpt-5-mini`

---

### `predictors/clip_iu/chat_engine.py`

#### 1. Retrieved context block：由單一警告改為雙規則
**舊：**
```
NOTE: The past conversation excerpts below are for BACKGROUND CONTEXT ONLY.
Do NOT assume the current photograph has the same details.
```
**新：**
```
Rule 1: Do NOT project these details onto the current photograph's essentials.
Rule 2: If the User explicitly asks whether you remember something they mentioned before,
        you SHOULD reference the past conversation to confirm the specific fact,
        then gently return to exploring the current photo's own essentials.
```
解決模型在使用者問「你還記得哪間學校嗎」時無法回答的問題。

#### 2. `generate_opening`：改用 CoT + postprocess_response
- 加入 `instruct_prompt` 到 `first_turn`，讓開場白也走 CoT 格式
- 改用 `postprocess_response` 提取 step 9，避免策略分析內容外洩到使用者介面
- `system_strategies` 維持在 `system_content`，保留治療策略選擇機制

#### 3. `generate_kwargs`：gpt-5 系列略過 sampling 參數
```python
_sampling_unsupported = args.model_name.startswith('gpt-5')
if _sampling_unsupported:
    self.generate_kwargs = {}
else:
    self.generate_kwargs = {temperature, top_p, frequency_penalty, presence_penalty}
```
gpt-5-mini 不支援 `temperature=0`，傳入會觸發 400 BadRequestError。

#### 4. `postprocess_response`：改為直接搜尋 step 9
**舊做法：** regex 切割全部 9 步 → 取最後一項 → 切第一個 colon
**新做法：**
- **Primary path**：`re.search(r'(?<!\d)9\.[^:：\n]*[:：](.*)', ...)` 直接命中 step 9，不依賴步驟間有換行
- **Fallback**：原本的 numbered-step regex 切法（向後相容 gpt-3.5/4）
- **Cleanup**：整併重複的 strip 邏輯

---

## 已知問題 / 後續觀察

- gpt-5-mini 沒有 `temperature` 可調，輸出風格固定，後續可觀察是否需要用 `reasoning_effort` 參數調整
- 開場白的 `instruct_prompt` 讓模型填 step 1–8 時有些欄位不適用（如「User 說了什麼」），目前模型填 N/A 仍可正常運作，未來可考慮為開場白設計獨立的 prompt 格式

---

## 影響的檔案
- `predictors/clip_iu/config.py`
- `predictors/clip_iu/chat_engine.py`
