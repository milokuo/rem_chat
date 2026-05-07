# 2026-05-07 — Jung-Min Prompt Feature Alignment

## Status

Implemented.

---

## Background

After reading `reference_code/JungMinPaper/jung_min_journal_new_edition.pdf`
and the corresponding reference code, the storage and retrieval side of
`rem_chat` was already close to the paper:

- text-turn episodes are stored in ChromaDB
- each episode has Theme / Lifetime Period / General Event / Episodic metadata
- retrieval follows semantic / theme / event matching, then union + rerank

One prompt-level mismatch remained: the reference prompt gives GPT both
retrieved memories and the current dialogue's extracted features using the
same seven attributes. In `rem_chat`, retrieved episodic memories were injected,
but the current turn's extracted features were not shown explicitly.

There was also an outdated `chat_engine.py` wrapper label saying all retrieved
memories were "OTHER past photos", which became incorrect once the retrieved
context also included text-turn episodic memories.

---

## Changes

### Current dialogue feature block

Added `format_current_dialogue_feature_block()` to `memory_retriever.py`.

It formats the current user turn with the same paper-style attributes used for
retrieved memories:

```
Content, Time, Photo, Theme, People, Activities, Locations, Objects
```

The 8082 server now prepends this block to retrieved episodic memories when
episode RAG returns candidates. No-memory turns still avoid this memory prompt
path.

### Autobiographical memory wrapper wording

Updated `chat_engine.py` retrieved-context wrapper from an album-only label to:

```
[Related autobiographical memories]
```

The rules now clarify that the block may contain past photos, past conversation
turns, and a structured current-turn feature summary.

### Tests

Added focused coverage for:

- current-dialogue feature block formatting
- retrieved prompt wrapper no longer claiming every memory is another photo

---

## Verification

- `python predictors\clip_iu\test_memory_retriever.py`
- `python predictors\clip_iu\test_chat_engine_trace_fields.py`
- `python test_server_trace.py` from `ParlAI/projects/image_chat`
- `python -m py_compile predictors\clip_iu\memory_retriever.py predictors\clip_iu\chat_engine.py`
- `python -m py_compile projects\image_chat\server_updated_zhengxuan.py` from `ParlAI`

`chat_engine.py` still emits an existing SyntaxWarning around an old escaped
string in the Traditional Chinese prompt; this change did not touch that line.

---

## Files

- `predictors/clip_iu/memory_retriever.py`
- `predictors/clip_iu/chat_engine.py`
- `predictors/clip_iu/memory_extractor.py`
- `predictors/clip_iu/test_memory_retriever.py`
- `predictors/clip_iu/test_chat_engine_trace_fields.py`
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py`

