# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

rem_chat is a multi-modal reminiscence therapy chatbot system for patients with cognitive impairments. It uses photo-centered conversational memory exercises, integrating image understanding, dialogue management, and conversational AI to guide structured reminiscence sessions.

## Tech Stack

- **Deep Learning**: PyTorch 2.0, TensorFlow
- **Dialogue**: ParlAI (BlenderBot variants), fairseq
- **Vision**: CLIP (zero-shot classification), DETR (object detection), BLIP (captioning), dlib (face detection)
- **NLP**: OpenAI API (GPT-3.5-turbo, GPT-4), Universal Sentence Encoder
- **Web**: Flask microservices
- **Config**: omegaconf, Hydra

## Architecture

Microservice-based architecture with HTTP/JSON communication:

1. **Image Understanding Pipeline** (`predictors/clip_iu/`): CLIP classifier, DETR detector, BLIP captioner
2. **GPT Orchestration** (`predictors/clip_iu/chat_engine.py`): main dialogue generation and prompt logic
3. **Web Gateway** (`ParlAI/projects/image_chat/server_updated_zhengxuan.py`): active HTTP entrypoint (ParlAI-free) that orchestrates image analysis + RAG + chat_engine
4. **Auxiliary Similarity Service** (`VQA_enhancement/`): USE-based history/similarity tracking (`9110`), still called by web gateway but not the primary reply generator
5. **Legacy Paths** (`predictors/iu/`, `server_updated_blenderbot*.py`, `server_updated_xiaobei.py`): older TensorFlow/BlenderBot flows kept for compatibility

### Key Files

- `predictors/clip_iu/chat_engine.py` - Main chat orchestration, dialogue history, support strategies
- `predictors/clip_iu/config.py` - Configuration with language selection, GPT parameters
- `predictors/clip_iu/photo_db.py` - ChromaDB wrapper for album RAG (added 2026-03-10)
- `predictors/clip_iu/album_indexer.py` - Offline batch script to pre-index a patient's album (added 2026-03-10)
- `VQA_enhancement/GptGenerator.py` - Legacy OpenAI wrapper (not used by active `chat_engine.py` flow)
- `VQA_enhancement/QuestionItem.py` - Question corpus loading with pre-computed embeddings
- `ParlAI/projects/image_chat/server_updated_zhengxuan.py` - Active web server (ParlAI-free, port 8082); delegates all AI responses to chat_engine (port 8087)

## Development Setup

### Installation

```bash
# ParlAI
cd ParlAI
python setup.py develop

# fairseq
cd fairseq
pip install --editable ./
```

### Running Services

Service ports:
- 9205: CLIP predictor
- 9206: DETR detector
- 9207: Image captioning (BLIP)
- 9209: IU redirect handler
- 9110: Question similarity / dialogue-history similarity service (auxiliary)
- 8082: Active web server (`server_updated_zhengxuan.py`, ParlAI-free)
- 8087: `chat_engine` Flask service (main GPT orchestrator endpoint)

Start CLIP services (each in its own tab):
```bash
source predictors/clip_iu/clip_iu/bin/activate
cd predictors/clip_iu
python clip_predictor.py 9205
python detr_detector.py 9206
python image_caption.py 9207
```

Start chat engine (Flask service on port 8087):
```bash
source predictors/clip_iu/clip_iu/bin/activate
cd predictors/clip_iu
python chat_engine.py
```

Start IU redirect handler:
```bash
source predictors/rem_iu/bin/activate
cd predictors/iu
python run_iu_server_handler.py 9209
```

Start web server (port 8082, ParlAI-free):
```bash
source rem_env/bin/activate
cd ParlAI
python projects/image_chat/server_updated_zhengxuan.py
```

Start question similarity service:
```bash
cd VQA_enhancement
source sim_py3/bin/activate
python web_serv_sim.py 9110
```

### ParlAI Commands

```bash
# Display dataset samples
parlai display_data -t squad

# Evaluate model
parlai eval_model -m ir_baseline -t personachat -dt valid

# Train model
parlai train_model -t personachat -m transformer/ranker \
  -mf /tmp/model --n-layers 1 --embedding-size 300 \
  --ffn-size 600 --n-heads 4 --num-epochs 2 -bs 64
```

## Virtual Environments

- `rem_env/` - Main environment (Python 3.10)
- `predictors/clip_iu/clip_iu/` - CLIP services
- `predictors/rem_iu/` - Legacy IU
- `VQA_enhancement/sim_py3/` - Question similarity

## Data Files

- `VQA_enhancement/v3/questions.json` - Question corpus
- `VQA_enhancement/v3/questions_embedding.npy` - Pre-computed embeddings
- `VQA_enhancement/v3/sim_mat.npy` - Similarity matrix
- `predictors/iu/labels/` - Label definitions (events, places, relationships)

## Git (run in WSL, project is at /mnt/e/rem_chat equivalent via Windows path)

ParlAI and fairseq are git submodules. Commit order matters:

```bash
# 1. Commit changes inside a submodule first (e.g. ParlAI)
cd "E:/rem_chat/ParlAI"
git add <file>
git commit -m "..."

# 2. Then commit root repo (which records the updated submodule pointer)
cd "E:/rem_chat"
git add <submodule_dir> <other_files>
git commit -m "..."
git push origin main
```

## Changelogs

每次重大功能更新都會在 `changelogs/` 目錄下留下詳細記錄。

- **索引（先看這裡）：** `changelogs/INDEX.md` — 所有更新的一行摘要，附影響檔案清單
- **詳細日誌：** `changelogs/<date>_<topic>.md` — 每次更新的完整說明、架構決策、已知問題

## Notes

- Bilingual support: English and Traditional Chinese
- GPU required for inference
- Services communicate via HTTP POST with JSON payloads
- Pre-computed embeddings used for fast similarity matching
- OpenAI API key is hardcoded in `config.py` (intentional, excluded via .gitignore)
- Uses OpenAI SDK v1+ (`openai.OpenAI()` client style, not legacy `openai.ChatCompletion`)
- ChromaDB vector DB at `predictors/clip_iu/photo_index/` (RAG, added 2026-03-10); requires `pip install chromadb` in clip_env
- Uploaded photos stored per-patient at `ParlAI/projects/image_chat/uploads/<patient_id>/`
- Per-photo conversation logs are stored at `ParlAI/projects/image_chat/uploads/<patient_id>/conversations/<photo_basename>.json` (photo association is by file path/name, not a dedicated `photo_id` field inside each turn record)
- Active user tracked via `PATIENT_ID` global in server; persisted across restarts in `uploads/users.json`
