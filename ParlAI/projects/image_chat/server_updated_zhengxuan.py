# -*- coding:utf-8 -*-
"""
server_updated_zhengxuan.py

ParlAI-free version of server_updated_xiaobei.py.
Removes all BlenderBot / ParlAI dependencies.
The conversation response is delegated entirely to chat_engine.py (port 8087, GPT).
Image analysis still calls CLIP (9205), DETR (9206), and BLIP caption (9207).
Question-similarity tracking still calls sim service (9110).

Start:
    python projects/image_chat/server_updated_zhengxuan.py
(No -mf or ParlAI model args required.)
"""

from typing import Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import requests
import sys
import threading
import time
import json
from server_multipart import _parse_multipart
from server_timing import _TIMING_LOG, _timed_worker, _log_timing
from server_users import _load_users, _persist_user
from server_conv_store import _get_conv_file, _save_conv_turn, _load_conv_turns
from server_memory import _finalize_conversation_memory
import PIL.Image as Image
from base64 import b64decode
import io
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

# PhotoDB lives in predictors/clip_iu/
_CLIP_IU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "predictors", "clip_iu")
if _CLIP_IU_DIR not in sys.path:
    sys.path.insert(0, _CLIP_IU_DIR)
from photo_db import PhotoDB
from memory_extractor import (
    classify_theme_and_entities,
    classify_utterance_memory,
    embed_memory_text,
)
from memory_hierarchy import build_episode_hierarchy_metadata
from memory_retriever import (
    retrieve as _memory_retrieve,
    retrieve_episodes as _memory_retrieve_episodes,
    format_retrieved_block,
    format_episode_block,
    format_current_dialogue_feature_block,
)

import openai as _openai

from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
INPUT_LANGUAGE = 'en'                              # 'en' or 'zh'
PATIENT_ID = "default"                             # patient identifier for DB tagging
# 圖片儲存路徑：預設放在腳本同層的 uploads/ 資料夾，啟動時自動建立
SERVER_IMAGE_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
HOST_NAME = "0.0.0.0"
PORT = 8082
SHARED: Dict[str, Any] = {}

# Limit concurrent background (enrich/finalize) threads to avoid pile-up on rapid photo switches
_bg_semaphore = threading.Semaphore(4)

# Protects writes to PATIENT_ID and bulk-updates to SHARED from concurrent requests.
_state_lock = threading.Lock()

trans_zh_en = GoogleTranslator(source='zh-TW', target='en')
trans_en_zh = GoogleTranslator(source='en', target='zh-TW')

# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "predictors", "clip_iu", "photo_index")
_photo_db = PhotoDB(persist_dir=_DB_DIR)

# ---------------------------------------------------------------------------
# OpenAI client (shares key from env; memory_extractor calls reuse this)
# ---------------------------------------------------------------------------
_OPENAI_KEY_FILE = os.path.join(_CLIP_IU_DIR, "config.py")
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_config", _OPENAI_KEY_FILE)
    _cfg_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg_mod)
    _cfg_args = _cfg_mod.parse_args()
    _openai_client = _openai.OpenAI(api_key=_cfg_args.openai_key)
    _MEMORY_MODEL = _cfg_args.model_name
except Exception as _e:
    print(f"[Memory] Could not load config for OpenAI client: {_e}")
    _openai_client = None
    _MEMORY_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Debug-dashboard trace helper
# ---------------------------------------------------------------------------
def _post_trace(payload: dict) -> None:
    """Fire-and-forget: POST trace payload to debug dashboard (port 8090).

    300 ms timeout; silently drops the payload if the dashboard is not running.
    Uses a daemon thread so it never blocks process exit.
    """
    def _send():
        try:
            requests.post("http://localhost:8090/api/trace", json=payload, timeout=0.3)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


def _combine_retrieved_context(photo_context: str, episode_context: str) -> str:
    """Combine photo-level and turn-level memory blocks without accumulating stale text."""
    return "\n\n".join(
        part.strip() for part in (photo_context, episode_context) if part and part.strip()
    )


def _recent_context_text(photo_id: str, patient_id: str, uploads_dir: str, max_turns: int = 3) -> str:
    turns = _load_conv_turns(photo_id, patient_id, uploads_dir, max_turns=max_turns) if photo_id else []
    lines = []
    for turn in turns:
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Assistant: {turn.get('assistant', '')}")
    return "\n".join(lines)


def _entity_updates(features: dict) -> dict:
    return {
        "entities_people": json.dumps(features.get("people", []), ensure_ascii=False),
        "entities_activities": json.dumps(features.get("activities", []), ensure_ascii=False),
        "entities_locations": json.dumps(features.get("locations", []), ensure_ascii=False),
        "entities_objects": json.dumps(features.get("objects", []), ensure_ascii=False),
    }


def _episode_query_entities(features: dict) -> dict:
    return {
        "people": features.get("people", []),
        "activities": features.get("activities", []),
        "locations": features.get("locations", []),
        "objects": features.get("objects", []),
    }


def _strip_candidate_embedding(candidates: list[dict]) -> list[dict]:
    stripped = []
    for candidate in candidates:
        item = {k: v for k, v in candidate.items() if k != "embedding"}
        if "_visual_score" in candidate:
            item.setdefault("visual", candidate.get("_visual_score"))
        if "_semantic_score" in candidate:
            item.setdefault("semantic", candidate.get("_semantic_score"))
        if "_entity_score" in candidate:
            item.setdefault("entity", candidate.get("_entity_score"))
        if "_recency_score" in candidate:
            item.setdefault("recency", candidate.get("_recency_score"))
        if "_rank_score" in candidate:
            item.setdefault("rank_score", candidate.get("_rank_score"))
        if "_theme_match" in candidate:
            item.setdefault("theme_match", candidate.get("_theme_match"))
        stripped.append(item)
    return stripped


def _prepare_text_memory_context(
    user_utterance: str,
    current_photo_id: str,
    caption: str,
    objects: str,
    patient_id: str,
    uploads_dir: str,
) -> dict:
    """Extract features from the current text turn and retrieve episodic memories."""
    result = {
        "features": {"theme": "", "people": [], "activities": [], "locations": [], "objects": []},
        "embedding": [],
        "context": "",
        "candidates": [],
        "timing": {},
    }
    if not user_utterance or not _openai_client:
        return result

    recent_context = _recent_context_text(current_photo_id, patient_id, uploads_dir)
    t0 = time.perf_counter()
    features = classify_utterance_memory(
        user_utterance, recent_context, _openai_client, model=_MEMORY_MODEL
    )
    result["timing"]["memory_extract_ms"] = round((time.perf_counter() - t0) * 1000)
    result["features"] = features

    query_text = (
        f"Current photo caption: {caption}\n"
        f"Detected objects: {objects}\n"
        f"Recent conversation:\n{recent_context}\n"
        f"Latest user utterance: {user_utterance}"
    )
    t0 = time.perf_counter()
    embedding = embed_memory_text(query_text, _openai_client)
    result["timing"]["memory_embed_ms"] = round((time.perf_counter() - t0) * 1000)
    result["embedding"] = embedding

    if not embedding:
        return result

    t0 = time.perf_counter()
    candidates = _memory_retrieve_episodes(
        photo_db=_photo_db,
        query_embedding=embedding,
        query_theme=features.get("theme", ""),
        query_entities=_episode_query_entities(features),
        patient_id=patient_id,
        n_results=3,
    )
    result["timing"]["episode_rag_ms"] = round((time.perf_counter() - t0) * 1000)
    result["candidates"] = candidates
    episode_block = format_episode_block(candidates)
    if episode_block:
        current_feature_block = format_current_dialogue_feature_block(
            user_utterance=user_utterance,
            features=features,
            timestamp=datetime.datetime.now().isoformat(),
            photo_id=current_photo_id,
        )
        result["context"] = _combine_retrieved_context(current_feature_block, episode_block)
    return result


def _save_text_episode_memory(
    user_utterance: str,
    assistant_reply: str,
    current_photo_id: str,
    patient_id: str,
    features: dict,
    text_embedding: list,
) -> None:
    """Store one user/assistant text turn as an episodic memory node."""
    if not user_utterance or not assistant_reply or not text_embedding:
        return
    ts = datetime.datetime.now().isoformat()
    photo_id = current_photo_id or ""
    episode_id = f"{patient_id}/episode/{time.time_ns()}"
    hierarchy_meta = build_episode_hierarchy_metadata(
        episode_id=episode_id,
        patient_id=patient_id,
        timestamp=ts,
        theme=features.get("theme", ""),
        entities=_episode_query_entities(features),
    )
    metadata = {
        "patient_id": patient_id,
        "photo_id": photo_id,
        "timestamp": ts,
        "theme": features.get("theme", ""),
        "user_utterance": user_utterance,
        "assistant_reply": assistant_reply,
        "content": f"User: {user_utterance}\nAssistant: {assistant_reply}",
    }
    metadata.update(hierarchy_meta)
    metadata.update(_entity_updates(features))
    _photo_db.add_episode(episode_id, text_embedding, metadata)
    print(f"[EpisodeMemory] Saved {episode_id} theme={metadata['theme']}")


# ---------------------------------------------------------------------------
# Simple web UI (kept identical to xiaobei version)
# ---------------------------------------------------------------------------
STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
WEB_HTML = """
<html>
    <link rel="stylesheet" href={} />
    <script defer src={}></script>
    <head><meta charset="UTF-8"><title> Interactive Run </title>
    <style>
        .typing-dots {{ display: inline-flex; gap: 4px; align-items: center; padding: 2px 0; }}
        .typing-dots span {{ width: 8px; height: 8px; border-radius: 50%; background: #888; display: inline-block; animation: typing-bounce 1.2s infinite ease-in-out; }}
        .typing-dots span:nth-child(1) {{ animation-delay: 0s; }}
        .typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes typing-bounce {{ 0%, 80%, 100% {{ transform: scale(0.6); opacity: 0.4; }} 40% {{ transform: scale(1.0); opacity: 1; }} }}
    </style>
    </head>
    <body>
        <div class="columns">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-large has-background-light has-text-grey-dark">
                <div class="hero-head" style="padding:0.75rem 1.5rem;border-bottom:1px solid #ddd;background:#efefef;">
                  <div class="field is-grouped is-align-items-center mb-0">
                    <label class="label mb-0 mr-3" style="white-space:nowrap;color:#333;">使用者：</label>
                    <p class="control">
                      <div class="select">
                        <select id="userSelect"></select>
                      </div>
                    </p>
                    <p class="control">
                      <input class="input" type="text" id="newUserInput" placeholder="新增使用者名稱" style="max-width:180px;">
                    </p>
                    <p class="control">
                      <button id="addUserBtn" type="button" class="button has-background-grey-dark has-text-white-ter">新增</button>
                    </p>
                    <p class="control">
                      <span id="currentUserLabel" style="font-size:0.875em;color:#555;margin-left:0.5rem;"></span>
                    </p>
                  </div>
                </div>
                <div id="parent" class="hero-body">
                    <article class="media" id="photo-info">
                      <figure class="media-left">
                        <span class="icon is-large">
                          <i class="fas fa-robot fas fa-2x"></i>
                        </span>
                      </figure>
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <img id="preview" src="Examples.png"/ style="max-height:300px">
                          </p>
                        </div>
                      </div>
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>Model</strong>
                            <br>
                            Enter a message, and the model will respond interactively.
                          </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column">
                  <form id = "interact">
                      <div class="field is-grouped">
                      <p class="control">
                        Type a message:
                        <input class="input" form="interact" type="text" id="userIn" placeholder="Type in a message" size="10">
                      </p>
                        <p class="control is-expanded">
                          Upload an image:
                          <input class="input" type="file" id="userInImage" accept="image/*">
                        </p>
                        <p class="control">
                          <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            Submit
                          </button>
                        </p>
                      </div>
                  </form>
                  <p class="control" style="padding:0.25rem 0 0.4rem;">
                    <label class="checkbox" style="font-size:0.85em;color:#444;">
                      <input type="checkbox" id="formatAlignToggle"> 精確模式（較慢）
                    </label>
                  </p>
                  <p class="control">
                    <button id="newImage" class="button has-text-white-ter has-background-grey-dark">
                      New Image
                    </button>
                  </p>
                </div>
              </section>
            </div>
        </div>

        <script>
            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"

                var figure = document.createElement("figure");
                figure.className = "media-left";

                var span = document.createElement("span");
                span.className = "icon is-large";

                var icon = document.createElement("i");
                icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" : "");

                var media = document.createElement("div");
                media.className = "media-content";

                var content = document.createElement("div");
                content.className = "content";

                var para = document.createElement("p");
                var paraText = document.createTextNode(text);

                var strong = document.createElement("strong");
                strong.innerHTML = agent;
                var br = document.createElement("br");

                para.appendChild(strong);
                para.appendChild(br);
                para.appendChild(paraText);
                content.appendChild(para);
                media.appendChild(content);

                span.appendChild(icon);
                figure.appendChild(span);

                media.id = "model-response1";
                figure.id = "model-response2";

                article.appendChild(figure);
                article.appendChild(media);

                return article;
            }}
            var _submitting = false;

            function createThinkingRow() {{
                var article = document.createElement("article");
                article.className = "media";
                var figure = document.createElement("figure");
                figure.className = "media-left";
                var span = document.createElement("span");
                span.className = "icon is-large";
                var icon = document.createElement("i");
                icon.className = "fas fa-robot fas fa-2x";
                span.appendChild(icon);
                figure.appendChild(span);
                var media = document.createElement("div");
                media.className = "media-content";
                var content = document.createElement("div");
                content.className = "content";
                var para = document.createElement("p");
                var strong = document.createElement("strong");
                strong.innerHTML = "Model";
                var br = document.createElement("br");
                var dots = document.createElement("span");
                dots.className = "typing-dots";
                dots.innerHTML = "<span></span><span></span><span></span>";
                para.appendChild(strong);
                para.appendChild(br);
                para.appendChild(dots);
                content.appendChild(para);
                media.appendChild(content);
                article.appendChild(figure);
                article.appendChild(media);
                return article;
            }}

            function createStreamingBotRow() {{
                var article = document.createElement("article");
                article.className = "media";
                var figure = document.createElement("figure");
                figure.className = "media-left";
                var span = document.createElement("span");
                span.className = "icon is-large";
                var icon = document.createElement("i");
                icon.className = "fas fa-robot fas fa-2x";
                span.appendChild(icon);
                figure.appendChild(span);
                var media = document.createElement("div");
                media.className = "media-content";
                var content = document.createElement("div");
                content.className = "content";
                var para = document.createElement("p");
                var strong = document.createElement("strong");
                strong.innerHTML = "Model";
                var br = document.createElement("br");
                var textSpan = document.createElement("span");
                para.appendChild(strong);
                para.appendChild(br);
                para.appendChild(textSpan);
                content.appendChild(para);
                media.appendChild(content);
                media.id = "model-response1";
                figure.id = "model-response2";
                article.appendChild(figure);
                article.appendChild(media);
                return {{article: article, textSpan: textSpan}};
            }}

            function clearConversationRows() {{
                var parDiv = document.getElementById("parent");
                Array.prototype.slice.call(parDiv.children).forEach(function(child) {{
                    if (child.id !== "photo-info") {{
                        child.remove();
                    }}
                }});
            }}

            function fetchResult(image_data, text) {{
                var parDiv = document.getElementById("parent");
                if (image_data !== "") {{
                    clearConversationRows();
                }}
                // Show user message immediately
                if (text !== "") {{
                    parDiv.append(createChatRow("You", text));
                    window.scrollTo(0, document.body.scrollHeight);
                }}

                // Image uploads always use accurate mode (no streaming).
                var useStream = (image_data === "") && !document.getElementById("formatAlignToggle").checked;

                var formData = new FormData();
                formData.append('image', image_data);
                formData.append('text', text);

                if (!useStream) {{
                    // --- Accurate mode: thinking indicator, wait for full JSON response ---
                    var thinkingRow = createThinkingRow();
                    parDiv.append(thinkingRow);
                    window.scrollTo(0, document.body.scrollHeight);
                    fetch('/interact', {{
                        method: 'POST',
                        body: formData
                    }}).then(function(response) {{ return response.json(); }}).then(function(data) {{
                        thinkingRow.remove();
                        parDiv.append(createChatRow("Model", data.text));
                        document.getElementById("userInImage").value = "";
                        window.scrollTo(0, document.body.scrollHeight);
                    }}).catch(function(err) {{
                        thinkingRow.remove();
                        console.error("fetch error:", err);
                    }}).finally(function() {{
                        _submitting = false;
                        document.getElementById("respond").disabled = false;
                    }});
                }} else {{
                    // --- Streaming mode: show thinking indicator until first SSE payload, then append tokens as they arrive ---
                    var thinkingRow = createThinkingRow();
                    var streamObj = null;
                    parDiv.append(thinkingRow);
                    window.scrollTo(0, document.body.scrollHeight);
                    function ensureStreamingRow() {{
                        if (!streamObj) {{
                            streamObj = createStreamingBotRow();
                            if (thinkingRow.parentNode) {{
                                thinkingRow.replaceWith(streamObj.article);
                            }} else {{
                                parDiv.append(streamObj.article);
                            }}
                        }}
                        return streamObj;
                    }}
                    fetch('/interact_stream', {{
                        method: 'POST',
                        body: formData
                    }}).then(function(response) {{
                        if (!response.ok) {{
                            ensureStreamingRow().textSpan.textContent = "(伺服器錯誤 " + response.status + "，請重試)";
                            return;
                        }}
                        var reader = response.body.getReader();
                        var decoder = new TextDecoder();
                        var buffer = '';
                        function processLines(chunk) {{
                            buffer += chunk;
                            var lines = buffer.split('\\n');
                            buffer = lines.pop();
                            lines.forEach(function(line) {{
                                if (line.startsWith('data: ')) {{
                                    try {{
                                        var payload = JSON.parse(line.slice(6));
                                        if (payload.token) {{
                                            ensureStreamingRow().textSpan.textContent += payload.token;
                                            window.scrollTo(0, document.body.scrollHeight);
                                        }}
                                        if (payload.error) {{
                                            ensureStreamingRow().textSpan.textContent += "(錯誤：" + payload.error + ")";
                                            window.scrollTo(0, document.body.scrollHeight);
                                        }}
                                        if (payload.done && !streamObj) {{
                                            ensureStreamingRow().textSpan.textContent = payload.full || "(沒有收到回覆，請重試)";
                                            window.scrollTo(0, document.body.scrollHeight);
                                        }}
                                    }} catch(e) {{}}
                                }}
                            }});
                        }}
                        function readChunk() {{
                            return reader.read().then(function(result) {{
                                if (result.done) {{
                                    // Final flush: decode any remaining bytes
                                    var tail = decoder.decode();
                                    if (tail) processLines(tail);
                                    return;
                                }}
                                processLines(decoder.decode(result.value, {{stream: true}}));
                                return readChunk();
                            }});
                        }}
                        return readChunk();
                    }}).catch(function(err) {{
                        ensureStreamingRow().textSpan.textContent = "(連線錯誤，請重試)";
                        console.error("stream error:", err);
                    }}).finally(function() {{
                        document.getElementById("userInImage").value = "";
                        _submitting = false;
                        document.getElementById("respond").disabled = false;
                    }});
                }}
            }}
            document.getElementById("interact").addEventListener("submit", function(event) {{
                event.preventDefault();
                if (_submitting) return;
                var text = document.getElementById("userIn").value;
                var img_input = document.getElementById("userInImage");
                if (!text.trim() && !(img_input.files && img_input.files[0])) return;
                _submitting = true;
                document.getElementById("respond").disabled = true;
                document.getElementById("userIn").value = "";

                var preview = document.getElementById("preview");
                if (img_input.files && img_input.files[0]) {{
                    var reader = new FileReader();
                    reader.onload = function(e) {{
                        preview.setAttribute('src', e.target.result);
                        fetchResult(e.target.result, text);
                    }};
                    reader.onerror = reader.onabort = function() {{
                        _submitting = false;
                        document.getElementById("respond").disabled = false;
                        console.error("FileReader failed");
                    }};
                    reader.readAsDataURL(img_input.files[0]);
                }} else {{
                    fetchResult('', text);
                }}
            }});
            document.getElementById("newImage").addEventListener("click", function(event){{
                event.preventDefault()
                clearConversationRows();
                var preview = document.getElementById("preview");
                preview.setAttribute('src', '');
            }});

            // ---- 使用者選擇 ----
            function loadUsers() {{
                fetch('/users', {{cache: 'no-store'}}).then(function(r) {{ return r.json(); }}).then(function(data) {{
                    var select = document.getElementById('userSelect');
                    select.innerHTML = '';
                    data.users.forEach(function(u) {{
                        var opt = document.createElement('option');
                        opt.value = u;
                        opt.textContent = u;
                        if (u === data.current) opt.selected = true;
                        select.appendChild(opt);
                    }});
                    document.getElementById('currentUserLabel').textContent = '目前使用者：' + data.current;
                }}).catch(function() {{
                    document.getElementById('userSelect').innerHTML = '<option value="default">default</option>';
                }});
            }}

            document.getElementById('userSelect').addEventListener('change', function() {{
                var userId = this.value;
                if (!userId) return;
                fetch('/set_user', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{user_id: userId}})
                }}).then(function(r) {{ return r.json(); }}).then(function(data) {{
                    document.getElementById('currentUserLabel').textContent = '目前使用者：' + data.current;
                }});
            }});

            document.getElementById('addUserBtn').addEventListener('click', function() {{
                var newUser = document.getElementById('newUserInput').value.trim();
                if (!newUser) return;
                fetch('/set_user', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{user_id: newUser}})
                }}).then(function(r) {{
                    if (!r.ok) throw new Error('set_user failed: ' + r.status);
                    return r.json();
                }}).then(function(data) {{
                    document.getElementById('newUserInput').value = '';
                    loadUsers();
                }}).catch(function(err) {{
                    console.error('新增使用者失敗：', err);
                    alert('新增使用者失敗，請檢查伺服器狀態。');
                }});
            }});

            loadUsers();
        </script>

    </body>
</html>
"""  # noqa: E501


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class MyHandler(BaseHTTPRequestHandler):
    """
    Lightweight HTTP handler without ParlAI.

    Flow for a text message:
      1. Receive user input via POST /interact
      2. Notify sim service (9110)
      3. Call chat_engine (8087, GPT) for a response
      4. Translate zh-TW ↔ en where needed
      5. Return JSON {"text": <reply>}
    """

    global INPUT_LANGUAGE
    input_language = INPUT_LANGUAGE

    # ------------------------------------------------------------------
    # New-image handler (shared by path-based and base64 upload paths)
    # ------------------------------------------------------------------
    def _handle_new_image(self, img_save_path: str, img_filename: str, user_text: str) -> dict:
        """
        Called after the image has been saved to disk.
        Calls CLIP / DETR / BLIP in parallel, runs RAG, asks chat_engine
        for an opening turn, persists the turn, and returns {"text": reply}.
        """
        timing = {"type": "new_image"}
        t_total = time.perf_counter()

        # Notify chat_engine to close previous conversation.
        if SHARED.get("dialog_started"):
            t0 = time.perf_counter()
            self.send_post_message_to_chat({'user_message': 'conversation over'})
            timing["conv_over_ms"] = round((time.perf_counter() - t0) * 1000)
            # Finalize memory for the photo that just ended (background thread).
            prev_photo_id = SHARED.get("current_photo_id", "")
            if prev_photo_id:
                _finalize_conversation_memory(
                    prev_photo_id, PATIENT_ID, SERVER_IMAGE_LOCATION,
                    photo_db=_photo_db, openai_client=_openai_client,
                    model=_MEMORY_MODEL, bg_semaphore=_bg_semaphore,
                )

        # Reset sim history.
        t0 = time.perf_counter()
        self.send_post_message_to_sim({'reset_history': True})
        timing["sim_reset_ms"] = round((time.perf_counter() - t0) * 1000)

        # Call CLIP / DETR / BLIP in parallel — they are fully independent.
        _img_payload = {'img_id': img_filename, 'full_path': img_save_path}
        t_parallel = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_clip    = executor.submit(_timed_worker, self.send_post_message_iu, {**_img_payload, 'cate': 'clip'   }, 'http://127.0.0.1:9205/')
            fut_detr    = executor.submit(_timed_worker, self.send_post_message_iu, {**_img_payload, 'cate': 'detr'   }, 'http://127.0.0.1:9206/')
            fut_caption = executor.submit(_timed_worker, self.send_post_message_iu, {**_img_payload, 'cate': 'caption'}, 'http://127.0.0.1:9207/')
        timing["parallel_iu_ms"] = round((time.perf_counter() - t_parallel) * 1000)

        clip_result,    timing["clip_ms"]  = fut_clip.result()
        detr_result,    timing["detr_ms"]  = fut_detr.result()
        caption_result, timing["blip_ms"]  = fut_caption.result()

        with _state_lock:
            SHARED['metadata']        = clip_result
            SHARED['image_embedding'] = SHARED['metadata'].get('embedding', [])
            SHARED['obj_str']         = detr_result.get('objects', '')
            SHARED['caption_str']     = caption_result.get('caption', '')

        # RAG: save current photo to DB and retrieve related photos.
        embedding = SHARED.get('image_embedding', [])
        photo_id  = f"{PATIENT_ID}/{img_filename}"
        with _state_lock:
            SHARED['current_photo_id'] = photo_id

        candidates = []  # populated below when embedding is available
        t0 = time.perf_counter()
        if embedding:
            try:
                db_metadata = {
                    'caption':      SHARED.get('caption_str', ''),
                    'objects':      SHARED.get('obj_str', ''),
                    'event':        (SHARED.get('metadata') or {}).get('event', {}).get('label', 'unknown'),
                    'place':        (SHARED.get('metadata') or {}).get('place', {}).get('label', 'unknown'),
                    'relationship': (SHARED.get('metadata') or {}).get('relationship', {}).get('label', 'unknown'),
                    'patient_id':   PATIENT_ID,
                    'filename':     img_filename,
                }

                # Fetch existing record BEFORE upsert:
                # (a) preserves enriched fields (theme/entities/conv_summary) on re-upload — P2 fix
                # (b) enables theme+entity retrieval signals on revisit — P1 fix
                _existing_meta = {}
                try:
                    for _r in _photo_db.query_by_patient(PATIENT_ID, n_results=500):
                        if _r["id"] == photo_id:
                            _existing_meta = _r
                            break
                except Exception:
                    pass
                for _f in ("theme", "entities_people", "entities_activities",
                           "entities_locations", "entities_objects",
                           "conv_summary", "last_chatted", "upload_timestamp"):
                    if _existing_meta.get(_f):
                        db_metadata[_f] = _existing_meta[_f]

                _photo_db.add_photo(photo_id, embedding, db_metadata)
                print(f"[PhotoDB] Saved: {photo_id} (total: {_photo_db.count()})")

                # Three-way retrieval: use existing enriched data when available (e.g. revisit)
                _query_theme = _existing_meta.get("theme", "")
                _query_entities = {
                    "people":     json.loads(_existing_meta.get("entities_people",     "[]") or "[]"),
                    "activities": json.loads(_existing_meta.get("entities_activities", "[]") or "[]"),
                    "locations":  json.loads(_existing_meta.get("entities_locations",  "[]") or "[]"),
                    "objects":    json.loads(_existing_meta.get("entities_objects",    "[]") or "[]"),
                }
                candidates = _memory_retrieve(
                    photo_db=_photo_db,
                    query_embedding=embedding,
                    query_theme=_query_theme,
                    query_entities=_query_entities,
                    patient_id=PATIENT_ID,
                    current_photo_id=photo_id,
                    n_results=3,
                )
                _conv_loader = lambda pid, phid, n=5: _load_conv_turns(pid, phid, SERVER_IMAGE_LOCATION, max_turns=n)
                retrieved_text = format_retrieved_block(candidates, _conv_loader, PATIENT_ID)
                with _state_lock:
                    SHARED['photo_retrieved_context'] = retrieved_text
                    SHARED['turn_retrieved_context'] = ''
                    SHARED['retrieved_context'] = retrieved_text
                print(f"[Retrieved context]:\n{retrieved_text}")
            except Exception as e:
                print(f"[RAG] Error: {e}")
                with _state_lock:
                    SHARED['photo_retrieved_context'] = ''
                    SHARED['turn_retrieved_context'] = ''
                    SHARED['retrieved_context'] = ''
        else:
            with _state_lock:
                SHARED['photo_retrieved_context'] = ''
                SHARED['turn_retrieved_context'] = ''
                SHARED['retrieved_context'] = ''
        timing["rag_ms"] = round((time.perf_counter() - t0) * 1000)

        # Enrich new photo with theme + entities (background thread, non-blocking).
        if embedding and _openai_client:
            _enrich_caption = SHARED.get('caption_str', '')
            _enrich_objects = SHARED.get('obj_str', '')
            _enrich_photo_id = photo_id
            def _enrich_photo():
                try:
                    info = classify_theme_and_entities(
                        _enrich_caption, _enrich_objects, _openai_client, model=_MEMORY_MODEL
                    )
                    updates = {
                        "theme":                info["theme"],
                        "entities_people":      json.dumps(info["people"],     ensure_ascii=False),
                        "entities_activities":  json.dumps(info["activities"], ensure_ascii=False),
                        "entities_locations":   json.dumps(info["locations"],  ensure_ascii=False),
                        "entities_objects":     json.dumps(info["objects"],    ensure_ascii=False),
                        "upload_timestamp":     datetime.datetime.now().isoformat(),
                    }
                    _photo_db.update_metadata(_enrich_photo_id, updates)
                    print(f"[Memory] Enriched {_enrich_photo_id}: theme={info['theme']}")
                except Exception as exc:
                    print(f"[Memory] enrich failed for {_enrich_photo_id}: {exc}")
            def _enrich_photo_sem():
                with _bg_semaphore:
                    _enrich_photo()
            threading.Thread(target=_enrich_photo_sem, daemon=True).start()

        # Ask chat_engine (GPT) to generate an opening turn.
        t0 = time.perf_counter()
        res_chat = self.send_post_message_to_chat({
            'lang':              self.input_language,
            'caption_str':       SHARED.get('caption_str', ''),
            'obj_str':           SHARED.get('obj_str', ''),
            'retrieved_context': SHARED.get('retrieved_context', ''),
            'reset':             True,
            'user_message':      user_text,
        })
        timing["chat_engine_ms"] = round((time.perf_counter() - t0) * 1000)
        timing["gpt_ms"] = (res_chat or {}).get("timing", {}).get("gpt_ms")
        reply = res_chat.get("return_message", "What do you remember about this photograph?")

        # Persist opening turn.
        if reply:
            try:
                _save_conv_turn(photo_id, PATIENT_ID, user_text, reply, SERVER_IMAGE_LOCATION)
            except Exception as e:
                print(f"[ConvStore] Save failed: {e}")

        SHARED["dialog_started"] = False
        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
        _log_timing(timing)
        print(f"[Timing] {timing}")
        _post_trace({
            "ts":               datetime.datetime.now().isoformat(),
            "patient_id":       PATIENT_ID,
            "user_input":       user_text,
            "model_name":       (res_chat or {}).get("model_name", ""),
            "full_prompt":      (res_chat or {}).get("full_prompt", []),
            "raw_response":     (res_chat or {}).get("raw_response", ""),
            "final_response":   reply,
            "timing":           timing,
            "photo_id":         SHARED.get("current_photo_id", ""),
            "retrieved_context": SHARED.get("retrieved_context", ""),
            "rag_candidates":   _strip_candidate_embedding(candidates),
        })
        return {"text": reply}

    # ------------------------------------------------------------------
    # Core response logic
    # ------------------------------------------------------------------
    def interactive_running(self, data):
        user_utterance = data.get("text", "")
        timing = {"type": "text"}
        t_total = time.perf_counter()

        # Notify sim service (update dialogue similarity tracking) — pre.
        t0 = time.perf_counter()
        msg_sim = {'robot_reply': "", 'user_utterance': user_utterance}
        self.send_post_message_to_sim(msg_sim)
        timing["sim_pre_ms"] = round((time.perf_counter() - t0) * 1000)

        # Log input.
        print("_" * 90)
        print("Input utterance:")
        print(user_utterance)
        with open("bl_conversation_logs.txt", "a") as f:
            f.write("_" * 90 + "\n")
            f.write(f"{datetime.datetime.now()}\n")
            f.write(f"Input: {user_utterance}\n")

        # Snapshot shared image context under lock to avoid races with new-image uploads.
        with _state_lock:
            snap_caption   = SHARED.get('caption_str', '')
            snap_obj       = SHARED.get('obj_str', '')
            snap_photo_ctx = SHARED.get('photo_retrieved_context', SHARED.get('retrieved_context', ''))
            current_photo_id = SHARED.get('current_photo_id', '')

        text_memory = _prepare_text_memory_context(
            user_utterance,
            current_photo_id,
            snap_caption,
            snap_obj,
            PATIENT_ID,
            SERVER_IMAGE_LOCATION,
        )
        timing.update(text_memory.get("timing", {}))
        snap_retrieved = _combine_retrieved_context(snap_photo_ctx, text_memory.get("context", ""))
        with _state_lock:
            SHARED['turn_retrieved_context'] = text_memory.get("context", "")
            SHARED['retrieved_context'] = snap_retrieved

        # Call chat_engine (port 8087) — this is GPT under the hood.
        msg_chat = {
            'lang': self.input_language,
            'user_message': user_utterance,
            'caption_str': snap_caption,
            'obj_str': snap_obj,
            'retrieved_context': snap_retrieved,
        }
        t0 = time.perf_counter()
        res_chat = self.send_post_message_to_chat(msg_chat)
        timing["chat_engine_ms"] = round((time.perf_counter() - t0) * 1000)
        timing["gpt_ms"] = (res_chat or {}).get("timing", {}).get("gpt_ms")
        # GPT outputs in the correct language directly (zh instruction injected into prompt).
        reply = res_chat.get("return_message", "") if res_chat else ""

        print(f"Reply: {reply}")
        with open("bl_conversation_logs.txt", "a") as f:
            f.write(f"Reply: {reply}\n")

        # Persist this conversation turn linked to the current photo.
        if current_photo_id and reply:
            try:
                _save_conv_turn(current_photo_id, PATIENT_ID, user_utterance, reply, SERVER_IMAGE_LOCATION)
            except Exception as e:
                print(f"[ConvStore] Save failed: {e}")
        if reply:
            try:
                _save_text_episode_memory(
                    user_utterance,
                    reply,
                    current_photo_id,
                    PATIENT_ID,
                    text_memory.get("features", {}),
                    text_memory.get("embedding", []),
                )
            except Exception as e:
                print(f"[EpisodeMemory] Save failed: {e}")

        # Update sim with robot reply — post.
        t0 = time.perf_counter()
        msg_sim = {'robot_reply': reply, 'user_utterance': ""}
        self.send_post_message_to_sim(msg_sim)
        timing["sim_post_ms"] = round((time.perf_counter() - t0) * 1000)

        timing["total_ms"] = round((time.perf_counter() - t_total) * 1000)
        _log_timing(timing)
        print(f"[Timing] {timing}")
        _post_trace({
            "ts":               datetime.datetime.now().isoformat(),
            "patient_id":       PATIENT_ID,
            "user_input":       user_utterance,
            "model_name":       (res_chat or {}).get("model_name", ""),
            "full_prompt":      (res_chat or {}).get("full_prompt", []),
            "raw_response":     (res_chat or {}).get("raw_response", ""),
            "final_response":   reply,
            "timing":           timing,
            "photo_id":         SHARED.get("current_photo_id", ""),
            "retrieved_context": SHARED.get("retrieved_context", ""),
            "rag_candidates":   _strip_candidate_embedding(text_memory.get("candidates", [])),
        })
        return {"text": reply}

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------
    def _write_chunked(self, data: bytes) -> None:
        """Write one chunk in HTTP chunked-transfer-encoding format."""
        self.wfile.write(f"{len(data):X}\r\n".encode())
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _stream_interact(self, postvars: dict) -> None:
        """Handle /interact_stream: proxy SSE tokens from chat_engine to browser.

        Side-effect ordering mirrors interactive_running():
          sim_pre → GPT stream → sim_post + conv_save + trace
        """
        user_utterance = postvars.get("text", "")
        t_start = time.perf_counter()

        # sim_pre
        self.send_post_message_to_sim({'robot_reply': "", 'user_utterance': user_utterance})

        # Snapshot shared image context (thread-safe).
        with _state_lock:
            snap_caption   = SHARED.get('caption_str', '')
            snap_obj       = SHARED.get('obj_str', '')
            snap_photo_ctx = SHARED.get('photo_retrieved_context', SHARED.get('retrieved_context', ''))
            current_photo_id = SHARED.get('current_photo_id', '')

        text_memory = _prepare_text_memory_context(
            user_utterance,
            current_photo_id,
            snap_caption,
            snap_obj,
            PATIENT_ID,
            SERVER_IMAGE_LOCATION,
        )
        snap_retrieved = _combine_retrieved_context(snap_photo_ctx, text_memory.get("context", ""))
        with _state_lock:
            SHARED['turn_retrieved_context'] = text_memory.get("context", "")
            SHARED['retrieved_context'] = snap_retrieved

        msg_chat = {
            'lang':              self.input_language,
            'user_message':      user_utterance,
            'caption_str':       snap_caption,
            'obj_str':           snap_obj,
            'retrieved_context': snap_retrieved,
            'post_time':         str(time.time()),
        }

        # Send SSE response headers (no Content-Length; use chunked).
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

        full_reply = ""
        stream_full_prompt = []
        stream_raw_response = ""
        stream_model_name = ""
        stream_timing = {}
        got_done = False
        try:
            with requests.post(
                'http://127.0.0.1:8087/stream',
                json=msg_chat,
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"chat_engine /stream returned HTTP {r.status_code}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        try:
                            payload = json.loads(decoded[6:])
                            if payload.get('done'):
                                full_reply = payload.get('full', '')
                                stream_full_prompt = payload.get('full_prompt', [])
                                stream_raw_response = payload.get('raw_response', '')
                                stream_model_name = payload.get('model_name', '')
                                stream_timing = payload.get('timing', {}) or {}
                                got_done = True
                        except Exception:
                            pass
                    # Forward SSE line + double-newline to browser.
                    self._write_chunked((decoded + '\n\n').encode('utf-8'))
        except Exception as exc:
            print(f"[stream] chat_engine error: {exc}")
            try:
                self._write_chunked(
                    f"data: {json.dumps({'error': str(exc)})}\n\n".encode('utf-8')
                )
            except Exception:
                pass

        # Always send a done event so the frontend never hangs.
        if not got_done:
            try:
                self._write_chunked(
                    f"data: {json.dumps({'done': True, 'full': full_reply})}\n\n".encode('utf-8')
                )
            except Exception:
                pass

        # End of chunked stream.
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

        # Post-stream side effects (mirrors interactive_running).
        if full_reply and current_photo_id:
            try:
                _save_conv_turn(current_photo_id, PATIENT_ID, user_utterance, full_reply, SERVER_IMAGE_LOCATION)
            except Exception as exc:
                print(f"[ConvStore] Save failed: {exc}")
        if full_reply:
            try:
                _save_text_episode_memory(
                    user_utterance,
                    full_reply,
                    current_photo_id,
                    PATIENT_ID,
                    text_memory.get("features", {}),
                    text_memory.get("embedding", []),
                )
            except Exception as exc:
                print(f"[EpisodeMemory] Save failed: {exc}")

        # sim_post
        self.send_post_message_to_sim({'robot_reply': full_reply, 'user_utterance': ""})

        total_ms = round((time.perf_counter() - t_start) * 1000)
        timing = {"type": "text_stream", "total_ms": total_ms}
        timing.update(text_memory.get("timing", {}))
        timing.update(stream_timing)
        _log_timing(timing)
        print(f"[Timing] {timing}")
        _post_trace({
            "ts":                datetime.datetime.now().isoformat(),
            "patient_id":        PATIENT_ID,
            "user_input":        user_utterance,
            "model_name":        stream_model_name,
            "full_prompt":       stream_full_prompt,
            "raw_response":      stream_raw_response,
            "final_response":    full_reply,
            "timing":            timing,
            "photo_id":          SHARED.get("current_photo_id", ""),
            "retrieved_context": SHARED.get("retrieved_context", ""),
            "rag_candidates":    _strip_candidate_embedding(text_memory.get("candidates", [])),
        })

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def send_post_message_to_sim(self, msg):
        url = 'http://127.0.0.1:9110/'
        headers = {"Content-Type": "application/json"}
        msg["post_time"] = str(time.time())
        try:
            req = requests.post(url, data=json.dumps(msg, sort_keys=True, separators=(',', ':')), headers=headers)
            if req.status_code == requests.codes.ok:
                return req.json()
        except Exception as e:
            print(f"[sim] request failed: {e}")
        return {}

    def send_post_message_to_chat(self, msg):
        url = 'http://127.0.0.1:8087/'
        headers = {"Content-Type": "application/json"}
        msg["post_time"] = str(time.time())
        try:
            req = requests.post(url, data=json.dumps(msg, sort_keys=True, separators=(',', ':')), headers=headers)
            if req.status_code == requests.codes.ok:
                return req.json()
        except Exception as e:
            print(f"[chat_engine] request failed: {e}")
        return {}

    def send_post_message_iu(self, msg, url):
        headers = {"Content-Type": "application/json"}
        msg["post_time"] = str(time.time())
        try:
            req = requests.post(url, data=json.dumps(msg, sort_keys=True, separators=(',', ':')), headers=headers)
            if req.status_code == requests.codes.ok:
                return req.json()
        except Exception as e:
            print(f"[IU] request to {url} failed: {e}")
        return {}

    # ------------------------------------------------------------------
    # POST /interact
    # ------------------------------------------------------------------
    def process_post(self, form):
        text = form.get("text", "")
        image_name = form.get("image_name", "")
        image_interactive = form.get("image", "")
        metadata = form.get("metadata", "")
        if metadata:
            print(f">>> get metadata: {metadata}")
        img_id = form.get("img_id", "")
        cate = form.get("cate", "")
        return {"text": text, "image_name": image_name,
                "image_interactive": image_interactive, "metadata": metadata,
                "img_id": img_id, "cate": cate}

    def do_POST(self):
        if self.path == "/set_user":
            global PATIENT_ID
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            new_id = data.get("user_id", "").strip()
            if new_id:
                with _state_lock:
                    PATIENT_ID = new_id
                _persist_user(PATIENT_ID, SERVER_IMAGE_LOCATION)
                print(f"[PATIENT_ID] 切換至：{PATIENT_ID}")
            body = bytes(json.dumps({"ok": True, "current": PATIENT_ID}), "utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/interact_stream":
            form = _parse_multipart(self.rfile, self.headers)
            postvars = self.process_post(form)
            if postvars.get("text"):
                SHARED["dialog_started"] = True
                self._stream_interact(postvars)
            else:
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()
            return

        if self.path != "/interact":
            return self.respond({"status": 500})

        form = _parse_multipart(self.rfile, self.headers)
        postvars = self.process_post(form)

        model_response = {}

        # -------- New image (path-based) --------
        if postvars["image_name"]:
            SHARED["image_name"] = postvars["image_name"]
            image_location = os.path.join(SERVER_IMAGE_LOCATION, SHARED["image_name"])

            # Copy to patient folder for permanent storage.
            patient_dir = os.path.join(SERVER_IMAGE_LOCATION, PATIENT_ID)
            os.makedirs(patient_dir, exist_ok=True)
            img_save_path = os.path.join(patient_dir, SHARED["image_name"])
            if SERVER_IMAGE_LOCATION.find('http') == -1:
                img = Image.open(image_location).convert("RGB")
            else:
                img = Image.open(requests.get(image_location, stream=True).raw).convert("RGB")
            img.save(img_save_path)

            model_response = self._handle_new_image(img_save_path, SHARED["image_name"], postvars.get('text', ''))

        # -------- New image (base64 / interactive upload) --------
        elif postvars["image_interactive"] != "":
            img_data = str(postvars["image_interactive"])
            _, encoded = img_data.split(",", 1)
            image = Image.open(io.BytesIO(b64decode(encoded))).convert("RGB")
            img_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

            # Save to patient folder for permanent storage.
            patient_dir = os.path.join(SERVER_IMAGE_LOCATION, PATIENT_ID)
            os.makedirs(patient_dir, exist_ok=True)
            img_save_path = os.path.join(patient_dir, img_filename)
            image.save(img_save_path)

            model_response = self._handle_new_image(img_save_path, img_filename, postvars.get('text', ''))

        # -------- Text message (normal conversation turn) --------
        elif postvars["text"]:
            SHARED["dialog_started"] = True
            model_response = self.interactive_running(postvars)

        # -------- Metadata update --------
        elif postvars["metadata"]:
            print(postvars["metadata"])
            SHARED['metadata'] = postvars["metadata"]
            msg = {'metadata': postvars["metadata"]}
            model_response = self.send_post_message_to_sim(msg)

            if self.input_language == 'en' and 'question_question' in model_response:
                model_response["question_question"] = trans_zh_en.translate(
                    model_response["question_question"])

        body = bytes(json.dumps(model_response), "utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # GET (serve the web UI or /users)
    # ------------------------------------------------------------------
    def do_GET(self):
        if self.path == "/Examples.png":
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples.png")
            try:
                with open(img_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
            return

        if self.path == "/users":
            # Merge users.json (all ever-added users) with ChromaDB (has photos)
            users = set(_load_users(SERVER_IMAGE_LOCATION, default_user=PATIENT_ID))
            try:
                users |= set(_photo_db.list_patients())
            except Exception as _e:
                print(f"[Users] list_patients failed: {_e}")
            users.add(PATIENT_ID)
            body = bytes(json.dumps({"users": sorted(users), "current": PATIENT_ID}), "utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        paths = {
            "/":           {"status": 200},
            "/favicon.ico": {"status": 202},
        }
        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({"status": 500})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def handle_http(self, status_code, path, text=None):
        content = bytes(WEB_HTML.format(STYLE_SHEET, FONT_AWESOME), "UTF-8")
        self.send_response(status_code)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        return content

    def respond(self, opts):
        response = self.handle_http(opts["status"], self.path)
        self.wfile.write(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise shared state.
    SHARED['metadata']           = {}
    SHARED['caption_str']        = ""
    SHARED['obj_str']            = ""
    SHARED['image_embedding']    = []
    SHARED['photo_retrieved_context'] = ""
    SHARED['turn_retrieved_context'] = ""
    SHARED['retrieved_context']  = ""
    SHARED['dialog_started']     = False

    # 確保圖片上傳目錄存在
    os.makedirs(SERVER_IMAGE_LOCATION, exist_ok=True)
    print(f"Image upload directory: {SERVER_IMAGE_LOCATION}")
    _persist_user(PATIENT_ID, SERVER_IMAGE_LOCATION)  # 確保預設使用者寫入 users.json

    server_class = ThreadingHTTPServer
    Handler = MyHandler
    Handler.protocol_version = "HTTP/1.1"
    httpd = server_class((HOST_NAME, PORT), Handler)

    print(f"\nVisit http://{HOST_NAME}:{PORT}/ to chat with the model!")
    print("(No ParlAI / BlenderBot required.)\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
