# -*- coding:utf-8 -*-
import re
import json
import logging
import datetime
import time
from flask import Flask, request, Response
from deep_translator import GoogleTranslator

from config import parse_args
args = parse_args()

import os
import openai
os.environ['OPENAI_API_KEY'] = args.openai_key
openai.api_key = args.openai_key


app = Flask(__name__)

class SocialREMChat(object):
    def __init__(self, lang) -> None:
        self.caption_str = ""
        self.obj_str = ""
        self.retrieved_context = ""
        if lang == 'zh':
            self.system_task = "\n你是一個陪伴機器人，引導User完成以照片為中心的回憶任務。\
                                    在這裡，以照片為中心的回憶是一種常見的療法，可以幫助患有維度障礙的患者增強認知能力\
                                    通過根據圖像內容與User聊天。 \
                                    \n 您需要提出問題來幫助User回憶當天的記憶，\
                                    並從User回復中提取以下五個主要問題的答案，這裡我們將其稱為攝影要點：\
                                    \n 1.這張照片中的主要事件是什麼？ \
                                    \n 2. 該事件何時發生？ \
                                    \n 3.本次活動在哪裡舉行？ \
                                    \n 4. 當時User和誰在一起？ \
                                    \n 5. User可以記住哪些細節？ \
                                    \n 提出這些問題的順序可以隨意，並使用更生動的詞語。 \
                                    同時，您需要向User提供同理心的響應，\
                                    這意味著，如果User試圖講述故事，它應該積極響應User的表達。 \
                                    但需要注意的是，回覆應該是一兩句話，以使對話流暢。\
                                    \n 此外，如果User提及了相冊中其他照片的回憶，請簡短認同並串聯那段記憶（例如：『對，那張照片也很有紀念性！』），再溫和地引導User繼續回憶當前照片的要點。" + '\n\n'
            self.system_strategies = "\n根據User的最後一句話，首先推斷其中是否有要點的答案。 \
                                    \n 如果答案存在，\
                                        更新哪些要點仍有待回答。 \
                                    \n 然後從支持策略中選擇一個支持策略並解釋選擇它的原因。 最後，給User一個適當的響應。" + '\n\n' \
                                    "支持策略：[提出問題]、[重述或釋義]、[感受反映]、[自我表露]、[肯定和保證]、[提供建議]、[提供資訊]、[其他]" + '\n\ n ' + \
                                    "支持策略定義：" + '\n' \
                                    "[提出問題]：當助理需要提出問題以幫助User回憶提供五個問題答案的照片的要點時，可以使用此策略。問題需要與照片的內容或其中出現的物件相關。" + '\n' \
                                    "[重述或釋義]：助手可以使用此策略重述User的情況，以幫助User了解他們所面臨的情況。" + '\n' \
                                    "[感受的反映]：助理需要澄清或描述User的感受時可以使用此策略。" + '\n' \
                                    "[自我披露]：當需要同情User或分享基於經驗的回覆時，助理可以使用此策略。" + '\n' \
                                    "[肯定和保證]：當需要肯定User的能力或提供鼓勵和保證時，助手可以使用此策略。" + '\n' \
                                    "[提供建議]：當需要提供改變建議時，助手會利用這個策略來提供一些建議或者解決方案。" + '\n' \
                                    "[提供資訊]：當需要提供有關特定主題的知識或信息時，助理可以使用此策略。" + '\n' \
                                    "[其他]：當上述策略都不合適時，或者當助理想要提供熱情友好的問候時，可以使用此策略。" + '\n\n'
            self.system_observations = "鑑於照片的描述："
            self.system_observations_obj = "我們還可以注意到這張照片中有這些物件："
            self.conversation_prompt = "對話歷史記錄：" + '\n'
            self.last_message_prompt = '\n\n' + "=== User的最後一句話 ==" + '\n'
            self.seperate_prompt = '\n\n###\n\n'
            self.instruct_prompt = "助理將根據事實回答需求格式的問題，並以需求格式做出回應。" + '\n' \
                                    "需求格式：" + '\n' \
                                    "1. User當前話語：" + '\n' \
                                    "2. 下列句子的基本要素是： " + '\n' \
                                    "3. 該話語包含必要的答案：" + '\n' \
                                    "4. 剩下要回答的要點：" + '\n' \
                                    "5. 助手做了什麼：" + '\n' \
                                    "6. ===User最後一句話的完整內容是什麼===: " + '\n' \
                                    "7. 根據 === User最後的陳述選擇支持策略 ===: " + '\n' \
                                    "8. 選擇支持策略的原因：" + '\n' + \
                                    "9. 回覆===User最後一句===（最多2句）：\n" \
                                    'JSON_OUTPUT: {"reply": "<回覆內容>", "strategy": "<所選策略>"}'
        elif lang == 'en':
            self.system_task = "\n You are a companion robot that guide User to complete the task of photo-centered reminiscence. \
                                        Here, photo-centered reminiscence is a common therapy that help patients suffering from Dimensia enhance their cognitive capability \
                                        through chatting with User according to the content of an image. \
                                        \n You need to propose questions that help User recall the memomery of that day, \
                                        and extract the answers from User replies of the following five main questions, here we term them photograph essentials:\
                                        \n 1. what was the main event in this photograph? \
                                        \n 2. when did this event happen? \
                                        \n 3. where did this event be hold? \
                                        \n 4. who were with User at that moment? \
                                        \n 5. Which details can User memorize? \
                                        \n The order of proposing these questions can be random, and use more vivid words. \
                                        And the same time, you need to provide empathetic response to User, \
                                        which means, it should actively response to User's expressions if they try to tell the story. \
                                        But it need to be memtioned that the reponse should be one or two sentences to make the conversation fluent. \
                                        \n Additionally, if the User mentions a memory related to another photo in their album, briefly acknowledge and affirm that connection (e.g. 'That sounds like another wonderful memory!'), then gently guide the conversation back to the current photograph's essentials." + '\n\n'
            self.system_strategies = "\n Accoriding to User's last utterance, you should first infer if there are answers of essentials in it. \
                                        \n If the answer exist,\
                                                update which essentials are remained to be answered. \
                                        \n Then select a Support Strategy from the Support Strategies and explains why it was chosen. In the end, an appropriate response is given to the User." + '\n\n' \
                                        "Support Strategies: [Proposing Question], [Restatement or Paraphrasing], [Reflection of feelings], [Self-disclosure], [Affirmation and Reassurance], [Providing Suggestions], [Providing Information], [Others]" + '\n\n' + \
                                        "Support Strategies Definitions:" + '\n' \
                                        "[Proposing Question]: This Strategy is used when the Assistant needs to ask a question to help User recall the essentials of photograph that provides the answers of the five questions. Questions need to be related to the content of the photo or the objects that appear in it." + '\n' \
                                        "[Restatement or Paraphrasing]: Assistant can use this Strategy to restate the User's situation to help the User know what they are facing." + '\n' \
                                        "[Reflection of feelings]: Assistant can use this Strategy when it needs to clarify or describe User's feelings." + '\n' \
                                        "[Self-disclosure]: This Strategy can be used by the Assistant when there is a need to empathize with the User or share an experience based reply." + '\n' \
                                        "[Affirmation and Reassurance]: This Strategy can be used by the Assistant when there is a need to affirm the User's abilities or provide encouragement and reassurance." + '\n' \
                                        "[Providing Suggestions]: When there is a need to provide suggestions for change, the Assistant will use this Strategy to provide some suggestions or solutions." + '\n' \
                                        "[Providing Information]: This Strategy can be used by Assistant when there is a need to provide knowledge or information about a specific topic." + '\n' \
                                        "[Others]: This Strategy is used when none of the above strategies are appropriate, or when the Assistant wants to offer a warm and friendly greeting." + '\n\n'
            self.system_observations = "Given the description of the photograph: " 
            self.system_observations_obj = "We can also notice there are objects in this photograph: "
            self.conversation_prompt = "Conversation History: " + '\n'
            self.last_message_prompt = '\n\n' + "=== User's last utterance ===" + '\n'
            self.seperate_prompt = '\n\n###\n\n'
            self.instruct_prompt = "Assistant will answer Demand Format questions based on facts and will respond in Demand Format." + '\n' \
                                   "Demand Format: " + '\n' \
                                   "1. User's current utterance: " + '\n' \
                                   "2. Which essential is the utterance related to : " + '\n' \
                                   "3. The utterance contains the answer of essenital:" + '\n' \
                                   "4. The remaining essentials to be answered: " + '\n' \
                                   "5. What the Assistant has done: " + '\n' \
                                   "6. What is the full statement in the === User's last utterance ===: " + '\n' \
                                   "7. Select Support Strategy based on === User's last statement ===: " + '\n' \
                                   "8. The reason for choosing the Support Strategy: " + '\n' + \
                                   "9. Reply === User's last sentence === (up to 2 sentences): \n" \
                                   'JSON_OUTPUT: {"reply": "<reply text>", "strategy": "<chosen strategy>"}'
        
        self._last_full_prompt: list = []
        self._last_raw_response: str = ""

        # gpt-5-mini (and similar reasoning models) do not support sampling parameters.
        _sampling_unsupported = args.model_name.startswith('gpt-5')
        if _sampling_unsupported:
            self.generate_kwargs = {}
        else:
            self.generate_kwargs = {
                'temperature': args.temparature,
                'top_p': args.top_p,
                'frequency_penalty': args.frequency_penalty,
                'presence_penalty': args.presence_penalty,
            }
    
    def _build_retrieved_block(self):
        if not self.retrieved_context:
            return ""
        return (
            "\n[Related autobiographical memories]\n"
            "The block below may contain related past photos, past conversation turns, "
            "and a structured feature summary of the user's current turn.\n"
            "Rule 1: Do NOT project details from past photos or past turns onto the current photograph's essentials. "
            "Use them only as autobiographical context unless the user explicitly connects them.\n"
            "Rule 2: If the User explicitly asks whether you remember something they mentioned before, "
            "you SHOULD reference the past conversation below to confirm the specific fact, "
            "then gently return to exploring the current photo's own essentials.\n"
            "Rule 3: You MAY proactively say 'This reminds me of [brief description from memory]...' "
            "ONLY when the connection is strong and you can cite a specific verifiable detail. "
            "If uncertain, ask: 'Did you also talk about...?' instead of asserting.\n"
            + self.retrieved_context
            + "\n[End of related autobiographical memories.]\n"
        )

    def preprocess_conversation(self, context, max_turn, reply_lang=None):
        _context = list()

        self.observation_prompt = self.system_observations + self.caption_str + "\n" + self.system_observations_obj + self.obj_str
        self.system_prompt = self.system_task + self.system_strategies + self.observation_prompt + self._build_retrieved_block()

        for idx in range(len(context)):
            for k, v in context[idx].items():
                _context.append('{}：{}'.format(k, v))

        if max_turn == -1: pass
        else: _context = _context[-(max_turn * 2): ]

        print('[Context]: \n' + '\n'.join(_context))

        _context[-1] = _context[-1] + self.last_message_prompt + _context[-1]

        lang_to_use = reply_lang or args.lang
        lang_suffix = "\n請以繁體中文（台灣用語）回覆使用者。" if lang_to_use == 'zh' else ""
        _context = self.system_prompt + self.conversation_prompt + '\n'.join(_context) + self.seperate_prompt + self.instruct_prompt + lang_suffix

        full_prompt = [{'role': 'system', 'content': _context}]

        print('\n' + '='*60)
        print('[FULL PROMPT TO GPT]')
        print(_context)
        print('='*60 + '\n')

        return full_prompt

    def preprocess_conversation_simple(self, context, max_turn, reply_lang=None):
        """Simplified prompt for streaming mode: no CoT/JSON, direct reply in 1-2 sentences."""
        _context = list()

        self.observation_prompt = self.system_observations + self.caption_str + "\n" + self.system_observations_obj + self.obj_str

        for idx in range(len(context)):
            for k, v in context[idx].items():
                _context.append('{}：{}'.format(k, v))

        if max_turn != -1:
            _context = _context[-(max_turn * 2):]

        lang_to_use = reply_lang or args.lang
        lang_suffix = "\n請以繁體中文（台灣用語）回覆使用者。" if lang_to_use == 'zh' else ""
        if lang_to_use == 'zh':
            simple_strategy = (
                "\n支援策略只作為內部回覆風格使用，不要說出策略名稱或選擇原因。"
                "輸出只能是要直接顯示給使用者的一到兩句自然回覆。"
                "禁止輸出『選用策略』、『支援策略』、『原因』、『分析』、條列、格式模板或 JSON。"
                "如果使用者已經回答了時間、地點或人物，先簡短肯定，再只追問一個尚未回答或最自然的細節問題。\n"
            )
            simple_inst = "請直接回覆使用者（最多2句話）。不要輸出分析步驟、策略說明或 JSON。"
        else:
            simple_strategy = (
                "\nUse a support strategy silently as an internal style guide only. "
                "Never mention the strategy name, selected strategy, or reason. "
                "Output only the exact assistant message that should be shown to the user, in 1-2 natural sentences. "
                "Do not output labels, bullets, analysis, structured templates, or JSON. "
                "Reply in the same language as the user's latest utterance; if the user writes Chinese, reply in Traditional Chinese (Taiwan usage). "
                "If the user has already answered when, where, or who, briefly affirm that information, then ask only one remaining or natural detail question.\n"
            )
            simple_inst = "Reply directly to the User in 1-2 sentences. Do not output analysis steps, strategy explanations, or JSON."

        self.system_prompt = self.system_task + self.observation_prompt + self._build_retrieved_block() + simple_strategy

        full_context = (
            self.system_prompt
            + self.conversation_prompt
            + '\n'.join(_context)
            + self.seperate_prompt
            + simple_inst
            + lang_suffix
        )
        return [{'role': 'system', 'content': full_context}]

    def postprocess_response_text(self, top_response):
        # Primary: parse JSON_OUTPUT tag produced by the instruct_prompt.
        # Use [^\n]+ (not [^}]+) so reply text containing } doesn't truncate early.
        json_match = re.search(r'JSON_OUTPUT:\s*(\{[^\n]+\})', top_response)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                assistant_response = parsed.get('reply', '').strip()
                if assistant_response:
                    return assistant_response, [top_response]
            except (json.JSONDecodeError, KeyError):
                pass

        # Strip JSON_OUTPUT line before regex fallbacks to prevent noise leaking to user.
        response_for_regex = re.sub(r'\nJSON_OUTPUT:[^\n]*', '', top_response)

        # Secondary: find step 9 directly in the raw output,
        # robust to missing newlines between steps (e.g. gpt-5-mini merging steps).
        step9_match = re.search(r'(?<!\d)9\.[^:：\n]*[:：](.*)', response_for_regex, re.DOTALL)
        if step9_match:
            assistant_response = step9_match.group(1).strip()
            cot_response = [top_response]
        else:
            # Fallback: split by numbered steps (original logic, works for gpt-3.5/4).
            pattern = r'\b\d+\..+?(?=\n\d+\.|\Z)'
            cot_response = re.findall(pattern, response_for_regex, re.DOTALL)
            if len(cot_response) > 9:
                cot_response[8] = '\n'.join(cot_response[8:])
                cot_response = cot_response[:9]
            assistant_response = cot_response[-1]
            if ':' in assistant_response or '：' in assistant_response:
                assistant_response = re.split(r':|：', assistant_response, 1)[-1].strip()

        # Shared cleanup
        if assistant_response.startswith('9. '):
            assistant_response = assistant_response[3:].strip()
        if assistant_response.startswith('回覆'):
            assistant_response = assistant_response[2:].strip()
        if 'Assistant：' in assistant_response:
            assistant_response = re.sub('Assistant：', '', assistant_response)
        if assistant_response and assistant_response[0] in ('"', '「', '['):
            assistant_response = assistant_response[1:]
        if assistant_response and assistant_response[-1] in ('"', '」', ']'):
            assistant_response = assistant_response[:-1]

        return assistant_response, cot_response

    def postprocess_response(self, response):
        return self.postprocess_response_text(response.choices[0].message.content)
    
    def generate_opening(self, user_message='', reply_lang=None):
        # Always reset context when a new image is uploaded.
        self.context = []

        self.observation_prompt = self.system_observations + self.caption_str + "\n" + self.system_observations_obj + self.obj_str
        system_content = self.system_task + self.system_strategies + self.observation_prompt + self._build_retrieved_block()

        lang_to_use = reply_lang or args.lang
        lang_suffix = "\n請以繁體中文（台灣用語）回覆使用者。" if lang_to_use == 'zh' else ""

        if user_message:
            first_turn = user_message + '\n\n' + self.instruct_prompt + lang_suffix
            self.context = [{'User': user_message}]
        else:
            if lang_to_use == 'zh':
                opening_instruction = "這是對話的開場，尚未有User的發言。根據照片內容選擇一個合適的支援策略，並生成一個溫暖親切的開場問題（只問一個）。"
            else:
                opening_instruction = "This is the opening of the conversation with no prior user utterance. Based on the photo content, choose one Support Strategy and generate a warm opening with exactly ONE question."
            first_turn = opening_instruction + '\n\n' + self.instruct_prompt + lang_suffix
            self.context = []

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': first_turn},
        ]
        self._last_full_prompt = messages
        client = openai.OpenAI(api_key=args.openai_key)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            **self.generate_kwargs
        )
        gpt_ms = round((time.perf_counter() - t0) * 1000)
        self._last_raw_response = response.choices[0].message.content
        opening, _ = self.postprocess_response(response)
        self.context.append({'Assistant': opening})
        return opening, gpt_ms

    def _check_response_grounding(self, response: str, client: openai.OpenAI) -> str:
        """Lightweight evidence check: verify the response only cites verifiable memory details.

        Called only when retrieved_context is non-empty. Returns the original response
        if it passes, or a hedged rewrite if it cites unverifiable specifics.
        Skip if args.evidence_check is False (default off to avoid added latency).
        """
        if not getattr(args, 'evidence_check', False):
            return response
        if not self.retrieved_context:
            return response
        prompt = (
            f"Memory block available:\n{self.retrieved_context}\n\n"
            f"Assistant reply:\n{response}\n\n"
            "Does the reply cite only details present in the memory block above? "
            "If yes, respond with EXACTLY the original reply unchanged. "
            "If no, rewrite it so uncertain claims use hedged language "
            "('I think...', 'It looks like...', 'Did you mention...?'). "
            "Respond with the final reply text only."
        )
        try:
            result = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return result.choices[0].message.content.strip() or response
        except Exception as exc:
            logging.warning("evidence check failed: %s", exc)
            return response

    def chatting(self, context, reply_lang=None):
        processed_context = self.preprocess_conversation(context, args.max_turn, reply_lang=reply_lang)
        self._last_full_prompt = processed_context

        client = openai.OpenAI(api_key=args.openai_key)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model_name,
            messages=processed_context,
            **self.generate_kwargs
        )
        gpt_ms = round((time.perf_counter() - t0) * 1000)
        self._last_raw_response = response.choices[0].message.content

        assistant_response, cot_response = self.postprocess_response(response)
        assistant_response = self._check_response_grounding(assistant_response, client)

        return assistant_response, cot_response, gpt_ms


@app.route("/", methods=["POST"])
def post_method():
    if request.method == "POST":
        data = json.loads(request.data)

        if 'caption_str' in data:
            _socialREMChat.caption_str = data['caption_str']

        if 'obj_str' in data:
            _socialREMChat.obj_str = data['obj_str']

        if 'retrieved_context' in data:
            _socialREMChat.retrieved_context = data['retrieved_context']

        req_lang = data.get('lang', args.lang)

        # New image: reset context and generate GPT opening based on image content.
        if data.get('reset'):
            opening, gpt_ms = _socialREMChat.generate_opening(user_message=data.get('user_message', ''), reply_lang=req_lang)
            return json.dumps({
                "return_message": opening,
                "last": False,
                "timing": {"gpt_ms": gpt_ms},
                "full_prompt": _socialREMChat._last_full_prompt,
                "raw_response": _socialREMChat._last_raw_response,
                "model_name": args.model_name,
            })

        if 'user_message' in data:
            user_message = data['user_message']
            _socialREMChat.context.append({'User': user_message})

        if end_trigger in _socialREMChat.context[-1]['User'].lower():
            last = True
            response = "Closing this conversation"
            save_file(_socialREMChat.context)
            return json.dumps({"return_message": response, "last": last, "timing": {}})

        else:
            last = False
            response, cot_response, gpt_ms = _socialREMChat.chatting(context=_socialREMChat.context, reply_lang=req_lang)
            _socialREMChat.context.append({'Assistant': response})

            print('[Chain-of-Thought]: ')
            for output_step in cot_response:
                print(output_step)

        return json.dumps({
            "return_message": response,
            "last": last,
            "timing": {"gpt_ms": gpt_ms},
            "full_prompt": _socialREMChat._last_full_prompt,
            "raw_response": _socialREMChat._last_raw_response,
            "model_name": args.model_name,
        })
    else:
        return json.dumps({"return_message": 'Invalid request method'})


@app.route("/stream", methods=["POST"])
def post_stream():
    """Fast streaming endpoint for the unchecked precise-mode UI path.

    This intentionally uses the simplified direct-reply prompt instead of the
    Demand Format/JSON_OUTPUT prompt used by /. That lets us forward model
    tokens to the browser as they arrive without leaking CoT/debug text.
    """
    data = json.loads(request.data)

    if 'caption_str' in data:
        _socialREMChat.caption_str = data['caption_str']
    if 'obj_str' in data:
        _socialREMChat.obj_str = data['obj_str']
    if 'retrieved_context' in data:
        _socialREMChat.retrieved_context = data['retrieved_context']

    req_lang = data.get('lang', args.lang)
    user_message = data.get('user_message', '')
    _socialREMChat.context.append({'User': user_message})

    # End trigger: emit a single token and close.
    if end_trigger in user_message.lower():
        save_file(_socialREMChat.context)
        closing = 'Closing this conversation'
        def _end_gen():
            yield f"data: {json.dumps({'token': closing})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': closing})}\n\n"
        return Response(_end_gen(), mimetype='text/event-stream')

    processed_context = _socialREMChat.preprocess_conversation_simple(
        _socialREMChat.context, args.max_turn, reply_lang=req_lang
    )
    _socialREMChat._last_full_prompt = processed_context

    client = openai.OpenAI(api_key=args.openai_key)

    def generate():
        raw_text = ""
        final_reply = ""
        gpt_ms = None
        success = False
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=args.model_name,
                messages=processed_context,
                stream=True,
                **_socialREMChat.generate_kwargs
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                token = (delta.content or "") if delta else ""
                if token:
                    raw_text += token
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            gpt_ms = round((time.perf_counter() - t0) * 1000)
            _socialREMChat._last_raw_response = raw_text
            final_reply = raw_text.strip()
            success = True
            if final_reply:
                _socialREMChat.context.append({'Assistant': final_reply})
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        done_payload = {
            'done': True,
            'full': final_reply if success else raw_text.strip(),
            'full_prompt': processed_context,
            'raw_response': raw_text,
            'model_name': args.model_name,
            'timing': {'gpt_ms': gpt_ms} if gpt_ms is not None else {},
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


def save_file(context):
    now = datetime.datetime.now()
    time = str(now.month) + str(now.day) + '_' + now.strftime('%H%M')

    with open(f'./storage/{time}.json', 'w', encoding='utf-8') as f:
        json.dump({'context': context}, f, ensure_ascii=False)

    return True


if __name__ == "__main__":
    global _socialREMChat
    _socialREMChat = SocialREMChat(lang=args.lang)
    _socialREMChat.context = list()

    if args.lang == 'zh':
        end_trigger = '結束對話'
        # reset the context
        _socialREMChat.context = [{'Assistant': '你好，關於這張照片有什麼事情想跟我聊聊嗎？'}]

    else:
        end_trigger = 'conversation over'
        # reset the context
        _socialREMChat.context = [{'Assistant': 'Hello, Is there anything you want to talk to me about this photograph?'}]

    app.run(host="0.0.0.0", port=8087)
