# -*- coding:utf-8 -*-
import re
import json
import logging
import datetime
from flask import Flask, request
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
                                    但需要注意的是，回覆應該是一兩句話，以使對話流暢。" + '\n\n'
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
                                    "9. 回覆===User最後一句===（最多2句）："
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
                                        But it need to be memtioned that the reponse should be one or two sentences to make the conversation fluent." + '\n\n'
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
                                   "9. Reply === User's last sentence === (up to 2 sentences): "
        
        self.generate_kwargs = {
            'temperature': args.temparature, 
            'top_p': args.top_p, 
            'frequency_penalty': args.frequency_penalty, 
            'presence_penalty': args.presence_penalty
        }
    
    def preprocess_conversation(self, context, max_turn):
        _context = list()

        self.observation_prompt = self.system_observations + self.caption_str + self.system_observations_obj + self.obj_str
        self.system_prompt = self.system_task + self.system_strategies + self.observation_prompt
        
        for idx in range(len(context)):
            for k, v in context[idx].items():
                _context.append('{}：{}'.format(k, v))

        if max_turn == -1: pass
        else: _context = _context[-(max_turn * 2): ]
        
        print('[Context]: \n' + '\n'.join(_context))
        
        _context[-1] = _context[-1] + self.last_message_prompt + _context[-1]
        _context = self.system_prompt + self.conversation_prompt + '\n'.join(_context) + self.seperate_prompt + self.instruct_prompt
        
        full_prompt = [{'role': 'system', 'content': _context}]

        return full_prompt

    def postprocess_response(self, response):
        top_response = response.choices[0].message.content

        pattern = r'\b\d+\..+?(?=\n\d+\.|\Z)'
        cot_response = re.findall(pattern, top_response, re.DOTALL)
        if len(cot_response) > 9:
            cot_response[8] = '\n'.join(cot_response[8: ])
            cot_response = cot_response[: 9]
    
        assistant_response = cot_response[-1]

        if ':' in assistant_response or '：' in assistant_response:
            assistant_response = re.split(r':|：', assistant_response, 1)[-1].strip()

        if assistant_response.startswith('9. '): 
            assistant_response = assistant_response[3: ].strip()

        if assistant_response.startswith('回覆'):
            assistant_response = assistant_response[2: ].strip()

        if 'Assistant：' in assistant_response:
            assistant_response = re.sub('Assistant：', '', assistant_response)

        if assistant_response[0] == '\"' or assistant_response[0] == '「' or assistant_response[0] == '[':
            assistant_response = assistant_response[1: ]

        if assistant_response[-1] == '\"' or assistant_response[-1] == '」' or assistant_response[-1] == ']':
            assistant_response = assistant_response[: -1]

        if assistant_response.startswith('9. '): 
            assistant_response = assistant_response[3: ]
        if assistant_response[0] == '\"' or assistant_response[0] == '「' or assistant_response[0] == '[':
            assistant_response = assistant_response[1: ]
        if assistant_response[-1] == '\"' or assistant_response[-1] == '」' or assistant_response[-1] == ']':
            assistant_response = assistant_response[: -1]

        return assistant_response, cot_response
    
    def generate_opening(self):
        self.observation_prompt = self.system_observations + self.caption_str + self.system_observations_obj + self.obj_str
        system_content = self.system_task + self.system_strategies + self.observation_prompt

        if args.lang == 'zh':
            trigger = "請根據照片內容，用親切的方式開啟對話，提出第一個能引導User回憶的問題。（最多兩句）"
        else:
            trigger = "Based on the photo content, warmly open the conversation and ask the first question to help the User recall their memories. (at most 2 sentences)"

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': trigger},
        ]
        client = openai.OpenAI(api_key=args.openai_key)
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            **self.generate_kwargs
        )
        opening = response.choices[0].message.content.strip()
        self.context = [{'Assistant': opening}]
        return opening

    def chatting(self, context):
        processed_context = self.preprocess_conversation(context, args.max_turn)

        client = openai.OpenAI(api_key=args.openai_key)
        response = client.chat.completions.create(
            model=args.model_name,
            messages=processed_context,
            **self.generate_kwargs
        )

        assistant_response, cot_response = self.postprocess_response(response)

        return assistant_response, cot_response


@app.route("/", methods=["POST"])
def post_method():
    if request.method == "POST":
        data = json.loads(request.data)

        if 'caption_str' in data:
            _socialREMChat.caption_str = data['caption_str']

        if 'obj_str' in data:
            _socialREMChat.obj_str = data['obj_str']

        # New image: reset context and generate GPT opening based on image content.
        if data.get('reset'):
            opening = _socialREMChat.generate_opening()
            return json.dumps({"return_message": opening, "last": False})

        if 'user_message' in data:
            user_message = data['user_message']
            _socialREMChat.context.append({'User': user_message})

        if end_trigger in _socialREMChat.context[-1]['User'].lower():
            last = True
            response = "Closing this conversation"
            # user_transcripts = list()
            # for utter in context:
            #     if 'User' in utter.keys(): user_transcripts.append(utter['User'])

            # if args.lang == 'zh':
            #     user_transcripts = GoogleTranslator(source='zh-TW', target='en').translate_batch(user_transcripts)

            save_file( _socialREMChat.context)

        else:
            last = False
            response, cot_response = _socialREMChat.chatting(context=_socialREMChat.context)
            _socialREMChat.context.append({'Assistant': response})

            print('[Chain-of-Thought]: ')
            for output_step in cot_response:
                print(output_step)

        return json.dumps({"return_message": response, "last": last})
    else:
        return json.dumps({"return_message": 'Invalid request method'})


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