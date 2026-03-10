# -*- coding:utf-8 -*-
import requests
import time
import json

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from deep_translator import GoogleTranslator

trans_zh_en = GoogleTranslator(source='zh-CN', target='en')
trans_en_zh = GoogleTranslator(source='en', target='zh-CN')


class BlenderBot:
    def __init__(self):
        model_name = "facebook/blenderbot-400M-distill"
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

    def generateReply(self, utterance):
        # utterance = "My friends are cool, but they eat too many carbs."
        utterance = trans_zh_en.translate(utterance)
        inputs = self.tokenizer([utterance], return_tensors="pt")
        reply_ids = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(reply_ids)[0].replace("</s>", '').replace("<s>", '')
        response = trans_en_zh.translate(response)
        return response


def send_post_message(msg_, url):
    post_msg = msg_
    post_msg["post_time"] = str(time.time())  # 定義發送信息

    headers = {"Content-Type": "application/json"}  # 定義請求頭
    _post_data = json.dumps(post_msg, sort_keys=True, separators=(',', ':'))  # 將信息打包為json格式
    req = requests.post(url, data=_post_data, headers=headers)  # 使用requests，以POST形式發送信息
    if req.status_code == requests.codes.ok:  # 如果請求正常並收到回復
        # print('Sending ok')
        res = req.json()  # 讀取回復
        # print("get response: {}".format(result))
        return res
    else:
        print('Sending fail')


if __name__ == '__main__':
    bl = BlenderBot()

    msg = {'reset_history': True}
    result: object = send_post_message(msg, 'http://127.0.0.1:9110/')
    # print('sending over')
    if 'reset_done' in result and result['reset_done']:
        print("==== reset image and history ==== ==== ==== ==== ==== ==== ==== ")

    while True:
        # reset metadata
        event_tag = input("EVENT: \t\t")
        place_tag = input("PLACE: \t\t")
        relation_tag = input("RELATION: \t")

        # send metadata, request a question
        # msg = {"metadata": "{place={confidence=0.4, label=restaurant_table}, event={confidence=1.0, "
        #                    "label=graduation}, relationship={confidence=1.0, label=family}}"}
        msg_meta = {"metadata": "{place={confidence=0.4, label=" + place_tag
                                + "}, event={confidence=1.0, label=" + event_tag
                                + "}, relationship={confidence=1.0, label=" + relation_tag + "}}"}
        result = send_post_message(msg_meta, 'http://127.0.0.1:9110/')
        # print('sending over')
        if 'question_question' in result and result['question_question']:
            print("生成Robot問題: \t{}".format(result['question_question']))

        # # get user input, request first response
        # input_str1 = input("輸入User訊息: \t")
        # reply = bl.generateReply(input_str1)
        # print("生成Robot訊息: \t{}".format(reply))
        # msg = {'robot_reply': reply, 'user_utterance': input_str1}
        # result = send_post_message(msg, 'http://127.0.0.1:9110/')
        # # print('sending over')

        while True:
            # get user input, decide request type
            input_str2 = input("輸入User訊息: \t")

            if input_str2 == 'reset_history':
                msg = {'reset_history': True}
                result = send_post_message(msg, 'http://127.0.0.1:9110/')
                # print('sending over')
                if result['reset_done']:
                    print("==== reset image and history ==== ==== ==== ==== ==== ==== ==== ")
                    break

            if input_str2 == 'request_question':
                result = send_post_message(msg_meta, 'http://127.0.0.1:9110/')
                # print('sending over')
                if 'question_question' in result and result['question_question']:
                    print("生成Robot問題: \t{}".format(result['question_question']))
                continue

            else:
                # request a response
                reply2 = bl.generateReply(input_str2)
                msg = {"check_response": reply2}
                result = send_post_message(msg, 'http://127.0.0.1:9110/')
                # print('sending over')

                if 'tag' in result and not result['tag']:
                    print("生成Robot訊息: \t{}".format(reply2))
                else:
                    print("\tRobot語句重複: \t{}".format(reply2))
                    reply2 = "好喔，我了解了"
                    print("生成Robot訊息: \t{}".format(reply2))

                msg = {'robot_reply': reply2, 'user_utterance': input_str2}
                send_post_message(msg, 'http://127.0.0.1:9110/')
                # print('sending over')

                reply = reply2
