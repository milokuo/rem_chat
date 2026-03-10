# -*- coding:utf-8 -*-
import json
import time
import requests
from config import parse_args

def send_post_message(msg, url):
    headers = {'Content-Type': 'application/json'}

    post_msg = {'user_message': str(msg)}
    _post_data = json.dumps(post_msg, sort_keys=True, separators=(',', ':'))

    try:
        response = requests.post(f'{url}/', data=_post_data, headers=headers)
        print(response)
        result = response.json()

        print('Assistant:', result['return_message'].strip())
    except requests.exceptions.RequestException as e:
        print('[Error]:', e)

if __name__ == '__main__':
    args = parse_args()

    if args.lang == 'zh': print('Assistant: 你好，關於這張照片有什麼事情想跟我聊聊嗎？')
    else: print('Assistant: Hello, Is there anything you want to talk to me about this photograph?')

    while True:
        input_str = input("User: ")
        # send_post_message(input_str, 'http://140.112.95.5:9100/dep_sup')
        send_post_message(input_str, 'http://0.0.0.0:8089')
