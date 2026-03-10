# -*- coding:utf-8 -*-
import requests
import time
import json
import argparse

def send_post_message(msg_, url):
    post_msg = msg_
    post_msg["post_time"] = str(time.time())  # 定義發送信息

    headers = {"Content-Type": "application/json"}  # 定義請求頭
    _post_data = json.dumps(post_msg, sort_keys=True, separators=(',', ':'))  # 將信息打包為json格式
    req = requests.post(url, data=_post_data, headers=headers)  # 使用requests，以POST形式發送信息
    if req.status_code == requests.codes.ok:  # 如果請求正常並收到回復
        # print('Sending ok')
        res = req.json()  # 讀取回復
        print(json.dumps(res))
        # return res
    else:
        print('Sending fail')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_id', type=str)
    parser.add_argument('--cate', type=str)
    args = parser.parse_args()
    
    msg = {'img_id': args.img_id, 'cate': args.cate}

    if args.cate in ['event', 'place']:
        send_post_message(msg, 'http://127.0.0.1:9201/')
    elif args.cate == 'people':
        send_post_message(msg, 'http://127.0.0.1:9202/')
    elif args.cate == 'clip':
        send_post_message(msg, 'http://127.0.0.1:9205/')
        # send_post_message(msg, 'http://192.168.1.105:9205/')
    else:
        print('wrong cate type!')

