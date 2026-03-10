# -*- coding:utf-8 -*-
import requests
import time
import json
import argparse
import web
import datetime
import time
import os

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
        return res
    else:
        print('Sending fail')
        return {}

class web_server_iu_redirector:
    def __init__(self):
        now = datetime.datetime.now()
        print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))

    def POST(self):
        receive = json.loads(str(web.data(), encoding='utf-8'))

        if 'img_id' in receive and receive['img_id']:
            # pending_list = '/home/penguin37/companion/rem_chat/predictors/pending_list.txt'
            
            # if not os.path.exists(process_list):
            #     with open(process_list, 'w') as f:
            #         f.write(receive['img_id'] + ';')
            
            # else:
            #     with open(process_list, 'r') as record:
            #         for line in record.readlines():
            #             tmp = line.split(';')
            #             if receive['img_id'] in tmp:
            #                 if tmp[1] != "":
            #                     # read saved metadata
            #                     data = json.loads(tmp[1])
            #                     # if receive['cate'] in data:
            #                         # split_data = {receive['cate']: data[receive['cate']]}
            #                     print()
            #                     print("{} (get saved result)".format(receive['cate']))
            #                     print()
            #                     print(json.dumps(data))
            #                     print()
            #                     return json.dumps(data)
            #                 else:
            #                     # wait for previous process to finish
            #                     data = {"status": "running"}
            #                     return json.dumps(data)
            #     # no record found
            #     with open(process_list, 'a') as f:
            #         f.write(receive['img_id'] + ';')

            start = time.time()

            if 'cate' in receive and receive['cate']:
                msg = {'img_id': receive['img_id'], 'cate': receive['cate']}

                if receive['cate'] in ['event', 'place']:
                    return_json = send_post_message(msg, 'http://127.0.0.1:9201/')
                elif receive['cate'] == 'people':
                    return_json = send_post_message(msg, 'http://127.0.0.1:9202/')
                elif receive['cate'] == 'clip':
                    return_json = send_post_message(msg, 'http://127.0.0.1:9205/')
                    # return_json = send_post_message(msg, 'http://192.168.1.105:9205/')
                else:
                    print('wrong cate type!')
                    return {}

            print()
            print("predict cost time: {}".format(time.time()-start))
            print()
            
            print()
            print(receive['cate'])
            print(json.dumps(return_json))
            print()

            # with open(process_list, 'a') as f:
            #     f.write(json.dumps(return_json) + '\n')

            return_data = json.dumps(return_json)
            # return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            print(return_data)
            return return_data
        else:
            return {}


if __name__ == '__main__':
    print("Initialized IU Redirector\n -> 9201: event and place model\n -> 9202: people model\n -> 9205: clip model\n")

    URL_facereg_main = ("/", "web_server_iu_redirector")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())

    app.run()



