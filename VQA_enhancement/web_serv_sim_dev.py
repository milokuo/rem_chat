# -*- coding:utf-8 -*-
import datetime
import time
import requests
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import web

from QuestionItem import QuestionSet
from GptGenerator import GptGenerator


def send_post_message(msg_, url):
    post_msg = msg_
    post_msg["post_time"] = str(time.time())  # 定義發送信息

    headers = {"Content-Type": "application/json"}  # 定義請求頭
    _post_data = json.dumps(post_msg, sort_keys=True, separators=(',', ':'))  # 將信息打包為json格式
    req = requests.post(url, data=_post_data, headers=headers)  # 使用requests，以POST形式發送信息
    if req.status_code == requests.codes.ok:  # 如果請求正常並收到回復
        # print('Sending ok')
        res = req.json()  # 讀取回復
        # print(json.dumps(res))
        return res
    else:
        print('Sending fail')
        return {}

class SimTool:
    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        #cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        #tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')

        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        self.SCORE_THRESH = 0.6

        self.user_utterance = []
        self.robot_reply = []

        self.qSet = QuestionSet("./v3")
        self.metadata = {}
        self.filtered_questions = []
        self.df_filtered = None

        self.first_question_pass = False
        self.use_gpt = True
        if self.use_gpt:
            self.gpt_generator = GptGenerator()

    def generate_question(self):
        _sys = f"You are a companion robot chatting with the user, concerning the content of an image. Now we konw the metadata of the image is as following in json format: \n{self.metadata} \n. \nGiven the dialogue history, please provide the next quesion. The question should follow the topic flow of the dialogue, and should not be meaningless or redundant. I will show your two examples. \nmeaningless: asking about the weather if the event tag of image is an outdoor activity requiring a good weather absolutely. \n redundant: asking if the image is taken at some place abroad if the user already told you he went to a certain contry or city at that time.\n To easily start the conversation, asking relevant question to the event of the image is more acceptable. You should use Chinese to generate response.\n"
        _len = len(self.robot_reply) if len(self.user_utterance) > len(self.robot_reply) else len(self.user_utterance)
        _history = ""
        for i in range(_len):
            _history += f"user: {self.user_utterance[i]}\n"
            _history += f"robot: {self.robot_reply[i]}\n\n"
        if len(self.user_utterance) > len(self.robot_reply):
            _history += f"user: {self.user_utterance[_len-1]}\n"
        _query = "robot: "
        self.gpt_generator.reset_system(_sys)
        self.gpt_generator.update_user(_history + _query)
        _question = self.gpt_generator.get_response_turbo()
        return _question
    
    def update_metadata(self, _key, _value):
        self.metadata[_key] = _value
    
    def request_more_metadata(self, filename):
        for _cate in ['detr', 'caption']:
            _msg = {'img_id': filename, 'cate': _cate}
            if _cate == 'detr':
                return_json = send_post_message(_msg, 'http://127.0.0.1:9206/')
                self.update_metadata('objects', return_json['objects'])
            elif _cate == 'caption':
                return_json = send_post_message(_msg, 'http://127.0.0.1:9207/')
                self.update_metadata('caption', return_json['caption'])

    def filter_use_metadata(self, metadata):
        # reset
        self.metadata = {}
        self.filtered_questions = []

        # parse metadata
        # an example as follows
        # metadata = "{place={confidence=0.4, label=restaurant_table}, event={confidence=1.0, label=graduation}, " \
        #            "relationship={confidence=1.0, label=family}} "
        p = sorted([metadata.find('place'), metadata.find('event'), metadata.find('relationship')])
        substrings = [metadata[p[0]:p[1]], metadata[p[1]:p[2]], metadata[p[2]:]]
        keys = ['event', 'place', 'relationship']
        for string in substrings:
            for key in keys:
                if string.find(key) != -1:
                    ps = [string.find('confidence'), string.find('label')]
                    self.metadata[key] = {}
                    self.metadata[key]['confidence'] = float(string[ps[0] + 11:ps[1]].replace(',', ''))
                    self.metadata[key]['label'] = string[ps[1] + 6:].replace('}', '').replace(',', '').replace(' ', '')
        print("get tag:\n {}".format(self.metadata))

        # select q set w.r.t. metadata to get three lists
        self.qSet.get_questions(self.metadata)

        # sort w.r.t. confidence order
        self.filtered_questions = self.qSet.qPlace + self.qSet.qEvent + self.qSet.qRelation
        filtered_temp = []
        for qItem in self.filtered_questions:
            temp = {'id': qItem.getIdx(),
                    'category': qItem.getCategory(),
                    'label': qItem.getLabel(),
                    'question': qItem.getQuestion(),
                    'similar': qItem.getSimilar(),
                    'confidence': qItem.getConfidence(),
                    'embedding': qItem.getEmbedding()
                    }
            filtered_temp.append(temp)
        self.df_filtered = pd.DataFrame(filtered_temp).sort_values(by=['confidence'], ascending=False)
        print("sorted selected questions by tag with confidence:\n {}".format(self.df_filtered))
        print(">>> df_filtered generated! total #{}".format(self.df_filtered.shape[0]))

    def filter_use_static_similarity(self, sim_ids):
        print(sim_ids)
        for idx in sim_ids:
            if idx in self.df_filtered['id']:
                temp = self.df_filtered
                self.df_filtered = temp.drop(temp[temp.id == idx].index)
        # print("\n {}".format(self.df_filtered))
        print(">>> df_filtered updated!(s1) total #{}".format(self.df_filtered.shape[0]))

    def remove_a_question(self, selected):
        # remove high similar question from set
        sim_ids = selected['similar']
        self.filter_use_static_similarity(sim_ids)

        temp = self.df_filtered
        self.df_filtered = temp.drop(temp[temp.id == selected['id']].index)

        # print("\n {}".format(self.df_filtered))
        print(">>> df_filtered updated!(s2) total #{}".format(self.df_filtered.shape[0]))

    def reset_history(self):
        self.robot_reply = []
        self.user_utterance = []
        print("reset history")

    def update_history(self, ut, re):
        if ut != "":
            self.user_utterance.append(ut)
        if re != "":
            self.robot_reply.append(re)
        print("update history:\n {} \n\n {} \n".format(self.robot_reply, self.user_utterance))

    def select_a_question(self):
        tag = True
        selected = None
        idx = -1
        question = ""

        while tag and self.df_filtered.shape[0] > 0:
            # resample
            seed = random.randint(0, self.df_filtered.shape[0] - 1)
            selected = self.df_filtered.iloc[int(seed)]
            # choose first question from event
            while self.first_question_pass and selected['category'] != 'event':
                seed = random.randint(0, self.df_filtered.shape[0] - 1)
                selected = self.df_filtered.iloc[int(seed)]
            # check similarity w.r.t. history
            print("check question:")
            score, tag = self.compute_sim_with_mem(self.robot_reply, selected['question'])
            if tag:
                print("duplicated question!")
                self.remove_a_question(selected)
            else:
                idx = selected['id']
                question = selected['question']
                self.first_question_pass = True

        if idx != -1 and question != "":
            print("\n>>> select #{} question: \n\n\t\t{}\n\n".format(idx, question))
            self.remove_a_question(selected)
            self.update_history("", selected['question'])
            return idx, question

        else:
            print("None question can be selected!!!")
            return -1, "我们换张照片聊聊吧"

    def compute_sim(self, s1, s2):
        em_s1 = self.embed(s1)
        em_s2 = self.embed(s2)
        similarity_matrix = np.inner(em_s1, em_s2)
        print(">>> sim shape: {},\n {}".format(similarity_matrix.shape, similarity_matrix))
        sim_score = similarity_matrix[0][0]
        if sim_score > self.SCORE_THRESH:
            tag = True
        else:
            tag = False
        print(sim_score, tag)
        return sim_score, tag

    def compute_sim_with_mem(self, history, u):
        if len(history) == 0:
            return -1, False

        v_seq_hist = self.embed(history)
        v_u = self.embed(u)
        similarity_matrix = np.inner(v_u, v_seq_hist)
        print(">>> sim shape: {},\n {}".format(similarity_matrix.shape, similarity_matrix))
        sim_score = similarity_matrix[0]
        for each in sim_score:
            if each > self.SCORE_THRESH:
                return sim_score, True
        return sim_score, False


class web_server_sim:
    global sim

    def __init__(self):
        now = datetime.datetime.now()
        print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))

    def POST(self):
        receive = json.loads(str(web.data(), encoding='utf-8'))

        if 'reset_history' in receive and receive['reset_history']:
            sim.reset_history()
            return_json = {'reset_done': True}
            return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data

        if 'update_metadata' in receive and receive['update_metadata']:
            # if 'caption' in receive['metadata']:
            #     sim.update_metadata('caption', receive['metadata']['caption'])
            #     return {}
            # if 'objects' in receive['metadata']:
            #     sim.update_metadata('objects', receive['metadata']['objects'])
            #     return {}
            sim.request_more_metadata(receive['image_name_regiven'])
            return_json = {'update_metadata': True}
            return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data
        
        if 'metadata' in receive and receive['metadata']:
            print("receive metadata:\n {}".format(receive['metadata']))
            if len(sim.robot_reply) == 0:
                # init Q set for opening an image
                sim.filter_use_metadata(receive['metadata'])
            # select an Q and update Q set
            if not sim.use_gpt:
                idx, question = sim.select_a_question()
            else:
                idx = "gpt"
                question = sim.generate_question()
                print("\n>>> select #{} question: \n\n\t\t{}\n\n".format(idx, question))

            return_json = {"question_question_ready": True,
                           "question_question_id": str(idx),
                           "question_question": question}
            return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data

        if 'check_response' in receive and receive['check_response']:
            print("check response:")
            sim_score, tag = sim.compute_sim_with_mem(sim.robot_reply, receive["check_response"])
            return_json = {"similarity": str(sim_score), "tag": tag}
            return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data

        if 'user_utterance' in receive and receive['user_utterance']:
            print("User: \t{}".format(receive['user_utterance']))
            sim.update_history(receive['user_utterance'], "")

        if 'robot_reply' in receive and receive['robot_reply']:
            print("Robot: \t{}".format(receive['robot_reply']))
            sim.update_history("", receive['robot_reply'])

        return_json = {"update_history": True}
        return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return return_data


if __name__ == '__main__':
    web.config.debug = False

    sim = SimTool()
    print("Initialized SimTool model")

    URL_facereg_main = ("/", "web_server_sim")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())
    
    session = web.session.Session(app, web.session.DiskStore("sessions"), initializer={"tag": "", "options": [], "yes_no_question": False, "go_back": False})

    app.run()
