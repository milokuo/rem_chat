# -*- coding:utf-8 -*-
from PIL import Image
import requests
import web
import json
import datetime
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/penguin37/companion/rem_chat/predictors/clip_iu/models'
# 改為本地路徑：server_updated_zhengxuan.py 旁的 uploads/ 資料夾
_SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ParlAI', 'projects', 'image_chat', 'uploads')
img_dir = os.path.realpath(_SERVER_DIR) + os.sep
label_file = '../../VQA_enhancement/v3/image_labels.json'

from transformers import CLIPProcessor, CLIPModel
# from transformers import AltCLIPModel, AltCLIPProcessor

class ClipPredictor:
    global receive
    
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # self.model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        # self.processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

        self.load()
    
    def load(self):
        with open(label_file, 'r') as f:
            labels = json.load(f)
            print(labels)

        self.event_labels = list(labels['event'].keys())
        self.event_candidates = [f'a photo of {label}' for label in self.event_labels]

        self.place_labels = list(labels['place'].keys())
        self.place_candidates = [f'a photo taken at the place of {label}' for label in self.place_labels]

        self.relation_labels = list(labels['relationship'].keys())

        print('labels loaded!')
    
    def predict(self, url, text_candidates, labels):
        if os.path.isfile(url):
            print(f'found {url}!')
            image = Image.open(url)
        else:
            raise FileNotFoundError(f'Image not found at local path: {url}')

        inputs = self.processor(text=text_candidates, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        # print(probs)
        _probs = probs[0].tolist()
        _max = -1
        _max_idx = -1
        for _index, _prob in enumerate(_probs):
            if _prob > _max:
                _max = _prob
                _max_idx = _index
        print(f'output:  {labels[_max_idx]} ---- {text_candidates[_max_idx]}\n\n')
        return text_candidates[_max_idx], labels[_max_idx], float("{:.4f}".format(_probs[_max_idx]))

class web_server_clip_iu:
    global cp
    global receive

    def __init__(self):
        now = datetime.datetime.now()
        print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))            

    def POST(self):
        receive = json.loads(str(web.data(), encoding='utf-8'))

        if 'img_id' in receive and receive['img_id']:
            # url = os.path.join(img_dir, 'test.jpg') # for DEBUG
            url = os.path.join(img_dir, receive['img_id']) # for test

            _, event_pred, event_prob = cp.predict(url, cp.event_candidates, cp.event_labels)
            _, place_pred, place_prob = cp.predict(url, cp.place_candidates, cp.place_labels)

            cp.relation_candidates = [f'the people were attending a {event_pred} at the place of {place_pred}, we can infer their relationship are {label}' for label in cp.relation_labels]
            _, relation_pred, relation_prob = cp.predict(url, cp.relation_candidates, cp.relation_labels)
            
            event_prob = f"{event_prob:.3f}"
            place_prob = f"{place_prob:.3f}"
            relation_prob = f"{relation_prob:.3f}"

            metadata = {}
            # metadata['filename'] = receive['img_id']
            metadata['event'] = {'label': event_pred, 'confidence': event_prob}
            metadata['place'] = {'label': place_pred, 'confidence': place_prob}
            metadata['relationship'] = {'label': relation_pred, 'confidence': relation_prob}
            print(f'>>> METADATA: {metadata}\n\n\n')

            return_data = json.dumps(metadata)
            # return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data
        else:
            return {}


if __name__ == '__main__':
    receive = {}
    cp = ClipPredictor()
    print("Initialized CLIP-IU model")

    URL_facereg_main = ("/", "web_server_clip_iu")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())

    app.run()