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

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import pipeline
from collections import Counter

class DETR_detector:
    global receive

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        # The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.

    def predict(self, url) -> str:
        if os.path.isfile(url):
            print(f'found {url}!')
            image = Image.open(url)
        else:
            raise FileNotFoundError(f'Image not found at local path: {url}')

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        #     break

        objects = [self.model.config.id2label[label.item()] for label in results["labels"]]
        #print(objects)
        counts = Counter(objects)
        result = " and ".join(f'{count} {name}s' for name, count in counts.items())

        return result

class web_server_detr_obj:
    global _model
    global receive

    def __init__(self):
        now = datetime.datetime.now()
        print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))            

    def POST(self):
        receive = json.loads(str(web.data(), encoding='utf-8'))

        if 'img_id' in receive and receive['img_id']:
            # url = os.path.join(img_dir, 'test.jpg') # for DEBUG
            url = os.path.join(img_dir, receive['img_id']) # for test

            obj_string = _model.predict(url)

            metadata = {}
            metadata['objects'] = obj_string
            print(f'>>> METADATA: {metadata}\n\n\n')

            return_data = json.dumps(metadata)
            # return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data
        else:
            return {}


if __name__ == '__main__':
    receive = {}
    _model = DETR_detector()
    print("Initialized DETR_detector model")

    URL_facereg_main = ("/", "web_server_detr_obj")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())

    app.run()






