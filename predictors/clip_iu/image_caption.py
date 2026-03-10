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
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaption:
    global receive

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")
        self.condition_set = False
        # max_length -> max_new_tokens

    def predict(self, url) -> str:
        if os.path.isfile(url):
            print(f'found {url}!')
            image = Image.open(url)
        else:
            raise FileNotFoundError(f'Image not found at local path: {url}')

        #print(type(image))

        if self.condition_set:
            # conditional image captioning
            text = "a photography of"
            inputs = self.processor(image, text, return_tensors="pt").to("cuda", torch.float16)

            out = self.model.generate(**inputs)
            caption_str = self.processor.decode(out[0], skip_special_tokens=True)
        
        else:
            # unconditional image captioning
            inputs = self.processor(image, return_tensors="pt").to("cuda", torch.float16)

            out = self.model.generate(**inputs)
            caption_str = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption_str

class web_server_blip_caption:
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

            _string = _model.predict(url)

            metadata = {}
            metadata['caption'] = _string
            print(f'>>> METADATA: {metadata}\n\n\n')

            return_data = json.dumps(metadata)
            # return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data
        else:
            return {}


if __name__ == '__main__':
    receive = {}
    _model = ImageCaption()
    print("Initialized BLIP_caption model")

    URL_facereg_main = ("/", "web_server_blip_caption")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())

    app.run()






