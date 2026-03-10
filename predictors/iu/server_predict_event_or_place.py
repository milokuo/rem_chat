#!/usr/bin/python3

import os
import argparse
import numpy as np
from PIL import Image
from cv2 import resize
from pathlib import Path
from glob import iglob
import json
import operator
import web
import datetime
import time

import model_vgg16_reminiscence as VGG16

class ImageUnderstanding():
    """ Deep learning based image understanding
    """
    def __init__(self):
        #Don't show TF verbose
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        path = os.path.dirname(os.path.abspath(__file__))

        self.weights_path = os.path.join(path, 'weights')
        self.labels_path = os.path.join(path, 'labels')
        
        #uploads_path = os.path.join(Path(path).parent, 'uploads')
        self.uploads_path = '/opt/lampp/htdocs/src/uploads'
    
    def learn(self, img_id, category):
        self.image_path = os.path.join(self.uploads_path, img_id)

        if category == 'event':
            self.model_events, self.headers_events = self.init_model('event', 'data')
            return self.predict('event', self.model_events, self.headers_events)

        elif category == 'place':
            self.model_places, self.headers_places = self.init_model('place', 'data')
            return self.predict('place', self.model_places, self.headers_places)
        
    def init_model(self, category, file_type):
        labels_file = os.path.join(self.labels_path, "image_labels.json")
        with open(labels_file) as label_file:
            data = json.load(label_file)
            headers = list(data[category].keys())

        # Define Model & load weights. Weights must be a previously saved model.
        weights_file = os.path.join(self.weights_path, category + '.h5')
        if os.path.exists(weights_file):
            model = VGG16.VGG16_Places365(weights=weights_file, classes=len(headers))
        else:
            print('Model not found for ' + category)
            model = None

        return model, headers

    def predict(self, category, model, headers):
        print(self.image_path)
        image = Image.open(self.image_path)
        image = np.array(image, dtype=np.uint8)
        image = resize(image, (224, 224))
        image = np.expand_dims(image, 0)
        image = 1./255*image

        # Predict the place and event in the images.
        if model is not None:
            preds = model.predict(image)[0]
        # For now if no model we create empty dummy data.
        else:
            preds = [''] * len(headers)

        output = {}

        for i in range(len(preds)):
            output[headers[i]] = float(str('%.1f' % preds[i]))
        
        top = max(output.items(), key=operator.itemgetter(1))
        
        result = {}
        result["label"] = top[0]
        result["confidence"] = top[1]
        # print("{}: {}".format(category, json.dumps(result)))

        return result
        
class web_server_iu:
    global iu

    def __init__(self):
        now = datetime.datetime.now()
        print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))

    def POST(self):
        receive = json.loads(str(web.data(), encoding='utf-8'))

        if 'img_id' in receive and receive['img_id']:

            process_list = '/home/penguin37/companion/rem_chat/predictors/process_list_{}.txt'.format(receive['cate'])
            if not os.path.exists(process_list):
                with open(process_list, 'w') as f:
                    f.write(receive['img_id'] + ';')
            else:
                with open(process_list, 'r') as record:
                    for line in record.readlines():
                        tmp = line.split(';')
                        if receive['img_id'] in tmp:
                            if tmp[1] != "":
                                # read saved metadata
                                data = json.loads(tmp[1])
                                # if receive['cate'] in data:
                                    # split_data = {receive['cate']: data[receive['cate']]}
                                print()
                                print("{} (get saved result)".format(receive['cate']))
                                print()
                                print(json.dumps(data))
                                print()
                                return json.dumps(data)
                            else:
                                # wait for previous process to finish
                                data = {"status": "running"}
                                return json.dumps(data)
                # no record found
                with open(process_list, 'a') as f:
                    f.write(receive['img_id'] + ';')

            start = time.time()
            return_json = iu.learn(receive['img_id'], receive['cate'])

            print()
            print("predict cost time: {}".format(time.time()-start))
            print()
            
            print()
            print(receive['cate'])
            print(json.dumps(return_json))
            print()

            with open(process_list, 'a') as f:
                f.write(json.dumps(return_json) + '\n')

            return_data = json.dumps(return_json)
            # return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data
        else:
            return {}
            

if __name__ == '__main__':
    iu = ImageUnderstanding()
    print("Initialized IU event and place model")

    URL_facereg_main = ("/", "web_server_iu")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())

    app.run()
