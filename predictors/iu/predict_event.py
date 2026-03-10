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
import collections

import model_vgg16_reminiscence as VGG16

class ImageUnderstanding():
    """ Deep learning based image understanding
    """
    def __init__(self, args):
        #Don't show TF verbose
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        path = os.path.dirname(os.path.abspath(__file__))

        self.weights_path = os.path.join(path, 'weights')
        self.labels_path = os.path.join(path, 'labels')
        
        #uploads_path = os.path.join(Path(path).parent, 'uploads')
        uploads_path = '/opt/lampp/htdocs/src/uploads'

        self.image_path = os.path.join(uploads_path, args.img_id)
        self.learn()
    
    def learn(self):
        model_events, headers_events = self.init_model('event', 'data')
        self.predict('event', model_events, headers_events)
        
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
        print(json.dumps(result))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_id', type=str)
    args = parser.parse_args()
    imageUnderstanding = ImageUnderstanding(args)