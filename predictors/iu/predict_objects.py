#!/usr/bin/python3

import os
import cgi
from PIL import Image
from glob import iglob
import json

from keras_yolo.yolo import YOLO

class ImageUnderstanding():
    """ Deep learning based image understanding
    """
    def __init__(self, args):
        #Don't show TF verbose
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.object_detection_threshold = 0.7
        
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(self.path, 'weights')
        self.labels_path = os.path.join(self.path, 'labels')
        self.anchors_path = os.path.join(self.path, 'anchors')
        self.image_path = os.path.join(self.path, args['img_id'])
        self.output_path = self.path
    
        self.predict()
 
    def predict(self):
        # Create dictionary for de data to save to a later file.
        output = {}
        image = Image.open(self.image_path)
        yolo = YOLO(self.weights_path, self.anchors_path, self.labels_path)
        objects, image = yolo.detect_image(image)

        # The image_name key should have been added in the events_places stage
        output['objects'] = {}

        # Save the predictions for YOLO objects
        for object_meta in objects:
            if object_meta[1] >= self.object_detection_threshold:
                object_name = object_meta[0]
                if object_name in output['objects']:
                    output['objects'][object_name]['count'] += 1
                    output['objects'][object_name]['location'].append(object_meta[2])
                else:
                    output['objects'][object_name] = {}
                    output['objects'][object_name]['count'] = 0
                    output['objects'][object_name]['location'] = []
                    
        yolo.close_session()
        print(json.dumps(output))
        

def cgiFieldStorageToDict(fieldStorage):
    params = {}
    for key in fieldStorage.keys(  ):
        params[key] = fieldStorage[key].value
    return params

if __name__ == '__main__':
    print("content-type: text/html\n\n" ) 
    args_dict =  cgiFieldStorageToDict(cgi.FieldStorage())
    #imageUnderstanding = ImageUnderstanding(args_dict)