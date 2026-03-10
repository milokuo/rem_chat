import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.disable_eager_execution()

import numpy as np
import os
import itertools
import json
from decimal import Decimal
from pathlib import Path
import web
import datetime
import time
from predict_people_copy import *

class web_server_iu2:

    def __init__(self):
        now = datetime.datetime.now()
        print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))

      
    def POST(self):
        receive = json.loads(str(web.data(), encoding='utf-8'))
        img_id = receive['img_id']
        return predictRelation(FLAGS, ROOTPATH ,img_id)


        if 'img_id' in receive and receive['img_id']:
            process_list = '/home/penguin37/companion/rem_chat/predictors/process_list_relation.txt'
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
                                print()
                                print("relationship (get saved result)")
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

            face_files, filenames = self.detector.run(os.path.join(self.uploads_path, receive['img_id']))
            image_files = list(filter(lambda x: x is not None, [find_files(f) for f in face_files]))

            index = 0
            ages = []
            genders = []
            locations = []
            
            name_list = receive['img_id'].split('/')
            name_list = name_list[len(name_list) - 1].split('.')
            basename = name_list[len(name_list) - 2]

            if len(face_files) != 0:
                for location in self.detector.locations:
                    loc = {}
                    loc["x"] = round(location[0],0)
                    loc["y"] = round(location[1],0)
                    loc["w"] = round(location[2],0)
                    loc["h"] = round(location[3],0)
                    locations.append(loc)

                
                ages, age_seconds = [], []
                age_top1_probs, age_top2_probs = [], []
                genders = []
                gender_top1_probs = []
                ratios = np.around(self.detector.ratios, decimals=3)

                # # age prediction
                # for f in image_files:
                #     best, best_prob, second, second_prob = self.classify_one_multi_crop(self.age_session, AGE_LIST, softmax_output_age, self.images_age, os.path.join(self.output_path,f), coder)  
                #     ages.append(best)
                #     age_top1_probs.append(best_prob)
                #     age_seconds.append(second)
                #     age_top2_probs.append(second_prob)
                    
                    
                #     # gender prediction
                #     best, best_prob, second_best, second_prob = self.classify_one_multi_crop(self.gender_session, GENDER_LIST, softmax_output_gender, self.images_gender, os.path.join(self.output_path,f), coder)      
                #     genders.append(best)  
                #     gender_top1_probs.append(best_prob)
                    

                #     index += 1

                # age prediction
                age_graph = tf.Graph()
                with tf.compat.v1.Session(config=config, graph=age_graph) as sess:

                    with tf.device(FLAGS.device_id):
                        
                        label_list = AGE_LIST
                        nlabels = len(label_list)

                        images = tf.compat.v1.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                        logits = levi_hassner_bn(nlabels, images, 1, False)
                        init = tf.compat.v1.global_variables_initializer()

                        model_checkpoint_path, _ = get_checkpoint(FLAGS.age_model_dir, None, 'checkpoint')

                        saver = tf.compat.v1.train.Saver()
                        saver.restore(sess, model_checkpoint_path)

                        softmax_output = tf.nn.softmax(logits)

                        image_files = list(filter(lambda x: x is not None, [find_files(f) for f in face_files]))

                        for f in image_files:
                            best, best_prob, second, second_prob = self.classify_one_multi_crop(sess, label_list, softmax_output, images, os.path.join(self.output_path,f), coder)  
                            ages.append(best)
                            age_top1_probs.append(best_prob)
                            age_seconds.append(second)
                            age_top2_probs.append(second_prob)

                            index += 1

                # gender prediction
                gender_graph = tf.Graph()
                with tf.compat.v1.Session(config=config, graph=gender_graph) as sess:

                    with tf.device(FLAGS.device_id):

                        label_list = GENDER_LIST
                        nlabels = len(label_list)

                        images = tf.compat.v1.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                        logits = levi_hassner_bn(nlabels, images, 1, False)
                        init = tf.compat.v1.global_variables_initializer()

                        model_checkpoint_path, _ = get_checkpoint(FLAGS.gender_model_dir, None, 'checkpoint')

                        saver = tf.compat.v1.train.Saver()
                        saver.restore(sess, model_checkpoint_path)

                        softmax_output = tf.nn.softmax(logits)

                        for f in image_files:
                            best, best_prob, second_best, second_prob = self.classify_one_multi_crop(sess, label_list, softmax_output, images, os.path.join(self.output_path,f), coder)      
                            genders.append(best)  
                            gender_top1_probs.append(best_prob)

                data = People(filenames, ages, genders, locations).getOutput()

                relationshipPredictor = RelationshipPredictor(data['people'])
                data['relationship'] = relationshipPredictor.predict()
                # data['status'] = "ready"

                #Move output faces to folder
                import shutil
                file_names = os.listdir(self.output_path)
                out_faces = os.path.join(self.uploads_path, 'faces')
                
                if not os.path.isdir(out_faces):
                    os.mkdir(out_faces)

                for file_name in file_names:
                    filepath = os.path.join(out_faces, file_name)
                    if os.path.exists(filepath):
                        os.remove(os.path.join(self.output_path, file_name))
                    else:
                        shutil.move(os.path.join(self.output_path, file_name), out_faces)
            else:
                data = {}
                data['people'] = {}
                relationshipPredictor = RelationshipPredictor(data['people'])
                data['relationship'] = relationshipPredictor.predict()
                # data['status'] = "ready"

            print()
            print("predict cost time: {}".format(time.time()-start))
            print()

            print()
            print('relationship')
            print()
            print(json.dumps(data))
            print()

            with open(process_list, 'a') as f:
                f.write(json.dumps(data) + '\n')
            
            return_data = json.dumps(data)
            # return_data = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            return return_data
        
        else:
            return {}
            


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


if __name__ == '__main__':
    del_all_flags(tf.compat.v1.app.flags.FLAGS)

    tf.compat.v1.disable_eager_execution()

    ROOTPATH = os.path.dirname(os.path.abspath(__file__))
    FLAGS = tf.compat.v1.app.flags.FLAGS
    tf.compat.v1.app.flags.DEFINE_string('gender_model_dir', os.path.join(ROOTPATH,'weights/gender'), 'Gender weights')
    tf.compat.v1.app.flags.DEFINE_string('age_model_dir', os.path.join(ROOTPATH,'weights/age'), 'Age weights')
    tf.compat.v1.app.flags.DEFINE_string('output_path', os.path.join(ROOTPATH,'output'),'output_path')
    tf.compat.v1.app.flags.DEFINE_string('device_id', '/gpu:0', 'What processing unit to execute inference on')
    tf.compat.v1.app.flags.DEFINE_string('process_list', '/home/penguin37/companion/rem_chat/predictors/process_list_relation.txt', 'porcess lists path')
    tf.compat.v1.app.flags.DEFINE_string('uploads_path', '/opt/lampp/htdocs/src/uploads', 'uploads path')
    tf.compat.v1.app.flags.DEFINE_float('gpu_memory_ratio', 0.3, 'gpu memory limit ratio')


    URL_facereg_main = ("/", "web_server_iu2")
    # app = web.application(URL_facereg_main,locals())
    app = web.application(URL_facereg_main, globals())

    app.run()
