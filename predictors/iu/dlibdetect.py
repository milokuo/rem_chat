"""Reference and modified from rude-carnie project"""

import dlib
import cv2
import os

FACE_PAD = 50
WRITE_ORIGINAL_IMAGE = True

class FaceDetectorDlib(object):
    def __init__(self, model_name, basename='frontal-face', tgtdir='./output'):
        print("dlib - DLIB_USE_CUDA: {}:".format(dlib.DLIB_USE_CUDA))
        print("dlib - count gpu devices: {}".format(dlib.cuda.get_num_devices()))
        self.tgtdir = tgtdir
        self.detector = dlib.get_frontal_face_detector()
        # if(dlib.DLIB_USE_CUDA):
        #     self.detector = dlib.cnn_face_detection_model_v1()
        # else:
        #     self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_name)
        self.imgs = []
        self.locations = []
        self.ratios = []

    def run(self, image_file):
        name_list = image_file.split('/')
        name_list = name_list[len(name_list) - 1].split('.')
        self.basename = name_list[len(name_list) - 2]
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        images = []
        filenames = []
        bb = []
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            bb.append((x,y,w,h))
            filename = '%s_%d.jpg' % (self.basename, i + 1)
            filenames.append(filename)
            images.append(self.sub_image(os.path.join(self.tgtdir, filename), img, x, y, w, h))

        #print('%d faces detected' % len(images))

        for (x, y, w, h) in bb:
            self.draw_rect(img, x, y, w, h)
                # Fix in case nothing found in the image
        outfile = '%s/%s.jpg' % (self.tgtdir, self.basename)
        if WRITE_ORIGINAL_IMAGE:
            cv2.imwrite(outfile, img)
        return images, filenames

    def sub_image(self, name, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        #self.locations.append((lower_cut[0]/img.shape[0], upper_cut[0]/img.shape[0], lower_cut[1]/img.shape[1], upper_cut[1]/img.shape[1]))
        #We multiply by 1000 so that the relative coordinates can be stored as integers with enough resolution.
        self.locations.append((1000*lower_cut[1]/img.shape[1], 1000*lower_cut[0]/img.shape[0], 1000*upper_cut[1]/img.shape[1], 1000*upper_cut[0]/img.shape[0]))
        self.imgs.append(roi_color)
        ratio = (roi_color.shape[0] * roi_color.shape[1]) / (img.shape[0] * img.shape[1])
        self.ratios.append(round(1000 * ratio, 0))
        cv2.imwrite(name, roi_color)
        return name

    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)
