#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from sklearn.preprocessing import normalize
import matplotlib.patches as patches
import argparse
import math
from itertools import repeat

from insightface.helper import nms, adjust_input, detect_first_stage_warpper, face_preprocess
from insightface.mtcnn_detector import MtcnnDetector
from math import sqrt

from paths.paths import model_path, image_path

# result codes for methods
RESULT_OK = 0
RESULT_GENERAL_ERROR = -1
RESULT_NO_FACE_DETECTED = -2


class InsightfaceDetector:
    def __init__(self, model_path, mtcnn_path=None, epoch_num='0000', image_size=(112, 112),
                 no_face_raise=True):
        self.model_path = ','.join([model_path, epoch_num])
        #print(self.model_path)
        self.no_face_raise = no_face_raise
        args = argparse.Namespace()
        args.model = self.model_path
        args.det = 0
        args.flip = 0
        args.threshold = 1.24
        args.ga_model = ''
        args.image_size = ",".join([str(i) for i in image_size])
        
        args.mtcnn_path = ''
        if mtcnn_path is not None:
            args.mtcnn_path = mtcnn_path
        self.model = FaceModel(args)

    def detect(self, image):
        # @brief performs face detection on input image
        # @retval: a tuple containing result code and the image

        result, blurred_image, face_boxes = self.model.get_input(image)

        if result != RESULT_OK:
            if self.no_face_raise:
                return RESULT_NO_FACE_DETECTED, image, None
            else:
                return RESULT_GENERAL_ERROR, image, None
        else:
            return RESULT_OK, blurred_image, face_boxes


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        print("INITIALIZATION ARGUMENTS: ", args)
        self.args = args
        ctx = mx.gpu()

        try:
            mx.nd.array([1, 2, 3], ctx = ctx)
        except mx.MXNetError:
            ctx = mx.cpu()
            print("no GPU available, running mxnet on CPU")
        
        _vec = args.image_size.split(',')
        assert len(_vec)==2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        self.ga_model = None
        if len(args.model)>0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')
        if len(args.ga_model)>0:
            self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

        self.threshold = args.threshold
        self.det_minsize = 50
        self.det_threshold = [0.6,0.7,0.8]
        self.image_size = image_size
      
        mtcnn_path = ''
        if len(args.mtcnn_path) > 0:
            mtcnn_path = args.mtcnn_path
        else:
            # code default
            mtcnn_path = os.path.join("insightface", 'mtcnn-model')

        print("MTCNN_PATH: ", mtcnn_path)
        # mtcnn_path = "/home/student/Documents/AnonAI/face_detection/models/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model"
        # TODO num_worker - can increase in number improve performance
        detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
        self.detector = detector


    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type = self.args.det)
        #print("coordinates:", ret)
        if ret is None:
            return RESULT_NO_FACE_DETECTED, None, None
            
        bbox, points = ret
        #blur_image = face_img.copy()
        blur_image = face_img.copy()
        points_on_img = face_img.copy()

        face_boxes = []
        for i in range(0, bbox.shape[0]):
            rec_x = int(bbox[i,0])
            rec_y = int(bbox[i,1])
            rec_width = int(bbox[i,2] - bbox[i,0])
            rec_height = int(bbox[i,3] - bbox[i,1])
            blur_image = blur_face(blur_image, rec_x, rec_y, rec_width, rec_height)
            face_boxes.append([rec_x, rec_y, rec_width, rec_height])

        # draw_points(points_on_img, points)
        
        # cv2.imshow("title", blur_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return RESULT_OK, blur_image, face_boxes



def detect_face_on_image(image_path, model_path):
    detector = InsightfaceDetector(model_path=model_path, epoch_num='0000', image_size=(112, 112))
    img_name = cv2.imread(image_path)
    detector.detect(img_name)

def detect_face_video_frame(frame, detector):
    # @brief detects faces on video frame and blurs them
    # @retval video framed with blurred faces

    res, frame_blurred, face_boxes = detector.detect(frame)
    return res, frame_blurred, face_boxes

def detection_model_init(model_path, mtcnn_path=None):
    detector = InsightfaceDetector(model_path=model_path, mtcnn_path=mtcnn_path, epoch_num='0000', image_size=(112, 112))
    return detector

    
def draw_points(face_img, points):

    for point in points:
        #print("point:", point)
        pts = np.array([[point[i], point[i+1]] for i in range(0, len(point)-1, 2)], np.int32)
        
        for i in range(0, len(pts)):
            #print((pts[i][0], pts[0][1]))
            #cv2.rectangle(points_on_img, (pts[i][0], pts[i][1]), (pts[i][0]+2, pts[i][1]+2), (255,255,0), 2)
            cv2.circle(face_img,(pts[i][0], pts[i][1]), 3, (255, 0,0), -1)
        
    #cv2.imshow("title", face_img)
    
def blur_face(face_img, x, y, w, h, filter_coeff=0.25, sigma_coeff=0.75):
  # @brief blur the rectangular area in the image
  # @param face_img: input image encoded as numpy array
  # @param x, y: top left corner of the rectangular area
  # @param w, h: width and height of the rectangular area

  if w<0 or h<0:
    print('blur_face: invalid box input dimensions')
    return face_img

  if x > face_img.shape[1] or y > face_img.shape[0]:
    return face_img

  if x<0:
    x = 0

  if y<0:
    y = 0

  blur_image = face_img
  cv2.rectangle(blur_image, (x, y), (x+w, y+h), (255,255,0), thickness=0)
  sub_face = blur_image[y:y+h, x:x+w]

  # set filter according to rectangle surface area
  # filter size must be an odd number
  face_surface = w*h
  filter_size = round(filter_coeff*sqrt(face_surface)/2)*2+1
  filter_sigma = filter_size/sigma_coeff

  #print('face surface: ', w*h)
  #print('filter parameters: ', filter_size, filter_sigma)
  sub_face = cv2.GaussianBlur(sub_face, (filter_size,filter_size), filter_sigma)
  blur_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
  return blur_image

# detect_face_on_image(image_path, model_path = model_path)
