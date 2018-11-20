#!/usr/bin/python
import math
import json
import cv2 
import numpy as np

class HomograpyTransform:
    pt_0_pix_src_image_X = 2
    pt_0_pix_src_image_Y = 2
    pt_0_pix_dst_image_X = 2
    pt_0_pix_dst_image_Y = 2	
    pt_1_pix_src_image_X = 1
    pt_1_pix_src_image_Y = 1
    pt_1_pix_dst_image_X = 1
    pt_1_pix_dst_image_Y = 1
    pt_2_pix_src_image_X = 6
    pt_2_pix_src_image_Y = 6
    pt_2_pix_dst_image_X = 6
    pt_2_pix_dst_image_Y = 6
    pt_3_pix_src_image_X = 3
    pt_3_pix_src_image_Y = 3
    pt_3_pix_dst_image_X = 3
    pt_3_pix_dst_image_Y = 3
    pt_4_pix_src_image_X = 2
    pt_4_pix_src_image_Y = 2
    pt_4_pix_dst_image_X = 2
    pt_4_pix_dst_image_Y = 2
    _log = None
    
    def __init__(self):
        with open('configReal.json') as json_data_file:
            data = json.load(json_data_file)  	
        #configuration homograpy parameters    
        self.pt_0_pix_src_image_X = float(data['config']['pt_0_pix_src_image_X'])
        print(self.pt_0_pix_src_image_X)
        self.pt_0_pix_src_image_Y = float(data['config']['pt_0_pix_src_image_Y'])
        self.pt_0_pix_dst_image_X = float(data['config']['pt_0_pix_dst_image_X'])
        self.pt_0_pix_dst_image_Y = float(data['config']['pt_0_pix_dst_image_Y'])		
        self.pt_1_pix_src_image_X = float(data['config']['pt_1_pix_src_image_X'])
        self.pt_1_pix_src_image_Y = float(data['config']['pt_1_pix_src_image_Y'])
        self.pt_1_pix_dst_image_X = float(data['config']['pt_1_pix_dst_image_X'])
        self.pt_1_pix_dst_image_Y = float(data['config']['pt_1_pix_dst_image_Y'])		
        self.pt_2_pix_src_image_X = float(data['config']['pt_2_pix_src_image_X'])
        self.pt_2_pix_src_image_Y = float(data['config']['pt_2_pix_src_image_Y'])
        self.pt_2_pix_dst_image_X = float(data['config']['pt_2_pix_dst_image_X'])
        self.pt_2_pix_dst_image_Y = float(data['config']['pt_2_pix_dst_image_Y'])
        self.pt_3_pix_src_image_X = float(data['config']['pt_3_pix_src_image_X'])
        self.pt_3_pix_src_image_Y = float(data['config']['pt_3_pix_src_image_Y'])
        self.pt_3_pix_dst_image_X = float(data['config']['pt_3_pix_dst_image_X'])
        self.pt_3_pix_dst_image_Y = float(data['config']['pt_3_pix_dst_image_Y'])
        self.pt_4_pix_src_image_X = float(data['config']['pt_4_pix_src_image_X'])
        self.pt_4_pix_src_image_Y = float(data['config']['pt_4_pix_src_image_Y'])
        self.pt_4_pix_dst_image_X = float(data['config']['pt_4_pix_dst_image_X'])
        self.pt_4_pix_dst_image_Y = float(data['config']['pt_4_pix_dst_image_Y'])   
        # https://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/
        # provide points from image 1
        pts_src = np.array([[self.pt_0_pix_src_image_X, self.pt_0_pix_src_image_Y],[self.pt_1_pix_src_image_Y, self.pt_1_pix_src_image_Y],
                            [self.pt_2_pix_src_image_Y, self.pt_2_pix_src_image_Y],[self.pt_3_pix_src_image_Y, self.pt_3_pix_src_image_Y],
                            [self.pt_4_pix_src_image_Y, self.pt_4_pix_src_image_Y]])
        # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
        pts_dst = np.array([[self.pt_0_pix_dst_image_X, self.pt_0_pix_dst_image_Y],[self.pt_1_pix_dst_image_Y,self.pt_1_pix_dst_image_Y],
                            [self.pt_2_pix_dst_image_Y, self.pt_2_pix_dst_image_Y],[self.pt_3_pix_dst_image_Y, self.pt_3_pix_dst_image_Y],
                            [self.pt_4_pix_dst_image_Y, self.pt_4_pix_dst_image_Y]])
        # calculate matrix H
        self.h, self.status = cv2.findHomography(pts_src, pts_dst)    
    
    def calculateTransformMatrix(self, point):
        print(cv2.perspectiveTransform(point, self.h))
        return cv2.perspectiveTransform(point, self.h)
    
    def calculateHomography(self, pts_src, pts_dst):
        self.h, self.status = cv2.findHomography(pts_src, pts_dst)
    
    
#pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
#pts_dst = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])
#homograpier = HomograpyTransform()
#homograpier.calculateHomography(pts_src,pts_src)
#a = np.array([[154, 174]], dtype='float32')
#a = np.array([a])
#b = homograpier.calculateTransformMatrix(a)
#print(b)

