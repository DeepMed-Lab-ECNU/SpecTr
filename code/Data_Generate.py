#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:14:29 2020
@author: Boxiang Yun   School:ECNU&HFUT   Email:971950297@qq.com
"""
from torch.utils.data.dataset import Dataset
import skimage.io
#from skimage.metrics import normalized_mutual_information
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import cv2
import os
from argument import Transform
from spectral import *
from spectral import open_image
import random
import math
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')
from einops import repeat

class Data_Generate_Cho(Dataset):#
    def __init__(self, img_paths, seg_paths=None,
                 cutting=None, transform=None,
                 channels=None, outtype='3d'):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.transform = transform
        self.cutting = cutting
        self.channels = channels
        self.outtype = outtype

    def __getitem__(self,index):
        img_path = self.img_paths[index]
        mask_path = self.seg_paths[index]
        mask = cv2.imread(mask_path, 0)/255
        img = envi.open(img_path)[:, :, :]
        img = img[:, :, self.channels] if self.channels is not None else img

        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        if self.transform != None:
            img, mask = self.transform((img, mask))

        mask = mask.astype(np.uint8)
        if self.cutting is not None:
            while(1):
                xx = random.randint(0, img.shape[0] - self.cutting)
                yy = random.randint(0, img.shape[1] - self.cutting)
                patch_img = img[xx:xx + self.cutting, yy:yy + self.cutting]
                patch_mask = mask[xx:xx + self.cutting, yy:yy + self.cutting]
                if patch_mask.sum()!=0: break
            img = patch_img
            mask = patch_mask


        img = img[:, :, None] if len(img.shape)==2 else img
        img = np.transpose(img, (2, 0, 1))
        if self.outtype == '3d':
            img = img[None]
        mask = mask[None, ].astype(np.float32)
        img = img.astype(np.float32)
        return img, mask
            
    def __len__(self):
        return len(self.img_paths)