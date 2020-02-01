# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:27:57 2019

@author: Ramshankar Yadhunath
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 3.0, theta, 8.0, 30.0, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
 accum = np.zeros_like(img)
 for kern in filters:
     fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
     np.maximum(accum, fimg, accum)
 return accum
 
if __name__ == '__main__':
 import sys
 
 try:
     img_fn = sys.argv[1]
 except:
     img_fn = "C:\\Users\\Ramshankar Yadhunath\\Desktop\\Detecting Building Defects\\Paint  Peeling (From home)\\Paint  Peeling (From home)\\Defect\\6.jpg"
 
 img = cv2.imread(img_fn)
 if img is None:
     print ('Failed to load image file:'), img_fn
     sys.exit(1)
 
 filters = build_filters()
 
 res1 = process(img, filters)
 cv2.imshow('result', res1)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

#im = cv2.imread("C:\\Users\\Ramshankar Yadhunath\\Desktop\\Detecting Building Defects\\Concrete Crack Images for Classification\\TheWallCrack_GUI_Test_Set_Positive\\00096.jpg")
#plt.imshow(im, cmap="gray")

