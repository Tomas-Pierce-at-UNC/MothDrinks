#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:21:38 2023

@author: tomas
"""

from cine import Cine
import glob
import align
import numpy as np
import random

from skimage.filters import sobel, threshold_li
from skimage.io import imshow
from matplotlib import pyplot

def difference(left, right):
    return left.astype(np.int16) - right.astype(np.int16)

def get_mask(frame, medframe, aligner):
    aligned = aligner.align(frame)
    delta = difference(aligned, medframe)
    delta[delta > 0] = 0
    delta = abs(delta)
    li = threshold_li(delta)
    above = delta > li
    return above

if False:
    names = glob.glob('data2/*.cine')
    
    exname = 'data2/mothM1_2022-09-23_Cine1.cine'
    
    c = Cine(names[0])
    
    med = c.get_video_median()
    
    aligner = align.SiftAligner(med)
    
    for i in range(c.image_count):
        frame = c.get_ith_image(i)
        m = get_mask(frame, med, aligner)
        if i % 31 == 0:
            imshow(frame)
            pyplot.show()
            imshow(m)
            pyplot.show()
    
    c.close()