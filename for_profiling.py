#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:08:54 2023

@author: tomas
"""

from meniscus3 import locate_objects
from cine import Cine
#from align import SiftAligner
import tube2
import glob
import random

#names = glob.glob("data2/*.cine")
#filename = random.choice(names)
#print(filename)
filename = "data2/moth26_2022-02-22_Cine1.cine"

c = Cine(filename)

med = c.get_video_median()

medtube = tube2.find_tube(med)

#aligner = SiftAligner(medtube)
locs = locate_objects(c, medtube)

print("done")

# data2/moth26_2022-02-22_Cine1.cine needs looking into