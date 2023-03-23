#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:04:14 2023

@author: tomas
"""

import glob
from cine import Cine
import tube2
from meniscus3 import locate_objects, find_real_measurements, get_stem
from matplotlib import pyplot

filenames = glob.glob("data2/unsuitableVideos/*.cine")

def process(filename):
    print(filename)
    video = Cine(filename)
    try:
        print("median")
        med = video.get_video_median()
        mtube = tube2.find_tube(med)
        loc_objs = locate_objects(video, mtube)
        w, lbls = find_real_measurements(loc_objs)
        filtered = w[lbls == 1]
        #low = filtered[filtered[:,7] < filtered[:,7].mean()]
        pyplot.scatter(filtered[:,-1], filtered[:,0], marker='.')
        stem = get_stem(filename)
        pyplot.title(stem)
        pyplot.savefig("meniscusTracksUnsuit/{}.png".format(stem))
        pyplot.close()
    except Exception as e:
        print(e)
    finally:
        video.close()
    return filename

for filename in filenames:
    name = process(filename)
    print(name)