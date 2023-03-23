#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:57:50 2023

@author: tomas
"""

import glob
from cine import Cine
import meniscus_rate as mr
import tube2
import numpy as np
import pathlib
from skimage.io import imsave
import pandas as pd


if __name__ == '__main__':
    videonames = glob.glob("data2/*.cine")
    videonames.extend(glob.glob("data2/unsuitableVideos/*.cine"))
    csvnames = glob.glob("meniscusTracks5/*.csv")
    videostems = list(map(mr.get_stem, videonames))
    
    framepath1 = pathlib.Path("framedata")
    if not framepath1.exists():
        framepath1.mkdir()
    
    for name in csvnames:
        stem = mr.get_stem(name)
        index = videostems.index(stem)
        videoname = videonames[index]
        table = mr.load_measurements(name)
        realtable = mr.find_real_data(table)       
        indices = np.unique(realtable[:,-1])
        video = Cine(videoname)
        subpath = framepath1 / stem
        if not subpath.exists():
            subpath.mkdir()
        for index in indices:
            frame = video.get_ith_image(int(index))
            cropped = tube2.tube_crop1(frame)
            img_name = subpath / "frame{}.png".format(index)
            imsave(img_name, cropped)
            subdata = realtable[realtable[:,-1] == index]
            tbl_name = subpath / "frame{}.csv".format(index)
            np.savetxt(str(tbl_name), subdata)
        video.close()
