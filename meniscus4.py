#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:53:55 2023

@author: tomas
"""

from tensorflow import keras
from cine import Cine
import tube2
import glob
from skimage import measure, transform, util, morphology as morpho
import numpy as np
import pathlib

BATCH_SIZE = 32

def measure_video(videoname, m_model):
    video = Cine(videoname)
    measurements = []
    for i in range(0, video.image_count, BATCH_SIZE):
        print('.',end='')
        images = []
        for j in range(i, min(i + BATCH_SIZE, video.image_count)):
            img = video.get_ith_image(j)
            tb1 = tube2.tube_crop1(img)
            if 0 in tb1.shape:
                # put in filler zeros to skip over this one
                tb1 = np.zeros((600,100))
                print("skipping {}".format(j))
            tb2 = transform.resize(tb1, (600,100))
            tb3 = util.img_as_ubyte(tb2)
            images.append(tb3)
        batch = np.array(images)
        batch = batch[...,np.newaxis]
        batch_out = m_model(batch)
        out_imgs = batch_out.numpy()
        meniscus_masks = out_imgs > 0
        for k in range(len(images)):
            mask = meniscus_masks[k,:,:,0]
            op = morpho.binary_opening(mask)
            labeled = measure.label(op)
            regions = measure.regionprops(labeled)
            if len(regions) == 0:
                print("skipped frame {}".format(i + k))
                continue
            biggest = max(regions, key = lambda prop: prop.area)
            centroid = biggest.centroid
            y_coord = centroid[0]
            measurements.append((i + k, y_coord, biggest.area, *biggest.bbox))
    video.close()
    return measurements

def process_video(videoname, model):
    stem = pathlib.Path(videoname).stem
    mmts = measure_video(videoname, model)
    table = np.array(mmts)
    np.savetxt("meniscusTrackNN/{}.csv".format(stem), table)
    

def main():
    meniscus_model = keras.models.load_model("model_attempt3")
    videonames = glob.glob("data2/*.cine")
    videonames.extend(glob.glob("data2/unsuitableVideos/*.cine"))
    for name in videonames:
        process_video(name, meniscus_model)
        print("{} handled".format(name))


if __name__ == '__main__':
    main()
