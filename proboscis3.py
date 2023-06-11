#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:31:10 2023

@author: tomas
"""


from tensorflow import keras
from cine import Cine
from tube2 import tube_crop1

import glob
import math

import numpy as np
from skimage.io import imshow
from skimage import measure, draw
from matplotlib import pyplot
from matplotlib.patches import Ellipse
import pathlib


def handle_vid(name, model):
    
    #print(name)
    cine = Cine(name)
    
    measurements = []
    
    for i in range(0, cine.image_count, 8):
        #print('.')
        frames = []
        if cine.image_count - 1 < i + 8:
            continue
        for j in range(i, min(i+8, cine.image_count - 1)):
            img = cine.get_ith_image(j)
            tube = tube_crop1(img)
            frames.append(tube)
            #imshow(tube)
            #pyplot.show()

        try:
            batch_in = np.array(frames)[...,np.newaxis]
            if batch_in.shape != (8, 600, 100, 1):
                print("wrong shape for data")
                continue
        except ValueError as e:
            print(f"experienced {e}, skipping {i}:{i+8} in {name}")
            continue
        model_out = model(batch_in)
        out_arr = model_out.numpy()
        for k in range(len(out_arr)):
            index = i + k
            mask = out_arr[k,:,:,0] > 0.15
            labels = measure.label(mask)
            regions = measure.regionprops(labels, batch_in[k,:,:,0])
            #fig, ax = pyplot.subplots()
            #ax.imshow(mask)
            for region in regions:
                bbox = region.bbox
                lowest = bbox[2]
                orient = region.orientation
                major = region.axis_major_length
                minor = region.axis_minor_length
                eccent = region.eccentricity
                center = region.centroid
                brightness = region.intensity_mean
                #el = Ellipse((center[1], center[0]), minor, major, angle=-math.degrees(orient), linewidth=1, edgecolor='r')
                #ax.add_patch(el)
                mmt = (index, lowest, orient, major, minor, eccent, orient, *center, brightness)
                measurements.append(mmt)
            #pyplot.show()
    
    cine.close()
    #measurement_assembly[name] = np.array(measurements)
    path = pathlib.Path(name)
    stem = path.stem
    position_data = np.array(measurements)
    np.savetxt("proboscisTrackNN/{}.csv".format(stem), position_data)


if __name__ == '__main__':
    model = keras.models.load_model("proboscis_wdecay_5")
    names = glob.glob("data2/**/*.cine", recursive=True)
    for name in reversed(names):
        print(name)
        handle_vid(name, model)
