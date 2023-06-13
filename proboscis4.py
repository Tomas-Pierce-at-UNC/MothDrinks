#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:41:46 2023

@author: tomas
"""

from tensorflow import keras
from skimage.io import imshow
# from skimage import segmentation, morphology
from matplotlib import pyplot
import numpy as np

from cine import Cine
import tube2
from skimage import measure, morphology
# import align

import glob
import json
import pathlib
import os
# import random

# model = keras.models.load_model("proboscis_utils/proboscis_model_b1")

# names = glob.glob("data2/*.cine")

# name0 = random.choice(names)
# print(name0)

# cine = Cine(name0)
# med = cine.get_video_median()
# m = tube2.tube_crop1(med)
# imshow(m)
# pyplot.show()

# # # I swear to god if this generalizes I will lose it
# # # as in permanently lose my mind
# # def classical_meniscus(frame, med):
# #     from skimage import feature
# #     import tube2
# #     f = tube2.tube_crop1(frame)
# #     m = tube2.tube_crop1(med)
# #     dif = f.astype(float) - m.astype(float)
# #     a = abs(dif)
# #     a[a < a.mean()] = 0
# #     blobs = feature.blob_log(a)
# #     blobs = blobs[blobs[:,2] > 1]
# #     return blobs


# def get_difference(frame, med):
#     return frame.astype(np.float32) - med.astype(np.float32)

def main():
    if not os.path.isdir("./proboscis_canidates"):
        os.mkdir("./proboscis_canidates")
    
    model = keras.models.load_model("proboscis_utils/proboscis_model_b1")
    names = glob.glob("data2/**/*.cine", recursive=True)
    names = reversed(names)
    #data = {}
    for name in names:
        measurements = []
        cine = Cine(name)
        # med = cine.get_video_median()
        # med = tube2.tube_crop1(med)
        for i in range(0,cine.image_count, 16):
            if i + 16 >= cine.image_count:
                break
            images = []
            for j in range(i,i+16):
                frame = cine.get_ith_image(j)
                tube = tube2.tube_crop1(frame)
                images.append(tube)
            try:
                in_arr = np.array(images)
            except ValueError as e:
                print(e)
                continue
            in_arr = in_arr[...,np.newaxis]
            if in_arr.shape != (16, 600, 100, 1):
                print(in_arr.shape)
                print("shape is wrong")
                continue
            out = model(in_arr)
            out_arr = out.numpy()
            for k in range(0,16):
                prediction = out_arr[k,:,:,0]
                #p = np.empty_like(prediction)
                #p[:] = prediction
                #p[p < 0] = 0
                #p = p / p.max()
                mask = prediction > 0
                ero = morphology.erosion(mask)
                labels = measure.label(ero)
                regions = measure.regionprops(labels, images[k])
                regions = [region for region in regions if region.area > 100]
                idx = i + k
                for region in regions:
                    bbox = region.bbox
                    label = region.label
                    area = region.area
                    feret = region.feret_diameter_max
                    orient = region.orientation
                    major_axis_len = region.axis_major_length
                    minor_axis_len = region.axis_minor_length
                    centroid = region.centroid
                    w_centroid = region.centroid_weighted
                    eq_diam = region.equivalent_diameter_area
                    eu = region.euler_number
                    i_max = region.intensity_max
                    i_mean = region.intensity_mean
                    i_min = region.intensity_min
                    perim = region.perimeter
                    perim_c = region.perimeter_crofton
                    solid = region.solidity
                    datum = (idx,
                             *bbox,
                             label,
                             area,
                             feret,
                             orient,
                             major_axis_len,
                             minor_axis_len,
                             *centroid,
                             *w_centroid,
                             eq_diam,
                             eu,
                             i_max,
                             i_mean,
                             i_min,
                             perim,
                             perim_c,
                             solid,
                             region.num_pixels,
                             region.area_bbox,
                             region.area_convex,
                             region.area_filled,
                             region.eccentricity,
                             region.extent,
                             )
                    measurements.append(datum)
                #imshow(p > 0.1)
                #pyplot.show()
        mmts = np.array(measurements)
        np.savetxt("./proboscis_canidates/{}.tsv".format(pathlib.Path(name).stem), mmts, delimiter='\t')
        cine.close()


if __name__ == '__main__':
    main()