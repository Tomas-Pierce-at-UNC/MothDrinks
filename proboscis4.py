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
import random

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

def difference(left, right):
    return left.astype(np.int32) - right.astype(np.int32)

def batch_difference(batch, right):
    out = np.empty_like(batch, dtype=np.int32)
    for i in range(len(batch)):
        out[i,:,:,0] = difference(batch[i,:,:,0], right)
    return out

def load_batch(video, i, batch_size=16):
    frames = []
    for j in range(i, i+batch_size):
        frame = video.get_ith_image(j)
        tube = tube2.tube_crop1(frame)
        frames.append(tube)
    try:
        batch = np.array(frames)
    except ValueError as e:
        print(e)
        return None
    return batch[..., np.newaxis]

def predict_batch(batch, p_model, thresh=0.1):
    p = p_model(batch)
    pred = p.numpy()
    masks = pred >= thresh
    out = np.empty_like(masks)
    for i in range(len(batch)):
        out[i,:,:,0] = morphology.remove_small_objects(masks[i,:,:,0],
                                                       min_size=50,
                                                       connectivity=2)
    return out

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
                mask = prediction > 0.1
                big = morphology.remove_small_objects(mask, min_size=50, connectivity=2)
                label = measure.label(big)
                regiontable = measure.regionprops_table(label,properties=('label','bbox','eccentricity'))
                datum = max(regiontable['bbox-2'], default=-10)
                measurements.append((i+k, datum))
                #imshow(p > 0.1)
                #pyplot.show()
        mmts = np.array(measurements)
        np.savetxt("./proboscis_canidates/{}.tsv".format(pathlib.Path(name).stem), mmts, delimiter='\t')
        cine.close()


def measure_proboscis_position(videoname):
    #main()
    prob_model = keras.models.load_model('proboscis_utils/proboscis_model_b1')
    video = Cine(videoname)
    m = video.get_video_median()
    median_img = tube2.tube_crop1(m)
    measurements = []
    for i in range(0, video.image_count, 16):
        print("*")
        if i + 15 >= video.image_count:
            break
        batch = load_batch(video, i)
        if batch is None:
            continue
        if batch.shape != (16, 600, 100, 1):
            continue
        difs = batch_difference(batch, median_img)
        p = predict_batch(batch, prob_model)
        for j in range(16):
            lbl = measure.label(p[j,:,:,0])
            regions = measure.regionprops(lbl, difs[j,:,:,0])
            regions.sort(key = lambda region : region.bbox[2], reverse=True)
            for region in regions:
                if region.intensity_mean < 0:
                    measurements.append(((i+j), region.bbox[2]))
                    break
                    
    video.close()
    return measurements

def display(lookup, vidname):
    video = Cine(vidname)
    for i in range(video.image_count):
        if i in lookup:
            y = lookup[i]
            frame = video.get_ith_image(i)
            imshow(frame)
            pyplot.hlines(y, 50, 750, colors='red', linestyle='dashed')
            pyplot.show(block=False)
            pyplot.pause(0.25)
            pyplot.close()
    video.close()


def main():
    if not os.path.isdir("./proboscis_measurements"):
        os.mkdir("./proboscis_measurements")
    vids = glob.glob("data2/*.cine")
    vids.extend(glob.glob("data2/unsuitableVideos/*.cine"))
    vids.reverse()
    for name in vids:
        try:
            measurements = measure_proboscis_position(name)
        except ValueError as e:
            print(e)
            print(name)
            continue
        mmts = np.array(measurements)
        stem = pathlib.Path(name).stem
        np.savetxt("./proboscis_measurements/{}.tsv".format(stem),
                   mmts,
                   delimiter='\t'
                   )
        print('#')


if __name__ == '__main__':
    main()
