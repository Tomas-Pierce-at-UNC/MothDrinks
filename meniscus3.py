#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:04:15 2023

@author: tomas
"""

import glob
import pathlib
import multiprocessing as multi
import random

from skimage import filters, measure, util, morphology as morpho
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot
import numpy as np

from cine import Cine
import tube2
import align


def difference(left, right):
    """Take the difference of two images of the same shape"""
    return util.img_as_int(left) - util.img_as_int(right)


def get_width(regionprop):
    return regionprop.bbox[3] - regionprop.bbox[1]


def locate_object_regions(frame, mediantube, aligner):
    tube = tube2.constrain_to_tube_refwidth(frame, mediantube)
    aligned = aligner.align(tube)
    delta = difference(aligned, mediantube)
    thresh = filters.threshold_isodata(delta)
    low = delta < thresh
    eroded = morpho.binary_erosion(low)
    applied = eroded * delta
    labeled = measure.label(eroded)
    props = measure.regionprops(labeled, applied)
    return props


def form_row(regionprop) -> tuple:
    return (*regionprop.centroid,  # 2, ;0, 1
            *regionprop.centroid_weighted,  # 2 ;2 3
            *regionprop.bbox,  # 4 ; 4 5 6 7
            regionprop.area,  # 1 ; 8
            regionprop.major_axis_length,  # 1 ; 9
            regionprop.minor_axis_length,  # 1 ; 10
            regionprop.orientation,  # 1 ; 11
            regionprop.euler_number,  # 1; 12
            regionprop.feret_diameter_max,  # 1 ; 13
            regionprop.intensity_max,  # 1 ; 14
            regionprop.intensity_mean,  # 1 ; 15
            regionprop.intensity_min,  # 1; 16
            regionprop.perimeter,  # 1 ; 17
            regionprop.solidity,  # 1 ; 18
            regionprop.perimeter_crofton,  # 1 ; 19
            regionprop.equivalent_diameter_area,  # 1 ; 20
            regionprop.area_convex,  # 1 ; 21
            regionprop.extent,  # 1 ; 22
            regionprop.eccentricity  # 1 ; 23
            )


def form_table(regionprops: list) -> np.ndarray:
    table = []
    bigprops = filter(lambda region: region.area > 20, regionprops)
    for region in bigprops:
        row = form_row(region)
        table.append(row)
    return np.array(table)


def attach_index(regiontable: np.ndarray, index):
    return np.c_[regiontable, np.ones((regiontable.shape[0],)) * index]


def locate_objects(video, mediantube):
    aligner = align.SiftAligner(mediantube)
    mytables = []
    for i in range(video.image_count):
        frame = video.get_ith_image(i)
        try:
            regions = locate_object_regions(frame, mediantube, aligner)
        except ValueError as e:
            print(f"frame {i} caused")
            print(e)
            continue
        except AttributeError as e:
            print(e)
            continue
        except RuntimeError as e:
            print(e)
            # skio.imshow(frame)
            continue
        table = form_table(regions)
        indexed = attach_index(table, i)
        if 0 not in indexed.shape:
            # weird things happen with empty arrays
            mytables.append(indexed)
    # try:
    sharedtable = np.concatenate(mytables)
    # except ValueError as e:
    #   # print(e)
    #   #breakpoint()
    return sharedtable


def find_real_measurements(loc_objs: np.ndarray):
    MIN_WIDTH = 15  # fifteen pixel width minimum
    MAX_AREA = 1000  # threshold of which meniscus is unlikely to be above
    wides = loc_objs[loc_objs[:, 7] - loc_objs[:, 5] > MIN_WIDTH]
    smalls = wides[wides[:, 8] < MAX_AREA]
    isofor = IsolationForest()
    lbls = isofor.fit_predict(smalls)
    return smalls, lbls


def get_random_vid():
    names = glob.glob("data2/*.cine")
    name = random.choice(names)
    return Cine(name)


def get_stem(filename: str) -> str:
    return pathlib.Path(filename).stem


def process(filename: str):
    print(filename)
    video = Cine(filename)
    filtered = None
    try:
        print("median")
        med = video.get_video_median()
        mtube = tube2.find_tube(med)
        loc_objs = locate_objects(video, mtube)
        # should remove hits to antenae
        #loc_objs = loc_objs[loc_objs[:,7] - loc_objs[:,5] < mtube.shape[1]]
        w, lbls = find_real_measurements(loc_objs)
        filtered = w[lbls == 1]
        # low = filtered[filtered[:,7] < filtered[:,7].mean()]
        pyplot.scatter(filtered[:, -1], filtered[:, 0], marker='.')
        stem = get_stem(filename)
        np.savetxt("meniscusTracks6/{}.csv".format(stem), loc_objs, delimiter=",")
        pyplot.title(stem)
        pyplot.savefig("meniscusTracks6/{}.png".format(stem))
        pyplot.close()
    except Exception as e:
        print(e)
    finally:
        video.close()
    return filename, filtered



if __name__ == '__main__':
    names = glob.glob("data2/*.cine")
    names.extend(glob.glob("data2/unsuitableVideos/*.cine"))
    measurements = {}
    with multi.Pool(16) as pool:
        for fname, data in pool.imap_unordered(process, names):
            measurements[fname] = data
