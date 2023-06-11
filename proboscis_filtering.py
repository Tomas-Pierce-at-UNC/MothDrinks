#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 08:11:31 2023

@author: tomas
"""


import numpy as np
import glob
import random
import pathlib
import math
from matplotlib import pyplot
from skimage.io import imshow
from matplotlib.patches import Ellipse
from sklearn import decomposition, cluster
from skimage.filters import threshold_isodata
import cine
import tube2

video_folder = pathlib.Path("data2")
video_filenames = list(video_folder.glob("./**/*.cine"))
measurement_folder = pathlib.Path("proboscis_canidates")
spreadsheets = list(measurement_folder.glob("*.tsv"))

stems1 = [sheet.stem for sheet in spreadsheets]
stems2 = [vid.stem for vid in video_filenames]

# at position i, gives the index of the entry in video_filenames which
# corresponds to the ith entry in spreadsheets
indexes = [stems2.index(s) for s in stems1]

assert -1 not in indexes

video_names = [video_filenames[index] for index in indexes]

idx = random.randint(0, len(spreadsheets) - 1)
z = spreadsheets[idx]
zvid = video_names[idx]


#z = "proboscisTrackNN/moth22_2022-02-01_Cine1.csv"
print(z)

#zvid = vid_names[30]
#zvid = "data2/moth22_2022-02-01_Cine1.cine"
print(zvid)

table = np.loadtxt(z, delimiter='\t')

video = cine.Cine(zvid)

base = table[table[:,0] == 0]

f = video.get_ith_image(0)

t = tube2.tube_crop1(f)

pyplot.figure()
ax = pyplot.gca()
imshow(t)
for entry in base:
    index, lowest, orient, major, minor, eccent, orient, row, col, brightness = entry
    ellipse = Ellipse((col, row), minor, major, orient, edgecolor='r', fc='None')
    ax.add_patch(ellipse)

pyplot.show()

thresh = threshold_isodata(table[:,5])

for i in range(video.image_count):
    f = video.get_ith_image(i)
    mmts = table[table[:,0] == i]
    t = tube2.tube_crop1(f)
    ax = pyplot.gca()
    imshow(t)
    for entry in mmts:
        index, lowest, orient, major, minor, eccent, orient, row, col, brightness = entry
        if eccent < thresh:
            continue
        ellipse = Ellipse((col, row), minor, major, -math.degrees(orient), edgecolor='r', fc='None')
        ax.add_patch(ellipse)
    pyplot.show()
    
video.close()