#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 11:50:50 2023

@author: tomas
"""

import pathlib
import math
import random
import numpy as np
from cine import Cine
import tube2
from skimage.io import imshow
from matplotlib import pyplot, patches
from sklearn import cluster, ensemble

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

idx = random.randint(0, len(video_names) - 1)

for idx in range(0, len(video_names) - 1):
    vname = video_names[idx]
    sheet = spreadsheets[idx]
    
    table = np.loadtxt(sheet)
    cine = Cine(vname)
    
    #isofor = ensemble.IsolationForest()
    #labels = isofor.fit_predict(table[:,[6,7,8,9,10,15,16,17,18,19,20,21]])
    
    pyplot.hist(table[:,10] / table[:, 9])
    pyplot.show()
    ratio = table[:,10] / table[:,9]
    
    for i in range(cine.image_count):
        img = cine.get_ith_image(i)
        res = tube2.tube_crop1(img)
        subtable = table[table[:,0] == i]
        subtable = subtable[subtable[:,6] > 50]
        subratio = ratio[table[:,0] == i]
        #sublabels = labels[table[:,0] == i]
        #kmean = cluster.KMeans(n_clusters=2)
        if len(subtable) > 0:
            row = max(subtable[:,3])
            #labels = kmean.fit_predict(subtable)
        
        my_patches = []
        for j,entry in enumerate(subtable):
            p = patches.Ellipse((entry[12], entry[11]), 
                                width=entry[10],
                                height=entry[9], 
                                angle=-math.degrees(entry[8]), 
                                fill=False,
                                color="red" if subratio[j] < 0.9 else "blue")
            my_patches.append(p)
        
        ax=pyplot.gca();ax.imshow(res);[ax.add_patch(p) for p in my_patches];pyplot.show();
        if len(subtable) > 0:
            #imshow(res);pyplot.hlines(row, 0, 100, colors="red");pyplot.show()
            pass
        #pyplot.show()
    
    cine.close()
