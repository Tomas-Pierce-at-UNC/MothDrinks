#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:59:24 2023

@author: tomas
"""

import pathlib
import random

import numpy as np
from matplotlib import pyplot
import pandas
from skimage.io import imshow

import meniscus_rate2
import real_unit_convert as ruc
from cine import Cine
import tube2

def get_cine(name, collection):
    for fname in collection:
        if str(fname).endswith(name):
            return Cine(fname)
    raise LookupError
    
def merged_datasheet(index, m_list, p_list):
    m_name = m_list[index]
    p_name = p_list[index]
    raw_m_table = np.loadtxt(m_name, dtype=float, delimiter=' ')
    m_table = meniscus_rate2.apply_isolation_forest(raw_m_table)
    raw_p_table = np.loadtxt(p_name, dtype=float, delimiter='\t')
    meniscus_dframe = pandas.DataFrame(m_table,
                                       columns=("idx", "y", "area", "minrow","mincol","maxrow","maxcol"))
    proboscis_dframe = pandas.DataFrame(raw_p_table,
                                        columns=('idx', 
                                                 'min_row_p',
                                                 'min_col_p',
                                                 'max_row_p',
                                                 'max_col_p',
                                                 'label_p',
                                                 'area_p',
                                                 'feret_p',
                                                 'orient_p',
                                                 'major_axis_len_p',
                                                 'minor_axis_len_p',
                                                 'centroid_row_p',
                                                 'centroid_col_p',
                                                 'wcentroid_row_p',
                                                 'wcentroid_col_p',
                                                 'eq_diam_p',
                                                 'eu_p',
                                                 'i_max_p', 
                                                 'i_mean_p', 
                                                 'i_min_p',
                                                 'perim_p', 
                                                 'perim_c_p', 
                                                 'solid_p')
                                        )
    #proboscis_dframe['idx'] = proboscis_dframe[0]
    merged = pandas.merge(meniscus_dframe, proboscis_dframe, how='outer', on=('idx'))
    return merged

vid_dir = pathlib.Path("data2")
vid_dir2 = pathlib.Path("data2/unsuitableVideos")
meniscus_dir = pathlib.Path("meniscusTrackNN")
proboscis_dir = pathlib.Path("proboscis_canidates")

video_names = []
video_names.extend(vid_dir.glob("*.cine"))
video_names.extend(vid_dir2.glob("*.cine"))

meniscus_filenames = sorted(meniscus_dir.glob("*.csv"))
proboscis_filenames = sorted(proboscis_dir.glob("*.tsv"))

uc = ruc.UnitConversion()

stop = random.choice(range(len(meniscus_filenames)))
#stop=4
#stop=58

v_name = meniscus_rate2.get_corresponding_name(meniscus_filenames[stop])

merged_dsheet = merged_datasheet(stop, meniscus_filenames, proboscis_filenames)

vid =  get_cine(v_name, video_names)



#dsheet_filt = merged_dsheet[(abs(dist_left) < 70) & (abs(dist_right) < 70)]
dsheet_filt = merged_dsheet[merged_dsheet['area_p'] > 10]

dist_left = dsheet_filt['mincol'] - dsheet_filt['min_col_p']
dist_right = dsheet_filt['maxcol'] - dsheet_filt['max_col_p']

dsheet_filt = dsheet_filt[(abs(dist_left) < 70) & (abs(dist_right) < 70)]
dsheet_filt = dsheet_filt[dsheet_filt['min_row_p'] < 590]
#dsheet_filt = dsheet_filt[dsheet_filt['']]

indices = pandas.unique(dsheet_filt['idx'])


try:
    for idx in indices:
        subgroup = dsheet_filt[dsheet_filt['idx'] == idx]
        frame = vid.get_ith_image(int(idx))
        f = tube2.tube_crop1(frame)
        lowest = subgroup.nlargest(3,'max_row_p')
        #lowest.sort_values(by="")
        imshow(f);pyplot.hlines(lowest["max_row_p"], 0, 100, colors="red");pyplot.show()
    
finally:
    vid.close()