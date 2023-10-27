import json
import pandas
import aggregate_figures as af
import tube2
import cine
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
import seaborn
from skimage.filters import threshold_li, threshold_isodata

def get_batch(video, idx, batch_size=8):
    zone = []
    if idx + batch_size >= video.image_count:
        return None
    for i in range(batch_size):
        frame = video.get_ith_image(idx + i)
        cropped = tube2.tube_crop1(frame)
        if cropped.shape != (600, 100):
            return None
        zone.append(cropped[..., np.newaxis])
    zone = np.array(zone)
    return zone

def load_medians(f_names: list):
    t_med = {}
    for name in f_names:
        print(name)
        try:
            vid = cine.Cine(name)
            m = vid.get_video_median()
            med = tube2.tube_crop1(m)
            t_med[name] = med
        finally:
            vid.close()
    return t_med

MENFILE = 'meniscus_measurements.json'
PROFILE = 'proboscis_measurements.json'

model = keras.models.load_model("proboscis_utils/proboscis_model_b3")

s_tables, l_tables = af.load_all()
af.convert_units(s_tables, l_tables)
shared_s_table = pandas.concat(s_tables)
shared_l_table = pandas.concat(l_tables)

count1 = len(shared_l_table[shared_l_table['rate'] < 0])
count2 = len(shared_l_table)
print("LOESS proportion negative rates {} %".format(count1 / count2 * 100))
count3 = len(shared_s_table[shared_s_table['rate'] < 0])
count4 = len(shared_s_table)
print("SVM proportion negative rates {} %".format(count3 / count4 * 100))

sub_negatives_s = shared_s_table[shared_s_table['submergence'] <= 0]
sub_negatives_l = shared_l_table[shared_l_table['submergence'] <= 0]

rate_negatives_s = shared_s_table[shared_s_table['rate'] <= 0]
rate_negatives_l = shared_l_table[shared_l_table['rate'] <= 0]

both_neg_s = shared_s_table[(shared_s_table['submergence']<=0)&(shared_s_table['rate']<=0)]

print(len(sub_negatives_s) / len(shared_s_table))
print(len(sub_negatives_l) / len(shared_l_table))

print(len(both_neg_s) / len(shared_s_table))

fnames = pandas.unique(sub_negatives_s['filename'])
t_med = load_medians(fnames)

##for item in negatives_s.index:
##    row = negatives_s.iloc[item]
##    try:
##        vid = cine.Cine(row['filename'])
##        framenum = int(row['frame_num'])
##        ith = vid.get_ith_image(framenum)
##        tube = tube2.tube_crop1(ith)
##        ax.imshow(tube)
##        fig.show()
##        pyplot.pause(1/32)
##        fig.clear()
##        fig.add_axes(ax)
##    finally:
##        vid.close()



##fig, ax = pyplot.subplots(ncols=2)
##ax1, ax2 = ax
##
##for item in sub_negatives_l.index:
##    row = sub_negatives_l.iloc[item]
##    medtube = t_med[row['filename']]
##    try:
##        vid = cine.Cine(row['filename'])
##        f = vid.get_ith_image(int(row['frame_num']))
##        tb = tube2.tube_crop1(f)
##        if medtube.shape == tb.shape:
##            diff = tb.astype(np.int16) - medtube.astype(np.int16)
##            li = threshold_li(diff)
##            print(li)
##            ax1.imshow(tb)
##            ax2.imshow(diff)
##            fig.show()
##            pyplot.pause(1 / 32)
##            fig.clear()
##            fig.add_axes(ax1)
##            fig.add_axes(ax2)
##    finally:
##        vid.close()
