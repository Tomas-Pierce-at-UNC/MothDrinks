
from cine import Cine
import align
import tube2
from skimage import filters, measure, feature, morphology
from skimage.io import imshow
from matplotlib import pyplot
import numpy as np

import glob

def difference(left: np.ndarray, right: np.ndarray):
    return left.astype(np.int16) - right.astype(np.int16)


def get_mask(image, median_tube):
    tib = tube2.constrain_to_tube_refwidth(image, median_tube)
    dif = difference(tib, median_tube)
    dif[dif > 0] = 0
    iso = filters.threshold_isodata(dif)
    low = dif < iso
    dil = morphology.dilation(low)
    return dil

videos = glob.glob("data2/**/*.cine",recursive=True)

vid_name = videos[20]

cine = Cine(vid_name)

median_frame = cine.get_video_median()
mtube = tube2.find_tube(median_frame)

for i in range(cine.image_count):
    frame = cine.get_ith_image(i)
    mask = get_mask(frame, mtube)
    imshow(mask)
    pyplot.show()