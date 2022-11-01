
from cine import Cine
import align
import tube2
from skimage import filters, measure, feature, morphology
from skimage.io import imshow
from matplotlib import pyplot
import numpy as np


def difference(left: np.ndarray, right: np.ndarray):
    return left.astype(np.int16) - right.astype(np.int16)


def mask(frame: np.ndarray, median: np.ndarray):
    aligner = align.SiftAligner(median)
    al_frame = aligner.align(frame)
    delta = difference(al_frame, median)
    tb = tube2.get_tube(al_frame)
    res_delta = tube2.apply_bounds(delta, tb)
    rd = res_delta[200:]
    s_flip = -rd
    s_flip[s_flip < 0] = 0
    li_t = filters.threshold_li(s_flip)
    above = s_flip > li_t
    eroded = morphology.binary_erosion(above)
    return eroded


def find_proboscis_region(mask: np.ndarray):
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    longest = max(regions, key=lambda r: r.feret_diameter_max)
    return longest


def get_lowest_point(region):
    return region.bbox[2]


if __name__ == '__main__':
    vidname = "data2/moth23_2022-02-11_a.cine"
    # vid = Cine(vidname)
    vid2 = Cine("data2/mothM6_2022-09-27_Cine1.cine")
    med = vid2.get_video_median()
    imagecount = vid2.image_count

    p_measurements = []

    for i in range(imagecount):
        print(i)
        framei = vid2.get_ith_image(i)
        try:
            mask_i = mask(framei, med)
        except AttribueError:
            continue
        p_reg = find_proboscis_region(mask_i)
        lowpoint = get_lowest_point(p_reg)

        p_measurements.append((i, lowpoint))

    prob_mmts = np.array(p_measurements)
