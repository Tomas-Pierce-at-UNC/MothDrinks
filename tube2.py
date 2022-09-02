
import numpy as np
import scipy
import skimage
from matplotlib import pyplot
from skimage import filters, morphology as morpho

import cine


def split_zones(image: np.ndarray):
    return [image[:, i:i+100] for i in range(0, image.shape[1], 100)]


def get_low_li(image: np.ndarray):
    li = filters.threshold_li(image)
    return image < li


def get_col_region(image: np.ndarray, colbounds: tuple) -> np.ndarray:
    left, right = colbounds
    sub = image[:, left:right]
    return sub


def select_zones_1(image: np.ndarray):
    m1 = get_low_li(image)
    boundaries = [(i, i+100) for i in range(0, image.shape[1], 100)]
    mzones = split_zones(m1)
    totals = [mz.sum(0) for mz in mzones]
    vspans = [image.shape[0] in tot for tot in totals]
    pos_zones = [
        boundaries[i] for i in range(len(boundaries)) if not vspans[i]
        ]
    return pos_zones


def select_zones_2(image: np.ndarray):
    pos_zones = select_zones_1(image)
    edges = filters.roberts(image)
    iso = filters.threshold_isodata(edges)
    high = edges > iso
    counts = []
    for boundpair in pos_zones:
        region = get_col_region(high, boundpair)
        pcount = region.sum()
        counts.append(pcount)
    which = counts.index(max(counts))
    left, right = pos_zones[which]
    return (left - 25, right + 25)


def get_tallness_histogram(image: np.ndarray):
    verticals = filters.sobel_v(image)
    mags = np.abs(verticals)
    threshold = filters.threshold_isodata(mags)
    mask = mags > threshold
    skel = morpho.skeletonize(mask)
    heights = skel.sum(axis=0)
    colheights = [(i, height) for i, height in enumerate(heights)]
    return np.array(colheights)


def restricted_histogram(image: np.ndarray):
    left, right = select_zones_2(image)
    tallness = get_tallness_histogram(image)
    restricted = tallness[left:right]
    return restricted


def get_bounds_from_histogram(histogram: np.ndarray):
    avg = histogram.mean(0)[1]
    for lcol, count in histogram:
        if count > avg:
            break
    for rcol, count in reversed(histogram):
        if count > avg:
            break
    return (lcol, rcol)


def get_bounds_old(image: np.ndarray):
    reshist = restricted_histogram(image)
    left, right = get_bounds_from_histogram(reshist)
    return left - 5, right + 5


def restrict_to_bounds(image: np.ndarray, bounds: tuple):
    left, right = bounds
    return image[:, left:right]


def isolate_stand(image: np.ndarray):
    low = image - image.mean()
    gaussed = skimage.filters.gaussian(low)
    frame = np.zeros(image.shape, dtype=np.int32)
    for i in range(gaussed.shape[1]):
        col = gaussed[:, i]
        if all((val < 0 for val in col)):
            frame[:, i] = 1

    tubular = frame.astype(bool)
    return tubular


def get_stand_bounds(stand_mask: np.ndarray):
    if stand_mask.sum() == 0:
        return None
    totals = stand_mask.sum(axis=0)
    for left, total in enumerate(totals):
        if total > 0:
            break
    right = left
    for r, total in enumerate(totals):
        if total > 0:
            right = r
    return left, right


def find_stand(image: np.ndarray):
    stand_mask = isolate_stand(image)
    bounds = get_stand_bounds(stand_mask)
    return bounds


def is_left(bounds, middle):
    return bounds[1] < middle


def is_right(bounds, middle):
    return bounds[0] > middle


def isolate_verticals(image: np.ndarray) -> np.ndarray:
    lines = skimage.filters.scharr_v(image)
    mags = abs(lines)
    iso = skimage.filters.threshold_isodata(mags)
    mask = mags > iso
    skel = skimage.morphology.skeletonize(mask)
    return skel


def find_tube_bounds(image: np.ndarray):
    stand_bounds = find_stand(image)
    middle = image.shape[1] // 2
    vert_mask = isolate_verticals(image)
    col_totals = vert_mask.sum(axis=0)
    if stand_bounds is None:
        return get_bounds_old(image)
    elif is_left(stand_bounds, middle):
        first_col = stand_bounds[1] + 5
        restricted = image[:, first_col:]
        bounds = get_bounds_old(restricted)
        left, right = bounds
        left = left + first_col
        right = right + first_col
        return left, right
    elif is_right(stand_bounds, middle):
        last_col = stand_bounds[0] - 5
        restricted = image[:, :last_col]
        return get_bounds_old(restricted)
    else:
        raise Exception("Not sure what to do if stand in middle")


if __name__ == '__main__':
    EX1 = "data/moth23_2022-02-14_Cine1.cine"
    EX2 = "data/moth22_2022_02_09_bad_Cine1.cine"
    EX3 = "data/moth26_2022-02-15_freeflight.cine"
    EX4 = "data/moth23_2022-02-15_Cine1.cine"

