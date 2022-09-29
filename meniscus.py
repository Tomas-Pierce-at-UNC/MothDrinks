
import cine
import align
import tube2
from skimage import filters
from skimage import io as skio
from skimage import morphology as morpho
from skimage import exposure
from skimage import feature
from skimage import measure
from matplotlib import pyplot
import numpy as np
import math


def isolate(filename: str, start=0, end=None):
    try:
        video = cine.Cine(filename)
        median = video.get_video_median()
        bounds = tube2.get_tube(median)
        print(bounds)
        aligner = align.Aligner(median)
        if end is None:
            end = video.image_count
        for i in range(start, end):
            frame = video.get_ith_image(i)
            aligned = aligner.align(frame)
            delta = aligned.astype(np.int16) - median.astype(np.int16)
            restricted_delta = tube2.apply_bounds(delta, bounds)
            restricted_delta[restricted_delta > 0] = 0
            iso = filters.threshold_isodata(restricted_delta)
            low = restricted_delta < (iso * 1.5)
            yield low
    finally:
        video.close()


def get_canidates(mask: np.ndarray):
    blobs = feature.blob_log(mask)
    canidates = blobs[blobs[:, 2] > 1]
    return canidates


def get_canidate_masks(canidates: np.ndarray, scene: np.ndarray) -> list:
    masks = []
    # want to remove holes so that a off pixel surrounded by on pixels
    # gets treated as being an actual point
    myscene = morpho.closing(scene)
    labels = measure.label(myscene)
    for canidate in canidates:
        row, col, sigma = canidate
        tag = labels[int(row), int(col)]
        if tag == 0:
            continue
        submask = labels == tag
        masks.append(submask)
    return masks


def get_width(region):
    major = region.axis_major_length
    orient = region.orientation
    return math.cos(orient) * major


def select(canidate_masks):
    if len(canidate_masks) == 1:
        return canidate_masks[0]
    else:
        widest = -1
        widemask = None
        for i, mask in enumerate(canidate_masks):
            regionprops = measure.regionprops(mask.astype(np.uint8))[0]
            width = get_width(regionprops)
            wideness = abs(width)
            if wideness > widest:
                widest = wideness
                widemask = mask
        return widemask


def find_meniscus_row(meniscus_mask):
    img = meniscus_mask.astype(np.uint8)
    props = measure.regionprops(img)[0]
    centroid = props.centroid
    return centroid[0]


def measure_meniscus_in_video(filename, start=0, end=None):
    isolateds = isolate(filename, start, end)
    i = start
    for mask in isolateds:
        i += 1
        canidates = get_canidates(mask)
        can_masks = get_canidate_masks(canidates, mask)
        meniscus_mask = select(can_masks)
        meniscus_row = find_meniscus_row(meniscus_mask)
        yield meniscus_row


def get_meniscus(filename: str, start: int = 0, end: int = None):
    rows = measure_meniscus_in_video(filename, start, end)
    mens = []
    for row_coord in rows:
        print(row_coord)
        mens.append(row_coord)
    mens = np.array(mens)
    if end is None:
        xs = np.arange(0, len(mens))
    else:
        xs = np.arange(start, end)
    return xs, mens


if __name__ == '__main__':
    filename1 = "data/moth23_2022-02-09_meh.cine"
    xs1, mens1 = get_meniscus(filename1, 2131, 2431)
    pyplot.scatter(xs1, mens1, marker='.')
    pyplot.show()
