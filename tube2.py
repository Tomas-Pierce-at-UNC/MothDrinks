
import numpy as np
import scipy
from matplotlib import pyplot
from skimage import filters, morphology as morpho, io as skio

import cine


def lowli(image: np.ndarray) -> np.ndarray:
    """Uses Li's threshold to find dark objects on light background,
    producing mask where dark objects are now
    True and thus displayed as white."""
    li = filters.threshold_li(image)
    return image < li


def find_stand(image: np.ndarray) -> tuple:
    """Locates the ringstand base that supports the
    moth's perch if it is presenct in the image and return it
    as a tuple of (left col, right col) coordinates. Otherwise,
    return None."""
    low = lowli(image)
    cols = low.sum(axis=0)
    left = None
    for i, col in enumerate(cols):
        if col == image.shape[0]:
            left = i
            break
    if left is None:
        return None
    right = None
    for j in range(left, len(cols)):
        column = cols[j]
        if column < image.shape[0]:
            right = j
            break
    if left is not None and right is not None:
        # deliberately overestimate ringstand
        # width to compensate for intrinsic
        # behavior of Li's threshold.
        return (left - 5, right + 5)
    else:
        return None


def flower_is_left(image: np.ndarray, stand: tuple) -> bool:
    """Returns whether the flower is on the left of the
    perch stand. Assumes the ringstand is present in the image."""
    left = image[:, :stand[0]]
    right = image[:, stand[1]:]
    # Because the images are set up to have a uniform backing light sheet,
    # the presence of the flower causes more complexity than the background.
    # the other side will be essentially uniform. This lets us use the
    # standard deviation as a way to detect which side the flower is on.
    return left.std() > right.std()


def isolate_verticals(image: np.ndarray) -> np.ndarray:
    """Creates a mask in which vertical edges of
    the input image are emphasized."""
    edges_v = filters.sobel_v(image)
    iso = filters.threshold_isodata(edges_v)
    low = edges_v < iso
    dlow = morpho.dilation(low)
    return dlow


def get_tube(image: np.ndarray) -> tuple:
    """Finds a pair of boundary columns that the
    area between them contains the simulated flower."""
    stand = find_stand(image)
    if stand is None:
        # glorious lack of visible vertical stand
        verts = isolate_verticals(image)
        cols = verts.sum(axis=0)
        lcols = list(cols)
        height_thresh = filters.threshold_isodata(cols)
        tallest = cols.max()
        tall_index = lcols.index(tallest)
        right = 0
        for i, col in enumerate(cols):
            if col > height_thresh:
                right = i
        left = 0
        for i, col in enumerate(cols):
            if col > height_thresh:
                left = i
                break
        between = 0
        for i in range(left+1, right):
            mycol = cols[i]
            if mycol > cols[between]:
                between = i
        lefthand = verts[:, left:between]
        righthand = verts[:, between:right]
        lsum = lefthand.sum()
        rsum = righthand.sum()
        # better to include edges than exclude them in marginal
        # cases so we use the +/- 5 pixels to do that.
        if lsum > rsum:
            return left - 5, between + 5
        elif lsum < rsum:
            return between - 5, right + 5
        else:
            return left - 5, right + 5
    elif flower_is_left(image, stand):
        restricted = image[:, :stand[0]]
        return get_tube(restricted)
    else:  # flower is right
        restricted = image[:, stand[1]:]
        relative = get_tube(restricted)
        left, right = relative
        return left + stand[1], right + stand[1]


def apply_bounds(image: np.ndarray, bounds: tuple) -> np.ndarray:
    return image[:, bounds[0]:bounds[1]]


if __name__ == '__main__':

    import time

    EX1 = "data/moth23_2022-02-14_Cine1.cine"
    EX2 = "data/moth22_2022_02_09_bad_Cine1.cine"
    EX3 = "data/moth26_2022-02-15_freeflight.cine"
    EX4 = "data/moth23_2022-02-15_Cine1.cine"

    cin2 = cine.Cine(EX2)
    med2 = cin2.get_video_median()
    cin2.close()

    cin4 = cine.Cine(EX4)
    med4 = cin4.get_video_median()
    cin4.close()

    cin1 = cine.Cine(EX1)
    med1 = cin1.get_video_median()
    h = cin1.get_ith_image(200)
    cin1.close()

    cin3 = cine.Cine(EX3)
    med3 = cin3.get_video_median()
    cin3.close()

    t1 = time.time()
    tb1 = get_tube(med1)
    t2 = time.time()
    tb2 = get_tube(med2)
    t3 = time.time()
    tb3 = get_tube(med3)
    t4 = time.time()
    tb4 = get_tube(med4)
    t5 = time.time()

    res1 = med1[:, tb1[0]:tb1[1]]
    res2 = med2[:, tb2[0]:tb2[1]]
    res3 = med3[:, tb3[0]:tb3[1]]
    res4 = med4[:, tb4[0]:tb4[1]]

    ht = get_tube(h)
    h_res = apply_bounds(h, ht)
