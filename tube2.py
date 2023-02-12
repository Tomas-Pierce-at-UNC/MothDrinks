
import numpy as np
import scipy
from matplotlib import pyplot
from skimage import filters, morphology as morpho, io as skio
import cine

def get_horiz_edges(image: np.ndarray) -> np.ndarray:
    edges = filters.sobel_h(image)
    mag_edges = np.abs(edges)
    li = filters.threshold_li(mag_edges)
    horiz = mag_edges > li
    totals = horiz.sum(axis=0)
    return totals

def tube_crop1(image: np.ndarray) -> np.ndarray:
    edgetotals = get_horiz_edges(image)
    peak = edgetotals.argmax()
    return image[:,peak - 50: peak + 50]

def graph_edges(edges: np.ndarray):
    pyplot.bar(list(range(len(edges))), edges)
    pyplot.show()

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
    # handle case of stand on edge explicitly
    if stand[1] >= image.shape[1]: 
        return True
    if stand[0] <= 0:
        return False
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
    area between them contains the artifical flower."""
    stand = find_stand(image)
    if stand is None:
        # glorious lack of visible vertical stand
        verts = isolate_verticals(image)
        cols = np.abs(verts).sum(axis=0)
        #cols = get_vert_edge_mags(image)
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


def constrain_to_tube_refwidth(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    '''Finds a subregion of target that contains
    the tube with the same width as the reference image'''
    left, _r = get_tube(target)
    width = ref.shape[1]
    right = left + width
    return apply_bounds(target, (left, right))

def find_tube(image: np.ndarray) -> np.ndarray:
    """locates the artifial flower in vertical position in an image by a
    two step process and returns an array cropped to that region"""
    cut = tube_crop1(image)
    cutbounds = get_tube(cut)
    return apply_bounds(cut, cutbounds)

if __name__ == '__main__':
    import glob
    import random
    cines = glob.glob("data2/*.cine")
    cinname = random.choice(cines)
    cinehandle = cine.Cine(cinname)
    frame = random.randint(0, cinehandle.image_count - 1)
    img = cinehandle.get_ith_image(frame)
    cinehandle.close()
    tube = find_tube(img)
    skio.imshow(tube)
    pyplot.show()
    skio.imshow(img)
    pyplot.show()
