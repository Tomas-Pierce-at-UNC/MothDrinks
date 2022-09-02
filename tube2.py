
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


if __name__ == '__main__':

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
    cin1.close()

    cin3 = cine.Cine(EX3)
    med3 = cin3.get_video_median()
    cin3.close()
