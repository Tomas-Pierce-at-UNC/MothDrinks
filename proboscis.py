
from cine import Cine
import align
import tube2
from skimage import filters, measure, feature, morphology
from skimage.io import imshow
from matplotlib import pyplot
import numpy as np


def difference(left: np.ndarray, right: np.ndarray):
    return left.astype(np.int16) - right.astype(np.int16)

