
from skimage.io import imsave, imshow
from skimage.feature import SIFT, plot_matches, match_descriptors
from skimage.measure import ransac
from skimage.filters import threshold_isodata
from skimage.morphology import binary_erosion

import cine

import tube2

import meniscus

from matplotlib import pyplot

video = cine.Cine("data2/mothM1_2022-09-27_Cine1.cine")

median = video.get_video_median()

z = video.get_ith_image(0)

imsave("FigureMaterials/exampleFrame.png", z)

tb1 = tube2.get_tube(median)
res_med = tube2.apply_bounds(median, tb1)
res_z = tube2.constrain_to_tube_refwidth(z, res_med)

delta = meniscus.difference(res_z, res_med)

de1 = SIFT()
de2 = SIFT()

de1.detect_and_extract(res_med)
de2.detect_and_extract(res_z)

matches = match_descriptors(
    de1.descriptors,
    de2.descriptors,
    cross_check=True
    )

# ref_matches = de1.keypoints[matches[:, 0]]
# matches = de2.keypoints[matches[:, 1]]

# fig, ax = pyplot.subplots()
# plot_matches(ax, res_med, res_z, de1.keypoints, de2.keypoints, matches)
# pyplot.show()

# imshow(delta)
# pyplot.show()

imsave("FigureMaterials/medianClipped.png", res_med)
imsave("FigureMaterials/exampleMedian.png", median)
imsave("FigureMaterials/exampleClipped.png", res_z)
imsave("FigureMaterials/exampleDelta.png", delta)

video.close()

iso = threshold_isodata(delta)

low = delta < iso

eroded = binary_erosion(low)

imsave("FigureMaterials/exampleMask1.png", low)

imsave("FigureMaterials/exampleEroded.png", eroded)