
import align
import tube2
import cine

from skimage import filters, io as skio, measure, util, exposure
import numpy as np
from matplotlib import pyplot


def find_height_widest_object(mask: np.ndarray, intensity: np.ndarray) -> float:
    labeled = measure.label(mask)
    reg_props = measure.regionprops(labeled, intensity)

    def wideness(region):
        mincol = region.bbox[1]
        maxcol = region.bbox[3]
        return maxcol - mincol

    widest_object = max(reg_props, key=wideness)
    position = widest_object.centroid_weighted
    return position[0]


def generate_masks_pairs(video: cine.Cine):
    med = video.get_video_median()
    tube_bounds = tube2.get_tube(med)
    restricted_med = tube2.apply_bounds(med, tube_bounds)
    aligner = align.Aligner(restricted_med)
    width = tube_bounds[1] - tube_bounds[0]
    retyped_med = restricted_med.astype(np.int16)
    for i in range(video.image_count):
        img = video.get_ith_image(i)
        aligned_img = aligner.align(img)
        restricted_img = aligned_img[:, :width]
        retyped_img = restricted_img.astype(np.int16)
        delta = retyped_img - retyped_med
        delta[delta > 0] = 0
        iso = filters.threshold_isodata(delta)
        low = delta < iso
        yield low, delta


if __name__ == '__main__':
    positions = []
    cin = cine.Cine("data/moth23_2022-02-14_Cine1.cine")
    i = 0
    for mask, delta in generate_masks_pairs(cin):
        i += 1
        if i == 200:
            skio.imshow(mask)
            pyplot.show()
    cin.close()
    pyplot.plot(positions)
