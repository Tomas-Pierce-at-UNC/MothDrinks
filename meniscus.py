from cine import Cine
import tube2
import align

from skimage import filters, feature, exposure, measure, util, restoration, morphology
from skimage.io import imshow
from matplotlib import pyplot
import numpy as np
import time
from cine_profiler import print_time

from skimage.io import imshow
from matplotlib import pyplot


def difference(left: np.ndarray, right: np.ndarray):
    return left.astype(np.int16) - right.astype(np.int16)


def get_deltas_from_median(video: Cine, start_frame: int, end_frame: int):
    if end_frame >= video.image_count:
        end_frame = video.image_count - 1
    count = end_frame - start_frame
    video_median = video.get_restricted_video_median(start_frame, count)
    # video_median = exposure.equalize_hist(video_median)
    video_median = video_median
    vm_tubebounds = tube2.get_tube(video_median)
    width = vm_tubebounds[1] - vm_tubebounds[0]
    res_vid_med = tube2.apply_bounds(video_median, vm_tubebounds)
    # brightened_med = subtract_background(res_vid_med)
    aligner = align.SiftAligner(res_vid_med)
    for i in range(start_frame, end_frame):
        print(i)
        img = video.get_ith_image(i)
        img = img
        # img = exposure.equalize_hist(img)
        restricted_img = tube2.constrain_to_tube_refwidth(img, res_vid_med)
        try:
            aligned_img = aligner.align(restricted_img)
        except Exception as e:
            print(e)
            print("had to skip image", i)
            continue
        delta = difference(aligned_img, res_vid_med)
        delta[delta > 0] = 0
        yield i, delta


def find_blobs(delta_img):
    iso = filters.threshold_isodata(delta_img)
    low = delta_img < iso
    blobs = feature.blob_log(low)
    blobs = blobs[blobs[:, 2] > 1.0]
    return blobs


def get_blobs_from_deltas(deltas):
    blob_seq = []
    for i, delt in deltas:
        blobs = find_blobs(delt)
        for blob in blobs:
            row, col, sigma = blob
            # project specific threshold on sizing
            if 6 <= sigma <= 7:
                blob_seq.append((i, row, col, sigma))
        print(i)
    blobs_array = np.array(blob_seq)
    return blobs_array


def get_widest_from_delta(delta: np.ndarray, i):
    width = delta.shape[0]
    middle = width / 2
    iso = filters.threshold_isodata(delta)
    low = delta < iso
    ero = morphology.binary_erosion(low)
    labeled = measure.label(ero)
    regions = measure.regionprops(labeled, intensity_image=delta)
    not_wider = filter(
        lambda region: region.bbox[3] - region.bbox[1] < width,
        regions
        )
    widest = max(not_wider, key=lambda r: r.bbox[3] - r.bbox[1])
    center = widest.centroid_weighted
    return (i,
            center[0],
            center[1],
            center[1] - middle,
            # *widest.bbox,
            widest.area,
            widest.area_bbox,
            widest.area_convex,
            widest.area_filled,
            widest.axis_major_length,
            widest.axis_minor_length,
            # *widest.centroid,
            widest.eccentricity,
            widest.equivalent_diameter_area,
            widest.euler_number,
            widest.extent,
            widest.feret_diameter_max,
            widest.intensity_max,
            widest.intensity_mean,
            widest.intensity_min,
            widest.label,
            widest.orientation,
            widest.perimeter,
            widest.perimeter_crofton,
            widest.solidity,
            widest.centroid[0],
            widest.centroid[1]
            )


def get_widest_blobs_from_deltas(deltas):
    blob_seq = []
    for i, delt in deltas:
        widest = get_widest_from_delta(delt, i)
        blob_seq.append(widest)
    return blob_seq


def find_meniscus(video: Cine, start_frame: int, end_frame: int) -> np.ndarray:
    if start_frame < 0 or start_frame is None:
        start_frame = 0
    if end_frame > video.image_count - 1 or end_frame is None:
        end_frame = video.image_count - 1
    deltas = get_deltas_from_median(video, start_frame, end_frame)
    widests = get_widest_blobs_from_deltas(deltas)
    arr = np.array(widests)
    return arr


def additive_measure_video(video: Cine, filename: str=None, start_frame: int=0, end_frame: int=None):
    if end_frame is None:
        end_frame = video.image_count - 1
    if filename is None:
        stem = video.filename[6:-5]
        out = "men_data/" + stem + ".tsv"
    deltas = get_deltas_from_median(video, start_frame, end_frame)
    for i, delta in deltas:
        widest = get_widest_from_delta(delta, i)
        with open(filename, 'a') as tablefile:
            tablefile.write("\t".join(map(str, widest)) + "\n")


def graph_vid(video="data2/mothM6_2022-09-22_Cine1.cine"):
    # vid = Cine("data2/mothM3_2022-09-22_Cine2.cine")
    vid = Cine(video)
    arr = find_meniscus(vid, 0, vid.image_count - 1)
    pyplot.scatter(x=arr[:, 0], y=arr[:, 1])
    pyplot.show()
    vid.close()


def tabulate_vid(video="data2/mothM6_2022-09-22_Cine1.cine"):
    stem = video[6:-5]
    out = "chopped_men_data/" + stem + ".tsv"
    vid = Cine(video)
    arr = find_meniscus(vid, 0, vid.image_count - 1)
    print(arr.shape)
    np.savetxt(out, arr, delimiter='\t')
    vid.close()
    return arr


if __name__ == '__main__':
    import glob
    video_names = glob.glob("data2/*.cine")
    video_names.sort()
    tables = []
    for name in video_names:
        try:
            table = tabulate_vid(name)
            tables.append(table)
        except Exception as e:
            print(e)
            with open("skipped.txt", "a") as skips:
                skips.write(name + '\n')
    for i, table in enumerate(tables):
        pyplot.scatter(x=table[:, 0], y=arr[:, 1])
        pyplot.errorbar(x=table[:, 0],
                        y=arr[:, 1],
                        yerr=arr[:, 9] / arr[:, 9].max()
                        )
        pyplot.title(video_names[i])
        pyplot.xlabel("Frame")
        pyplot.ylabel("Row")
        pyplot.show()
