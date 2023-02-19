
from skimage import filters, morphology as morpho, measure, util
from matplotlib import pyplot
#from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from cine import Cine
import tube2
import align
import numpy as np
import glob
import pathlib
from multiprocessing import Process
import math

def difference(left, right):
    return util.img_as_int(left) - util.img_as_int(right)

def get_width(regionprop) -> float:
    return math.cos(regionprop.orientation) * regionprop.major_axis_length

def locate_meniscus(frame, mediantube, aligner):
    tube = tube2.constrain_to_tube_refwidth(frame, mediantube)
    aligned = aligner.align(tube)
    delta = difference(aligned, mediantube)
    thresh = filters.threshold_isodata(delta)
    low = delta < thresh
    applied = low * delta
    labeled = measure.label(low)
    props = measure.regionprops(labeled, applied)
    try:
        #canidates = filter(lambda prop : (prop.bbox[3] - prop.bbox[1]) > frame.shape[1] // 4, props)
        biggest = max(props, key = lambda prop : prop.bbox[3] - prop.bbox[1])
    except ValueError as e:
        print(e)
        return None
    y,x = biggest.weighted_centroid
    return y

def measure_video_meniscus(vidname: str) -> (np.ndarray, np.ndarray):
    video = Cine(vidname)
    try:
        medframe = video.get_video_median()
        med_tube = tube2.find_tube(medframe)
        #med_tube = tube2.tube_crop1(medframe)
        aligner = align.SiftAligner(med_tube)
        positions = np.array([float('nan')]*video.image_count)
        times = np.zeros(video.image_count)
        for i in range(video.image_count):
            print('.')
            frame = video.get_ith_image(i)
            try:
                loc = locate_meniscus(frame, med_tube, aligner)
                if loc is not None:
                    positions[i] = loc
                times[i] = i
            except Exception as e:
                print(e)
    finally:
        video.close()
    return times, positions

def prep_data(times, positions):
    valids = ~np.isnan(positions)
    x = times[valids]
    pos = positions[valids]
    x = x.reshape(-1, 1)
    return x, pos

def unite_data(x, pos):
    return np.array([x[:,0], pos]).T

def process(name: str):
    stem = pathlib.Path(name).stem
    try:
        times, positions = measure_video_meniscus(name)
        x, pos = prep_data(times, positions)
        united = unite_data(x, pos)
        isofor = IsolationForest()
        isofor.fit(united)
        y_pred = isofor.predict(united)
        pyplot.scatter(united[:,0], united[:,1], c=y_pred, marker='.')
        pyplot.title(stem)
        pyplot.show()
        pyplot.savefig("meniscusTracks/" + stem + ".png")
        pyplot.close()
        return united
    except Exception as e:
        print(e)

if __name__ == '__main__' and False:
    names = glob.glob('data2/*.cine')
    procs = []
    for name in names:
        proc = Process(target=process, args=(name,))
        proc.start()
        procs.append(proc)
    donezo = [False] * len(procs)
    while not all(donezo):
        for i,process in enumerate(procs):
            if process.exitcode is not None:
                donezo[i] = True
    for process in procs:
        process.join()
