
from tensorflow import keras
import cine
import tube2
import numpy as np
from skimage import util, transform, measure
import glob
import json

def get_batch(video, idx, batch_size=16):
    zone = []
    for i in range(batch_size):
        frame = video.get_ith_image(idx + i)
        cropped = tube2.tube_crop1(frame)
        shaped = util.img_as_ubyte(transform.resize(cropped, (600, 100)))
        zone.append(shaped[...,np.newaxis])
    return np.array(zone)

def process(videonames, my_model, batch_size=16):
    videos = {}
    for vname in videonames:
        try:
            video = cine.Cine(vname)
            meniscus_pos = []
            for i in range(0, video.image_count - batch_size, batch_size):
                try:
                    batch = get_batch(video, i, batch_size)
                except ValueError as e:
                    #print(e, i, vname)
                    print(e)
                    print(i)
                    print(vname)
                    continue
                predictions = my_model(batch)
                p = predictions.numpy()
                probable = p > 0.5
                for j in range(batch_size):
                    labeled = measure.label(probable[j,:,:,0])
                    rprops = measure.regionprops(labeled, p[j,:,:,0])
                    meniscus = max(rprops, key = lambda reg:reg.area, default=None)
                    if meniscus is not None:
                        row, col = meniscus.centroid
                        mmt = (i + j,
                               float(row),
                               float(col),
                               float(meniscus.intensity_mean),
                               float(meniscus.intensity_max),
                               float(meniscus.intensity_min),
                               float(meniscus.area),
                               float(meniscus.axis_major_length),
                               float(meniscus.axis_minor_length),
                               float(meniscus.equivalent_diameter_area),
                               float(meniscus.feret_diameter_max),
                               float(meniscus.orientation),
                               float(meniscus.perimeter),
                               float(meniscus.eccentricity)
                               )
                        meniscus_pos.append(mmt)
            videos[vname] = meniscus_pos
        finally:
            video.close()
    return videos

def save_measurements(mmts, filename='meniscus_measurements.json'):
    with open(filename, 'w') as mm:
        json.dump(mmts, mm)
    return None

def main():
    videos = glob.glob("data2/**/*.cine", recursive=True)
    model = keras.models.load_model("meniscus_utils/meniscus_track_4")
    data = process(videos, model)
    save_measurements(data)
    return None

if __name__ == '__main__':
    main()
