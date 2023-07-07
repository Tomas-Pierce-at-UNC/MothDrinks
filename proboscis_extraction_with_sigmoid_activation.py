
from tensorflow import keras
import cine
import tube2
import align2
import numpy as np
from skimage import transform, util, io as skio, morphology as morpho
from matplotlib import pyplot
import glob
from skimage import measure, morphology
import json
import itertools

BATCH = 8

model = keras.models.load_model("proboscis_utils/proboscis_model_b3")

def get_batch(video, idx):
    zone = []
    for i in range(BATCH):
        frame = video.get_ith_image(idx + i)
        cropped = tube2.tube_crop1(frame)
        shaped = util.img_as_ubyte(transform.resize(cropped, (600,100)))
        zone.append(shaped[...,np.newaxis])
    zone = np.array(zone)
    return zone

vids = glob.glob("data2/**/*.cine", recursive=True)

videos = {}

for vname in vids:
    print(vname)
    try:
        video = cine.Cine(vname)
        med = video.get_video_median()
        medtube = tube2.tube_crop1(med).astype(np.int16)
        #skio.imshow(medtube)
        #skio.show()
        aligner = align2.SiftAligner(medtube)
        proboscis_pos = []
        for i in range(0, video.image_count - BATCH, BATCH):
            try:
                batch = get_batch(video, i)
            except ValueError as e:
                print(e)
                continue
            predictions = model(batch)
            p_vals = predictions.numpy()
            probable = predictions.numpy() > 0.5
            for j in range(BATCH):
                img = video.get_ith_image(i+j)
                imgtube = tube2.tube_crop1(img)
                try:
                    alignment = aligner.find_transform(imgtube)
                except ValueError as e:
                    print(f'skip {i+j} because {e}')
                    continue
                except RuntimeError as e:
                    print(f'skip {i+j} because {e}')
                    continue
                except Exception as e:
                    print(f'skip {i+j} because {e}')
                    continue
                if alignment is not None:
                    aligned = aligner.apply_transform(imgtube, alignment)
                    dif = aligned.astype(np.int16) - medtube
                else:
                    print(f"skipped {i+j} of {vname}: could not align")
                    continue
                labeled = measure.label(probable[j,:,:,0])
                rprops = measure.regionprops(labeled, p_vals[j,:,:,0])
                difprops = measure.regionprops(labeled, dif)
                labelnames = np.unique(labeled)
                pairs = filter(lambda pair: pair[0].label == pair[1].label, itertools.product(rprops, difprops))
                below = filter(lambda pair : pair[1].intensity_mean < 0, pairs)
                proboscis, _p_zone = max(below, key=lambda pair : pair[0].bbox[2], default=(None, None))
                #proboscis = max(rprops, key=lambda reg : reg.centroid[0], default=None)
                if proboscis is not None:
                    proboscis_pos.append((i + j, # 0
                    *proboscis.centroid, # 1 2
                    float(proboscis.eccentricity), # 3 
                    float(proboscis.orientation), # 4
                    float(proboscis.intensity_mean), # 5
                    float(proboscis.intensity_max), # 6
                    float(proboscis.intensity_min), # 7
                    *proboscis.bbox, # 8 9 10 11 # minrow, mincol, maxrow, maxcol
                    float(proboscis.major_axis_length), # 12
                    #proboscis.area
                    ))
        videos[vname] = proboscis_pos
    except Exception as e:
        with open("log.log", "a") as log:
            log.write(f"failure regarding {vname}: {e}")
        continue
    finally:
        video.close()

with open("proboscis_measurements.json", "w") as mm:
    json.dump(videos, mm)