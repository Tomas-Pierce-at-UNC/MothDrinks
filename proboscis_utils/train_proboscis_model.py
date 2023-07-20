
import os
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/usr/lib/cuda'

import model_arch
import data_augment
from xml.etree import ElementTree
import glob
from skimage.io import imread
from skimage import draw, morphology, transform, util
import numpy as np
from tensorflow import keras
import tensorflow as tf
import typing
from skimage.io import imshow
from matplotlib import pyplot

WIDTH = 100
HEIGHT = 600
BATCH_SIZE=8

print(tf.config.list_physical_devices())
tf.keras.backend.set_image_data_format('channels_last')

def get_annotations(filename) -> ElementTree.ElementTree:
    et = ElementTree.parse(filename)
    return et

def get_image_names(folder_pattern) -> typing.List[str]:
    image_names = sorted(glob.glob(folder_pattern))
    return image_names
    
def extract_points(polyline):
    pstr = polyline.attrib['points']
    pairs = pstr.split(';')
    points = []
    for pair in pairs:
        xstr, ystr = pair.split(',')
        x = float(xstr)
        y = float(ystr)
        points.append((x,y))
    return points

def extract_polylines(image_elem):
    polyline_elems = image_elem.findall('polyline')
    polylines = []
    for elem in polyline_elems:
        points = extract_points(elem)
        polylines.append(points)
    return polylines

def extract_all_paths(annotation):
    paths = {}
    image_elems = annotation.findall("image")
    for element in image_elems:
        name = element.attrib['name']
        polylines = extract_polylines(element)
        paths[name] = polylines
    return paths

def create_mask(polylines):
    blank = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for pline in polylines:
        for i in range(len(pline) - 1):
            start_x, start_y = pline[i]
            end_x, end_y = pline[i + 1]
            # enforcing bounds
            start_x = round(min(max(0, start_x), WIDTH - 1))
            start_y = round(min(max(0, start_y), HEIGHT - 1))
            end_x = round(min(max(0, end_x), WIDTH - 1))
            end_y = round(min(max(0, end_y), HEIGHT - 1))
            # bounds have been enforced
            line = draw.line(start_y, start_x, end_y, end_x)
            blank[line] = True
    # improves visual strength and better perceptual representation
    blank = morphology.dilation(blank)
    return blank

def organize_data(image_names, proboscis_coords):
    input_images = []
    output_images = []
    for name in image_names:
        img = imread(name)
        input_images.append(img)
        plines = proboscis_coords[name]
        mask = create_mask(plines)
        output_images.append(mask)
    input_data = np.array(input_images)
    input_data = input_data[..., np.newaxis]
    output_data = np.array(output_images)
    output_data = output_data[..., np.newaxis]
    return input_data, output_data
    
def load_training():
    annotations = get_annotations("trainInPlabels_annotations.xml")
    proboscis_labels_coord = extract_all_paths(annotations)
    image_names = get_image_names("trainInP/*")
    indata, outdata = organize_data(image_names, proboscis_labels_coord)
    imseq = data_augment.ImageSeq(indata, outdata, batch=BATCH_SIZE)
    rep = data_augment.RepeatSeq(imseq, 20)
    rf = data_augment.RandomFlip(rep)
    shift1 = data_augment.RandomXShift(rf)
    augmented = data_augment.RandomYShift(shift1)
    return augmented

def train():
    model = model_arch.our_model()
    data = load_training()
    print(model.summary())
    model.fit(data, epochs=40)
    model.save("proboscis_model_b7")
    
if __name__ == '__main__':
    train()
