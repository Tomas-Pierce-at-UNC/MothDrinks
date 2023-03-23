from xml.etree import ElementTree
from os import path
import re
import json
from tensorflow.keras import utils
from collections import OrderedDict
import tensorflow as tf
from tensorflow.data import Dataset

_base = path.dirname(__file__)
FILENAME = _base + "/CVATdata/annotations.xml"
FILENAME2 = _base + "/CVATdata/images"

def split_coord_string(s):
    pairstrs = s.split(';')
    strpairs = list(map(lambda pair : pair.split(','), pairstrs))
    coordpairs = list(map(lambda pair : [float(pair[0]),float(pair[1])], strpairs))
    return coordpairs


def get_annotations(filename=FILENAME):

    with open(filename) as handle:
        text = handle.read()

    mytree = ElementTree.fromstring(text)

    images = mytree.findall('image')

    image_annotations = []

    for image in images:
        imagedata = dict(image.items())
        
        # we know there's only going to be one meniscus
        box = image.find("box")
        mendata = dict(box.items())
        mendata['xtl'] = float(mendata['xtl'])
        mendata['ytl'] = float(mendata['ytl'])
        mendata['xbr'] = float(mendata['xbr'])
        mendata['ybr'] = float(mendata['ybr'])
        imagedata["Meniscus"] = mendata
        points = image.findall("points")
        for point in points:
            pointdata = dict(point.items())
            pointdata['coords'] = split_coord_string(pointdata['points'])[0]
            name = pointdata['label']
            imagedata[name] = pointdata
        imagedata['Proboscis'] = []
        polylines = image.findall("polyline")
        for poly in polylines:
            polydata = dict(poly.items())
            pstr = polydata['points']
            coord_pairs = split_coord_string(pstr)
            polydata['points'] = coord_pairs
            imagedata['Proboscis'].append(polydata)

        image_annotations.append(imagedata)

    # I want things sorted to line up with images
    image_annotations.sort(key = lambda d : d['name'])
    return image_annotations

def get_meniscus_boxes(annotations):
    menisci = OrderedDict()
    for image in annotations:
        menisci[image['name']] = image['Meniscus']
    return menisci

def meniscus_box_as_tensor(note):
    return tf.constant([note['xtl'], note['ytl'], note['xbr'], note['ybr']])

def get_meniscus_centers(annotations):
    menis_centers = OrderedDict()
    for image in annotations:
        menis_centers[image['name']] = image['MeniscusCenter']
        
    return menis_centers

def get_proboscis_tips(annotations):
    tips = OrderedDict()
    for image in annotations:
        if 'ProboscisTip' in image:
            tips[image['name']] = image['ProboscisTip']
    return tips

def load_images(foldername=FILENAME2):
    return utils.image_dataset_from_directory(
        foldername,
        labels=None,
        shuffle=False, # I want things sorted to line up with annotations
        batch_size=1 # disabling batching for debug reasons
        )

def create_box_dataset(annotations):
    menisci = get_meniscus_boxes(annotations)
    g = [meniscus_box_as_tensor(box) for box in menisci.values()]
    return Dataset.from_tensor_slices(g)
    
if __name__ == '__main__':
    annotations = get_annotations(FILENAME)
    images = load_images()
    bboxes = create_box_dataset(annotations)
    mydata = tf.data.Dataset.zip((images,bboxes))
