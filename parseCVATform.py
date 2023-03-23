from xml.etree import ElementTree
from os import path
import re
import json
from tensorflow.keras import utils

FILENAME = path.dirname(__file__) + "/CVATdata/annotations.xml"


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
    image_annotations.sort(key = lambda d : d['name'])
    return image_annotations

def get_meniscus_boxes(annotations):
    menisci = {}
    for image in annotations:
        menisci[image['name']] = image['Meniscus']
    return menisci

def get_meniscus_centers(annotations):
    menis_centers = {}
    for image in annotations:
        menis_centers[image['name']] = image['MeniscusCenter']
    return menis_centers

def get_proboscis_tips(annotations):
    tips = {}
    for image in annotations:
        if 'ProboscisTip' in image:
            tips[image['name']] = image['ProboscisTip']
    return tips



def load_images(foldername):
    return utils.image_dataset_from_directory(
        foldername,
        labels=None,
        shuffle=False
        )
