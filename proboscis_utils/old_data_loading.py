
from xml.etree import ElementTree
import glob
from tensorflow import keras
import tensorflow as tf
import sklearn
from skimage import draw
from skimage.io import imshow, imread
from skimage import morphology
from skimage import transform
from skimage import util
import numpy as np
from matplotlib import pyplot
# from tensorflow import keras
# import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model
from tensorflow.keras.utils import plot_model
import typing
tf.keras.backend.set_image_data_format('channels_last')


def get_annotations(filename="proboscis_annotations_big.xml") -> ElementTree.ElementTree:
    et = ElementTree.parse(filename)
    return et


def get_image_names(folder_pattern="big/*.png") -> typing.List[str]:
    image_names = sorted(glob.glob(folder_pattern))
    return image_names


def get_output_masks_images(et: ElementTree.ElementTree
                            ) -> typing.Dict[str, np.ndarray]:
    masks = {}
    for img in et.findall('image'):
        width = int(img.attrib['width']) + 1
        height = int(img.attrib['height']) + 1
        blank = np.zeros(shape=(height, width), dtype=np.uint8)
        polys = img.findall('polyline')
        for poly in polys:
            attribs = poly.attrib
            pstr = attribs['points']
            pairs = pstr.split(';')
            coord_list = []
            for pair in pairs:
                xstr, ystr = pair.split(',')
                x = float(xstr)
                y = float(ystr)
                coord_list.append((x, y))
            for i in range(len(coord_list) - 1):
                start = coord_list[i]
                end = coord_list[i + 1]
                line = draw.line(round(start[1]), round(
                    start[0]), round(end[1]), round(end[0]))
                blank[line] = np.uint8(255)
        masks[img.attrib['name']] = blank > 0
    return masks


def load_images(image_names: typing.List[str]) -> typing.Dict[str, np.ndarray]:
    images = {}
    for name in image_names:
        data = imread(name)
        images[name] = data
    return images


def get_short_names(image_names: typing.List[str]) -> typing.List[str]:
    names = [image_name[-22:] for image_name in image_names]
    return names


def pair_input_output(names: typing.List[str],  # short names
                      image_names: typing.List[str],  # long names
                      masks: typing.Dict[str, np.ndarray],
                      images: typing.Dict[str, np.ndarray]
                      ) -> typing.List[typing.Tuple[np.ndarray, np.ndarray]]:
    pairs = []
    cross = zip(names, image_names)
    for short_name, long_name in cross:
        if short_name in masks:
            mask = masks[short_name]
            trainee = images[long_name]
            pairs.append((trainee, morphology.binary_dilation(mask[:-1, :-1])))
        else:
            trainee = images[long_name]
            mask = np.zeros(shape=trainee.shape)
            pairs.append((trainee, mask))
    return pairs


def filter_unlabeled(pairs: typing.List[typing.Tuple[np.ndarray, np.ndarray]]) -> typing.List[typing.Tuple[np.ndarray, np.ndarray]]:
    with_probs = filter(lambda pair: pair[1].sum() > 0, pairs)
    labeled_pairs = list(with_probs)
    return labeled_pairs


def split_pairs(labeled_pairs: typing.List[typing.Tuple[np.ndarray, np.ndarray]]) -> typing.Tuple[np.ndarray, np.ndarray]:
    inputs = []
    outputs = []
    for my_in, my_out in labeled_pairs:
        inputs.append(my_in)
        outputs.append(my_out)

    ins = np.array(inputs)
    outs = np.array(outputs)
    return ins, outs


def add_dummy_channel_axis(ins: np.ndarray, outs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    input_data = ins[..., np.newaxis]
    output_data = outs[..., np.newaxis]
    return input_data, output_data


def remask(out_data: np.ndarray) -> np.ndarray:
    return out_data != 0


def load_data():
    annote: ElementTree.ElementTree = get_annotations()
    image_names: list = get_image_names()
    masks: dict = get_output_masks_images(annote)
    images: dict = load_images(image_names)
    short_names: list = get_short_names(image_names)
    pairs: list = pair_input_output(short_names, image_names, masks, images)
    labeled = filter_unlabeled(pairs)
    ins, outs = split_pairs(labeled)
    inp_data, out_data = add_dummy_channel_axis(ins, outs)
    out_data = remask(out_data)
    return inp_data, out_data
