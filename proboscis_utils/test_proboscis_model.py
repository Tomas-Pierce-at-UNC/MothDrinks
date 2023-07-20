
from tensorflow import keras
from skimage.io import imread
import train_proboscis_model as util
import numpy as np
from matplotlib import pyplot
import seaborn

def organize_data_partlabel(img_names, proboscis_coords):
    input_images = []
    output_images = []
    for name in img_names:
        if name in proboscis_coords:
            img = imread(name)
            input_images.append(img)
            plines = proboscis_coords[name]
            mask = util.create_mask(plines)
            output_images.append(mask)
    input_data = np.array(input_images)
    input_data = input_data[..., np.newaxis]
    output_data = np.array(output_images)
    output_data = output_data[..., np.newaxis]
    return input_data, output_data

def measure_IoU_on_test_set():
    annote = util.get_annotations("proboscis_annotations_big.xml")
    proboscis_labels_coord = util.extract_all_paths(annote)
    img_names = util.get_image_names("big/*")
    indata, outdata = organize_data_partlabel(img_names, proboscis_labels_coord)
    model = keras.models.load_model("proboscis_model_b7")
    predictions = model.predict(indata)
    high = predictions > 0.5
    measurements = []
    for i in range(len(high)):
        actual = outdata[i]
        predicted = high[i]
        intersection = actual & predicted
        union = actual | predicted
        intersect_count = intersection.sum()
        union_count = union.sum()
        jaccard = intersect_count / union_count
        measurements.append(jaccard)
    return measurements

if __name__ == '__main__':
    iou = measure_IoU_on_test_set()
    seaborn.histplot(iou)
    #pyplot.title("Proboscis Tracking Intersection Over Union")
    pyplot.xlabel("Insersection over Union")
    pyplot.show()
    
