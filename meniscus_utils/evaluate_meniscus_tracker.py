
from tensorflow.keras.models import load_model
import data_loading as data
import numpy as np
from skimage import measure
import math
from matplotlib import pyplot
import seaborn

BATCH = 16

X_test, y_test = data.load_dset("annotations_big.xml", "big")
X_test = X_test[...,np.newaxis]
y_test = y_test[..., np.newaxis]

extant_model = load_model("meniscus_track_4")

#evals = extant_model.evaluate(X_test, y_test, verbose=1)

correct_count_at = []
false_negative_at = []
false_positive_at = []
euclid_distances = []
manhattan_distances = []
iou = []

for i in range(0, len(X_test)-BATCH, BATCH):
    batch = X_test[i:i+BATCH]
    preds = extant_model(batch)
    p = preds.numpy()
    for j in range(BATCH):
        idx = i + j
        myprediction = p[j] > 0.5
        actual = y_test[idx]
        intersection = myprediction & actual
        union = myprediction | actual
        jaccard = intersection.sum() / union.sum()
        iou.append(jaccard)
        ylabel = measure.label(y_test[idx,:,:,0])
        y_regions = measure.regionprops(ylabel)
        high = p[j,:,:,0] > 0.5
        plabel = measure.label(high)
        p_regions = measure.regionprops(plabel)
        if len(y_regions) == len(p_regions):
            correct_count_at.append(idx)
        elif len(y_regions) > len(p_regions):
            false_negative_at.append(idx)
        elif len(y_regions) < len(p_regions):
            false_positive_at.append(idx)
        for y_reg in y_regions:
            for p_reg in p_regions:
                yrow, ycol = y_reg.centroid
                prow, pcol = y_reg.centroid
                euclid = math.sqrt(((yrow - prow)**2) + ((ycol - pcol)**2))
                euclid_distances.append(euclid)
                manhattan = abs(yrow - prow) + abs(ycol - pcol)
                manhattan_distances.append(manhattan)
            
seaborn.histplot(iou)
#pyplot.title("Meniscus Tracking Intersection Over Union")
pyplot.xlabel("Intersection over Union")
pyplot.show()
