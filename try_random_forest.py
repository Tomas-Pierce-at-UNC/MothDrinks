import glob
import os
import skimage.io as skio
from skimage.transform import resize
from sklearn import pipeline, preprocessing
from sklearn.ensemble import RandomForestRegressor
import loadTrainData as loader
from matplotlib import pyplot
import numpy as np

FOLDER = os.path.dirname(__file__) + "/CVATdata/images/training"
image_names = glob.glob(FOLDER + "/*.png")
image_names.sort()
images = [skio.imread(name) for name in image_names]
shapes = [image.shape for image in images]
my_imgs = [resize(image, (600, 50)) for image in images]
v_imgs = [img.reshape((img.shape[0] * img.shape[1],)) for img in my_imgs]
annotations = loader.get_annotations()
centers = loader.get_meniscus_centers(annotations)


center_coords = np.array(
    [np.array(centers[name[64:]]['coords']) for name in image_names]
    )

for i, geoms in enumerate(shapes):
    height, width = geoms
    coords = center_coords[i]
    x_raw,y_raw = coords
    x_fraction = x_raw / width
    y_fraction = y_raw / height
    x_transformed = x_fraction * 50
    y_transformed = y_fraction * 600
    center_coords[i][0] = x_transformed
    center_coords[i][1] = y_transformed

clf = pipeline.make_pipeline(
    preprocessing.StandardScaler(),
    RandomForestRegressor(n_estimators=20)
    )

clf.fit(v_imgs, center_coords[:,1])

test_names = glob.glob("/home/tomas/Projects/DrinkMoth2/test/*.tif")
test_names.sort()
test_imgs = [skio.imread(name) for name in test_names]
test_shapes = [image.shape for image in test_imgs]
my_test_imgs = [resize(image, (600, 50)) for image in test_imgs]
v_imgs = [img.reshape((img.shape[0]*img.shape[1],)) for img in my_test_imgs]

predictions = clf.predict(v_imgs)

for i in range(len(predictions)):
               skio.imshow(my_test_imgs[i])
               pyplot.hlines([predictions[i]], 0, 50, colors=['red'])
               pyplot.show()
