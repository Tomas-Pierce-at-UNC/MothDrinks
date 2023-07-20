
import os
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/usr/lib/cuda'

import data_loading as data
import model_arch
import data_augment as augment
from tensorflow.config import list_physical_devices
import numpy as np
import tensorflow as tf
import os

# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/cuda/nvvm:/usr/lib/cuda:/usr/local/cuda'
# os.environ['PATH'] = '/usr/lib/cuda:/usr/local/cuda:{}'.format(os.environ['PATH'])

print(list_physical_devices())
tf.keras.backend.set_image_data_format('channels_last')

X_train, y_train = data.load_dset()

X_test, y_test = data.load_dset("annotations_big.xml", "big")

X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

X_test = X_test[..., np.newaxis]
y_test = y_test[..., np.newaxis]

seq = augment.ImageSeq(X_train, y_train, batch=16)
augmented = augment.RepeatSeq(seq, 20)
augmented = augment.RandomFlip(augmented, thresh=0.2)
augmented = augment.RandomXShift(augmented, thresh=0.2)
augmented = augment.RandomYShift(augmented, thresh=0.2)

fresh_model = model_arch.our_model()

fresh_model.fit(augmented, epochs=15)

predictions = fresh_model.predict(X_test)

evals = fresh_model.evaluate(X_test, y_test, verbose=1)

fresh_model.save("meniscus_track_4")
