
import functools
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skimage.feature import daisy
import old_data_loading as data
import numpy as np

rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)

ins, outs = data.load_data()
print(ins.shape)
feature_box = []
for inp in ins:
    # print(inp.shape)
    # features = multiscale_basic_features(inp)
    # features = haar_like_feature(inp, 0, 0, inp.shape[0], inp.shape[1])
    features = daisy(inp[:, :, 0], visualize=False)
    feature_box.append(features.ravel())

all_features = np.array(feature_box)
labels = outs.reshape((outs.shape[0], outs.shape[1] * outs.shape[2] * outs.shape[3]))
rfc.fit(feature_box, labels)

# for feats, labels in zip(feature_box, outs):
    #train_dat = feats[labels]
#    train_dat = feats
#    train_label = labels.ravel()
#    rfc.fit(train_dat, train_label)
