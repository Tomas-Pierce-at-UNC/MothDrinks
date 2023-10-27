
import json
from matplotlib import pyplot
import numpy
from sklearn import neighbors, ensemble, svm

with open("meniscus_measurements.json") as mm:
    data = json.load(mm)

data2 = {}

for fname in data:
    table = numpy.array(data[fname])
    data2[fname] = table
    

for fname in data2:
    clf = neighbors.LocalOutlierFactor(n_neighbors=30, contamination=0.1)
    y_pred = clf.fit_predict(data2[fname])
    X_scores = clf.negative_outlier_factor_
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    pyplot.scatter(data2[fname][:,0],
                   data2[fname][:,1],
                   s=1000*radius,
                   marker='.',
                   facecolors='none',
                   edgecolors='r'
                   )
    pyplot.title(fname)
    pyplot.show()
    
