import json
import cine
import tube2
from matplotlib import pyplot
import numpy
from sklearn import svm


with open("meniscus_measurements.json") as mm:
    data = json.load(mm)

PAUSE = 1/64


def animate_meniscustrack(data):
    fig, ax = pyplot.subplots()
    for i, vname in enumerate(data):
        if i < 5:
            continue
        measurements = data[vname]
        try:
            vid = cine.Cine(vname)

            for index, row, col in measurements:
                f = vid.get_ith_image(index)
                frame = tube2.tube_crop1(f)
                ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
                ax.hlines(row, col - 5, col+5, color='red')
                ax.vlines(col, row - 5, row + 5, color='red')
                fig.show()
                pyplot.pause(PAUSE)
                fig.clear()
                fig.add_axes(ax)
        finally:
            vid.close()


svm_regressor = svm.SVR(C=0.9)

for i, vname in enumerate(data):
    measurements = numpy.array(data[vname])
    if len(measurements) < 50:
        continue
    variances = []
    for j, entry in enumerate(measurements):
        span = measurements[max(j - 15, 0):min(j + 15, len(measurements))]
        var = span[:, 1].var()
        variances.append(var)
    v = numpy.array(variances)
    svm_regressor.fit(measurements[:, 0].reshape(-1, 1), measurements[:, 1])
    should = svm_regressor.predict(measurements[:, 0].reshape(-1, 1))
    difs = measurements[:, 1] - should
    pyplot.scatter(measurements[:, 0],
                   measurements[:, 1],
                   marker='.',
                   color='blue')
    pyplot.errorbar(measurements[:, 0],
                    measurements[:, 1],
                    yerr=numpy.sqrt(v),
                    color='red')
    pyplot.scatter(measurements[:, 0], should, marker='+', color='green')
    pyplot.title(vname)
    print(difs.var())
    pyplot.show()
