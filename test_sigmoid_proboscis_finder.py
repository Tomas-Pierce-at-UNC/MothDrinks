
import json
import random
import cine
import tube2
from skimage import io as skio
from matplotlib import pyplot
from matplotlib import patches
#import scipy
import numpy
from sklearn import neighbors, svm, ensemble
import itertools

PAUSE = 1/64

def draw_object(ax, entry):
    row = entry[1]
    col = entry[2]
    maxrow = entry[10]
    message = str(round(entry[5],3))
    ax.hlines(row, col - 5, col + 5, color='red')
    ax.vlines(col, row - 5, row + 5, color='red')
    ax.hlines(maxrow, 5, 95, color='green')
    ax.text(col + 6, row, message)

def animate_proboscistrack(data, diffs):
    fig, ax = pyplot.subplots()
    for i,vname in enumerate(data):
        if "unsuitable" in vname:
            continue
        measurements = data[vname]
        deltas = diffs[vname]
        z = zip(measurements, deltas)
        try:
            vid = cine.Cine(vname)
            entries = filter(lambda item : item[0][5] > 0.9, z)
            below = filter(lambda pair : pair[1][5] < 0, entries)
            first = (pair[0] for pair in below)
            groups = itertools.groupby(first, key=lambda entry : entry[0])
            for key, group in groups:
                entry = max(group, key=lambda entry : entry[10])
                if entry[5] < 0.9:
                    # for net measurements
                    continue
                index = entry[0]
                row = entry[1]
                col = entry[2]
                #l = entry[8]
                minrow = entry[8]
                mincol = entry[9]
                maxrow = entry[10]
                maxcol = entry[11]
                xy = (mincol, minrow)
                width = maxcol - mincol
                height = maxrow - minrow
                f = vid.get_ith_image(index)
                frame = tube2.tube_crop1(f)
                #patch = patches.Rectangle(xy, width, height, linewidth=1, edgecolor='green', facecolor='none')
                #skio.imshow(frame)
                ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
                #pyplot.hlines(row, col - 5, col + 5, color='red')
##                ax.hlines(row, col - 5, col+5, color='red')
##                #pyplot.vlines(col, row - 5, row + 5, color='red')
##                ax.vlines(col, row - 5, row + 5, color='red')
##                #ax.add_patch(patch)
##                ax.hlines(maxrow, 5, 95, color='green')
##                #pyplot.show(block=False)
##                #pyplot.sca(ax)
##                #pyplot.show(block=False)
##                fig.show()
                draw_object(ax, entry)
                fig.show()
                pyplot.pause(PAUSE)
                fig.clear()
                fig.add_axes(ax)
        finally:
            vid.close()

if __name__ == '__main__':
    with open('proboscis_measurements3-cls.json') as pp:
        data_diff = json.load(pp)
    with open('proboscis_measurements3-net.json') as pp:
        data_net = json.load(pp)

    
    animate_proboscistrack(data_net, data_diff)
    
