
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

with open('proboscis_measurements.json') as pp:
    data = json.load(pp)
#with open("meniscus_measurements.json") as mm:
#    data = json.load(mm)

PAUSE = 1/64

def animate_proboscistrack(data):
    fig, ax = pyplot.subplots()
    for i,vname in enumerate(data):
        if "unsuitable" in vname:
            continue
        measurements = data[vname]
        try:
            vid = cine.Cine(vname)
            for entry in measurements:
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
                ax.hlines(row, col - 5, col+5, color='red')
                #pyplot.vlines(col, row - 5, row + 5, color='red')
                ax.vlines(col, row - 5, row + 5, color='red')
                #ax.add_patch(patch)
                ax.hlines(maxrow, 5, 95, color='green')
                #pyplot.show(block=False)
                #pyplot.sca(ax)
                #pyplot.show(block=False)
                fig.show()
                pyplot.pause(PAUSE)
                fig.clear()
                fig.add_axes(ax)
        finally:
            vid.close()

if __name__ == '__main__':
    animate_proboscistrack(data)