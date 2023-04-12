#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:09:20 2023

@author: tomas
"""

from loess import loess_1d
import numpy as np
import glob
from matplotlib import pyplot
from pathlib import Path
from sklearn.ensemble import IsolationForest

def main():
    table_names = glob.glob("meniscusTrackNN/*.csv")
    for name in table_names:
        outname = Path(name).with_suffix(".png")
        table = np.loadtxt(name,dtype=float,delimiter=" ")
        #width = table[:,6] - table[:,4]
        #table = table[width > 30]
        try:
            forest = IsolationForest()
            lbls = forest.fit_predict(table)
            table = table[lbls == 1]
        except Exception as e:
            print(e)
            continue
        try:
            predx,predy,predw = loess_1d.loess_1d(table[:,0],table[:,1])
        except Exception as e:
            print(e)
            continue
        pyplot.scatter(table[:,0],table[:,1],marker='.')
        pyplot.plot(predx,predy,color="red")
        pyplot.title(name)
        pyplot.savefig(outname)
        #pyplot.show()
        pyplot.close()

if __name__ == '__main__':
    main()
