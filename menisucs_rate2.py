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

table_names = glob.glob("meniscusTrackNN/*.csv")

for name in table_names:
    outname = Path(name).with_suffix(".png")
    table = np.loadtxt(name,dtype=float,delimiter=" ")
    predx,predy,predw = loess_1d.loess_1d(table[:,0],table[:,1])
    pyplot.scatter(table[:,0],table[:,1])
    pyplot.plot(predx,predy,color="red")
    pyplot.title(name)
    pyplot.savefig(outname)
    #pyplot.show()
    pyplot.close()