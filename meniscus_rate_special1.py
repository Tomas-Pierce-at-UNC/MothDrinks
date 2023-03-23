#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:13:57 2023

@author: tomas
"""

import meniscus_rate as mr
import glob
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot

def no_skinny(data):
    return data[data[:,7] - data[:,5] > mr.MIN_WIDTH]

def remove_outliers(data):
    isofor = IsolationForest()
    lbls = isofor.fit_predict(data)
    return data[lbls == 1]

if __name__ == '__main__':
    names = glob.glob("SpecialHandlingNeeded/*.csv")
    ms = []
    for name in names:
        data = mr.load_measurements(name)
        wide = no_skinny(data)
        typical = remove_outliers(wide)
        lmodx, lmody = mr.build_loess_model(typical)
        lx, deriv = mr.calc_model_derivative(lmodx, lmody)
        pyplot.scatter(typical[:,-1], typical[:,0])
        pyplot.scatter(lmodx,lmody, color="red", marker='.')
        pyplot.title(name)
        pyplot.show()
        pyplot.scatter(lx, deriv, marker='.')
        pyplot.title(name + " derivative")
        pyplot.hlines(0, min(lx), max(lx))
        pyplot.show()