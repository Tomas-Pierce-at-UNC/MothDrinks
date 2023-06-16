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

import real_unit_convert


def calc_derivative(predict_x, predict_y):
    l_delta_x = np.diff(predict_x)
    l_delta_y = np.diff(predict_y)
    l_deriv = l_delta_y / l_delta_x
    return predict_x[1:], l_deriv


def get_corresponding_name(path):
    p = Path(path)
    return p.with_suffix(".cine").parts[-1]


def load_table(name, unit_converter):
    vid_name = get_corresponding_name(name)
    framerate = unit_converter.get_framerate(vid_name)
    vol_conv = unit_converter.get_vol_factor(vid_name)
    table = np.loadtxt(name, dtype=float, delimiter=' ')
    table[:,0] = table[:,0] / framerate
    table[:,1] = table[:,1] * vol_conv
    table[:,1] = table[:,1] - table[:,1].min()
    return table


def apply_isolation_forest(table):
    forest = IsolationForest()
    lbls = forest.fit_predict(table)
    table = table[lbls == 1]
    return table

def form_model(table):
    return loess_1d.loess_1d(table[:,0], table[:,1])

def main():
    unit_conv = real_unit_convert.UnitConversion()
    table_names = glob.glob("meniscusTrackNN/*.csv")
    for name in table_names:
        outname = Path(name).with_suffix(".png")
        table = load_table(name, unit_conv)
        #table[:,0] = table[:,0] / 30
        #width = table[:,6] - table[:,4]
        #table = table[width > 30]
        try:
            table = apply_isolation_forest(table)
            predx,predy,predw = loess_1d.loess_1d(table[:,0],table[:,1])
        except Exception as e:
            print(e)
            continue
        
        p_x, l_derive = calc_derivative(predx, predy)
        
        pyplot.scatter(table[:,0], table[:,1],marker='.')
        pyplot.plot(predx,predy,color="red")
        pyplot.title(name)
        pyplot.ylabel("volume (mL)")
        pyplot.xlabel("time (seconds)")
        pyplot.savefig(outname)
        #pyplot.show()
        pyplot.close()


        pyplot.scatter(p_x, l_derive, marker='.', color='blue')
        pyplot.hlines(0, table[0,0], table[-1,0], color='green')
        pyplot.title("derivative of " + name)
        pyplot.ylabel("nectar flow rate (mL / second)")
        pyplot.xlabel("time (seconds)")
        pyplot.savefig("meniscus_graphs_nn/" + Path(name).stem + "_derivative.png")
        #pyplot.show()
        pyplot.close()

if __name__ == '__main__':
    main()
