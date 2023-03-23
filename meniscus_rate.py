

import numpy as np
from loess import loess_1d
import pathlib
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot
import glob
from skimage.filters import threshold_isodata

MIN_AREA = 1000
MIN_WIDTH = 15


def get_stem(filename: str) -> str:
    return pathlib.Path(filename).stem


def load_measurements(filename: str) -> np.ndarray:
    return np.loadtxt(filename, delimiter=',')


def width_filter(table: np.ndarray) -> np.ndarray:
    w = table[table[:,7] - table[:,5] > MIN_WIDTH]
    t = threshold_isodata(w[:,7] - w[:,5])
    w2 = w[w[:,7] - w[:,5] < t]
    return w2

def find_real_data(table: np.ndarray) -> np.ndarray:
    w = width_filter(table)
    isofor = IsolationForest()
    lbls = isofor.fit_predict(w)
    filt = w[lbls == 1]
    return filt


def build_loess_model(table):
    xpred, ypred, wpred = loess_1d.loess_1d(table[:,-1], table[:,0])
    return xpred, ypred


def calc_model_derivative(xpred, ypred):
    mod_xdif = np.diff(xpred)
    mod_ydif = np.diff(ypred)
    #nonzero_xdif = mod_xdif[mod_xdif != 0]
    #nonzero_ydif = mod_ydif[mod_xdif != 0]
    ratio = mod_ydif / mod_xdif
    return xpred[:-1], ratio

def filter_derivatives(xpos, deriv):
    valid_d = deriv[~np.isnan(deriv)]
    valid_x = xpos[~np.isnan(deriv)]
    cond = abs(valid_d - valid_d.mean()) < (3 * valid_d.std())
    return valid_x[cond], valid_d[cond]

def main():
    names = glob.glob("meniscusTracks5/*.csv")
    for name in names:
        try:
            data = load_measurements(name)
            real = find_real_data(data)
            pyplot.scatter(real[:,-1], real[:,0], marker='.', color="blue")
            xpred, ypred = build_loess_model(real)
            pyplot.scatter(xpred, ypred, marker='.', color="red")
            pyplot.title(name)
            pyplot.show()
            xpos, deriv = calc_model_derivative(xpred, ypred)
            xpos, deriv = filter_derivatives(xpos, deriv)
            #isofor = IsolationForest()
            #lbls = isofor.fit_predict(deriv.reshape(-1,1))
            pyplot.scatter(xpos, deriv, marker='.')
            pyplot.hlines(0, min(xpos), max(xpos), color="green")
            pyplot.title(name + " derivative")
            pyplot.show()
        except SystemError as se:
            print(name)
            print(se)
            continue


if __name__ == '__main__':
    main()