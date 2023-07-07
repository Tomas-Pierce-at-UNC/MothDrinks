
import json
import scipy
import numpy
from sklearn import ensemble, svm, neighbors
from loess import loess_1d
from matplotlib import pyplot
import pandas
import seaborn
import real_unit_convert


def handle_meniscus_array(mtable: list):
    table = numpy.array(mtable)
    # using an SVM with a soft regularization
    # to compute an outlier-resistant derivative
    model = svm.SVR(C=0.9)
    model.fit(table[:,0].reshape(-1,1), table[:,1])
    should = model.predict(table[:,0].reshape(-1,1))
    pos_difs = numpy.diff(should)
    t_difs = numpy.diff(table[:,0])
    derivs = pos_difs / t_difs
    d_table = numpy.c_[table[:-1,0], derivs, pos_difs, t_difs]
    m_frame = pandas.DataFrame(table, columns=("frame_num", "centroid_row_m", "centroid_col_m"))
    d_frame = pandas.DataFrame(d_table, columns=('frame_num', 'derivative', 'deltaY', 'deltaT'))
    my_frame = m_frame.merge(d_frame, on=("frame_num",))
    # remove outliers using IsolationForest and
    # compute derivative using LOESS model
    isofor = ensemble.IsolationForest()
    labels = isofor.fit_predict(table)
    subtable = table[labels == 1]
    predx, predy, predw = loess_1d.loess_1d(subtable[:,0], subtable[:,1])
    px_difs = numpy.diff(predx)
    py_difs = numpy.diff(predy)
    l_deriv = py_difs / px_difs
    l_table = numpy.c_[subtable[:-1,0], l_deriv, py_difs, px_difs, subtable[:-1, 1]]
    loess_frame = pandas.DataFrame(l_table, columns=("frame_num",
                                                     "LOESS_derivative",
                                                     "LOESS_deltaY",
                                                     "LOESS_deltaX",
                                                     "centroid_row_m",
                                                     ))
    return my_frame, loess_frame


def handle_proboscis_array(ptable: list):
    table = numpy.array(ptable)
    frame = pandas.DataFrame(table, columns=("frame_num",
                                               "centroid_row_p",
                                               "centroid_col_p",
                                               "eccentricity",
                                               "orientation",
                                               "intensity_mean",
                                               "intensity_max",
                                               "intensity_min",
                                               "minrow",
                                               "mincol",
                                               "maxrow",
                                               "maxcol",
                                               "major_axis_length")
                             )
    return frame


def load_proboscis(filename="proboscis_measurements.json"):
    with open(filename) as proboscisfile:
        proboscisdata = json.load(proboscisfile)
    proboscis2 = {}
    for fname in proboscisdata:
        # proboscis2[fname] = handle_proboscis_array(proboscisdata[fname])
        table = handle_proboscis_array(proboscisdata[fname])
        table['filename'] = [fname] * len(table)
        proboscis2[fname] = table
    return proboscis2


def load_meniscus(filename="meniscus_measurements.json"):
    with open(filename) as meniscusfile:
        meniscusdata = json.load(meniscusfile)
    meniscus2 = {}
    for fname in meniscusdata:
        print("*")
        s_frame, l_frame = handle_meniscus_array(meniscusdata[fname])
        s_frame['filename'] = [fname] * len(s_frame)
        l_frame['filename'] = [fname] * len(l_frame)
        meniscus2[fname] = (s_frame, l_frame)
    return meniscus2


def load_all():
    meniscus = load_meniscus()
    proboscis = load_proboscis()

    fnames = set(meniscus.keys()) & set(proboscis.keys())

    s_tables = []
    l_tables = []

    for filename in fnames:
        meniscus_svm, meniscus_loess = meniscus[filename]
        proboscis_f = proboscis[filename]
        table_s = proboscis_f.merge(meniscus_svm, on=("filename", "frame_num",))
        table_l = proboscis_f.merge(meniscus_loess, on=("filename", "frame_num",))
        
        s_tables.append(table_s)
        l_tables.append(table_l)
    return s_tables, l_tables


def convert_units(s_tables, l_tables):
    ruc = real_unit_convert.UnitConversion()

    for s, l in zip(s_tables, l_tables):
        sname = s['filename'][0]
        lname = l['filename'][0]
        s_framerate = ruc.get_framerate(sname)
        l_framerate = ruc.get_framerate(lname)
        
        s['submergence_raw'] = s['maxrow'] - s['centroid_row_m']
        l['submergence_raw'] = l['maxrow'] - l['centroid_row_m']
        
        s['submergence'] = s['submergence_raw'] * ruc.get_lin_factor(sname)
        l['submergence'] = l['submergence_raw'] * ruc.get_lin_factor(lname)

        sv = ruc.get_vol_factor(sname)
        s['rate']=s['derivative']*sv*s_framerate
        
        lv = ruc.get_vol_factor(lname)
        l['rate']=l['LOESS_derivative']*lv*l_framerate

        s['elapsed_time'] = (s['frame_num'] - min(s['frame_num'])) / s_framerate
        l['elapsed_time'] = (l['frame_num'] - min(l['frame_num'])) / l_framerate
    return None


def get_moth_name(filename):
    start = filename.index("moth")
    end = filename.index("_", start)
    return filename[start:end]


s_frames, l_frames = load_all()
convert_units(s_frames, l_frames)
shared_s_frame = pandas.concat(s_frames)
shared_l_frame = pandas.concat(l_frames)

s_names = list(pandas.unique(shared_s_frame['filename']))
ids_s = [s_names.index(name) for name in shared_s_frame['filename']]
l_names = list(pandas.unique(shared_l_frame['filename']))
ids_l = [l_names.index(name) for name in shared_l_frame['filename']]

#pyplot.scatter(shared_s_frame['rate'], shared_s_frame['submergence'], c=ids_s)
#pyplot.show()

#pyplot.scatter(shared_l_frame['rate'], shared_l_frame['submergence'], c=ids_l)
#pyplot.show()

shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(lambda name : not 'unsuitable' in name)]
shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(lambda name : not 'unsuitable' in name)]

pos_s_frame = shared_s_frame[shared_s_frame['submergence'] >= 0]
pos_s_frame = pos_s_frame[pos_s_frame['rate'] >= 0]
names = list(pandas.unique(pos_s_frame['filename']))
ids_pos = [names.index(name) for name in pos_s_frame['filename']]
pyplot.scatter(pos_s_frame['submergence'], pos_s_frame['rate'],c=ids_pos)
pyplot.title("submergence vs ingest rate w/ SVM")
pyplot.xlabel("submergence mm")
pyplot.ylabel("rate mL/s")
pyplot.show()

pos_l_frame = shared_l_frame[shared_l_frame['submergence'] >= 0]
pos_l_frame = pos_l_frame[pos_l_frame['rate'] >= 0]
names = list(pandas.unique(pos_l_frame['filename']))
ids_posl = [names.index(name) for name in pos_l_frame['filename']]
pyplot.scatter(pos_l_frame['submergence'], pos_l_frame['rate'],c=ids_posl)
pyplot.title("submergence vs ingest rate w/ LOESS")
pyplot.xlabel("submergence mm")
pyplot.ylabel("rate mL/s")
pyplot.show()

pos_l_frame['mothname'] = pos_l_frame['filename'].map(get_moth_name)
pos_s_frame['mothname'] = pos_s_frame['filename'].map(get_moth_name)

#seaborn.violinplot(pos_l_frame, x='mothname', y='submergence', inner=None)
#pyplot.show()

#seaborn.violinplot(pos_l_frame, x='mothname', y='rate', inner=None)
#pyplot.show()

seaborn.boxplot(pos_l_frame, x='mothname', y='submergence')
pyplot.title("submergence depth distribution LOESS")
pyplot.xlabel("moth name")
pyplot.ylabel("submergence mm")
pyplot.show()

seaborn.boxplot(pos_l_frame, x='mothname', y='rate')
pyplot.title("drinking rate distribution LOESS")
pyplot.xlabel("moth name")
pyplot.ylabel("rate mL/s")
pyplot.show()

seaborn.boxplot(pos_s_frame, x='mothname', y='submergence')
pyplot.title("submergence depth distribution SVM")
pyplot.xlabel("moth name")
pyplot.ylabel("submergence mm")
pyplot.show()

seaborn.boxplot(pos_s_frame, x='mothname', y='rate')
pyplot.title("drinking rate distribution SVM")
pyplot.xlabel("moth name")
pyplot.ylabel("rate mL/s")
pyplot.show()

seaborn.lineplot(pos_l_frame, x='elapsed_time', y='submergence', hue='filename')
pyplot.title('submergence over time LOESS')
pyplot.xlabel('elapsed time s')
pyplot.ylabel('submergence mm')
pyplot.show()

seaborn.lineplot(pos_s_frame, x='elapsed_time', y='submergence', hue='filename')
pyplot.title('submergence over time SVM')
pyplot.xlabel('elapsed time s')
pyplot.ylabel('submergence mm')
pyplot.show()

seaborn.lineplot(pos_l_frame, x='elapsed_time', y='rate', hue='filename')
pyplot.title('rate over time LOESS')
pyplot.xlabel("elapsed time s")
pyplot.ylabel("rate mL/s")
pyplot.show()

seaborn.scatterplot(pos_l_frame, x='elapsed_time', y='rate', hue='filename')
pyplot.title('rate over time LOESS')
pyplot.xlabel("elapsed time s")
pyplot.ylabel("rate mL/s")
pyplot.show()

seaborn.lineplot(pos_s_frame, x='elapsed_time', y='rate', hue='filename')
pyplot.title('rate over time SVM')
pyplot.xlabel("elapsed time s")
pyplot.ylabel("rate mL/s")
pyplot.show()
