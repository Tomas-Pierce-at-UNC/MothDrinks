import json
import re
import scipy
import numpy
from sklearn import ensemble, svm, neighbors, decomposition
from loess import loess_1d
from matplotlib import pyplot
import scipy
import pandas
import seaborn
import real_unit_convert

RATE_NAME = "nectar ingestion rate (mL/s)"
SUB_NAME = "proboscis submergence depth (mm)"
DATE_EXPR = re.compile(r"\d{4}.\d{2}.\d{2}")

def handle_meniscus_array(mtable: list):
    table = numpy.array(mtable)
    # using an SVM with a soft regularization
    # to compute an outlier-resistant derivative
    model = svm.SVR(C=0.9)
    #print(table.shape)
    model.fit(table[:, 0].reshape(-1, 1), table[:, 1])
    should = model.predict(table[:, 0].reshape(-1, 1))
    pos_difs = numpy.diff(should)
    t_difs = numpy.diff(table[:, 0])
    derivs = pos_difs / t_difs
    d_table = numpy.c_[table[:-1, 0], derivs, pos_difs, t_difs]
    m_frame = pandas.DataFrame(table, columns=("frame_num",
                                               "centroid_row_m",
                                               "centroid_col_m",
                                               "intensity_mean_m",
                                               "intensity_max_m",
                                               "intensity_min_m",
                                               "area_m",
                                               "axis_major_length_m",
                                               "axis_minor_length_m",
                                               "equivalent_diamter_area_m",
                                               "feret_diameter_max_m",
                                               "orientation_m",
                                               "perimiter_m",
                                               "eccentricity_m"
                                               ))
    d_frame = pandas.DataFrame(d_table, columns=('frame_num', 'derivative', 'deltaY', 'deltaT'))
    my_frame = m_frame.merge(d_frame, on=("frame_num",))
    # remove outliers using IsolationForest and
    # compute derivative using LOESS model
    isofor = ensemble.IsolationForest()
    pca = decomposition.PCA(n_components=5)
    reduced = pca.fit_transform(table)
    labels = isofor.fit_predict(reduced)
    subtable = table[labels == 1]
    #subtable = table[labels == 1]
    #subtable = table
    predx, predy, predw = loess_1d.loess_1d(subtable[:, 0], subtable[:, 1])
    px_difs = numpy.diff(predx)
    py_difs = numpy.diff(predy)
    l_deriv = py_difs / px_difs
    l_table = numpy.c_[subtable[:-1, 0], l_deriv, py_difs, px_difs, subtable[:-1, 1]]
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
                                             "major_axis_length",
                                             "area")
                             )
    high = frame[frame['intensity_mean'] > 0.5]
    groups = high.groupby(by=['frame_num'])
    idx = groups['maxrow'].transform(max) == high['maxrow']
    return high[idx]


def load_proboscis(filename="proboscis_measurements3-net.json"):
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
        if len(meniscusdata[fname]) == 0:
            print(fname)
            print("is empty")
            continue
        print("*")
        try:
            s_frame, l_frame = handle_meniscus_array(meniscusdata[fname])
        except SystemError as se:
            print(se)
            print(fname)
            continue
        except IndexError as ie:
            print(ie)
            print(fname)
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
        s['rate'] = s['derivative'] * sv * s_framerate

        lv = ruc.get_vol_factor(lname)
        l['rate'] = l['LOESS_derivative'] * lv * l_framerate

        s['elapsed_time'] = (s['frame_num'] - min(s['frame_num'])) / s_framerate
        l['elapsed_time'] = (l['frame_num'] - min(l['frame_num'])) / l_framerate
    return None


def get_moth_name(filename):
    # if "moth" not in filename:
        # print(filename)
    start = filename.index("moth")
    end = filename.index("_", start)
    return filename[start:end]

def get_date(filename):
    date = DATE_EXPR.findall(filename)[0]
    return date.replace("_", "-")


if __name__ == '__main__':
    s_frames, l_frames = load_all()
    convert_units(s_frames, l_frames)
    shared_s_frame = pandas.concat(s_frames)
    shared_l_frame = pandas.concat(l_frames)

    s_names = list(pandas.unique(shared_s_frame['filename']))
    ids_s = [s_names.index(name) for name in shared_s_frame['filename']]
    l_names = list(pandas.unique(shared_l_frame['filename']))
    ids_l = [l_names.index(name) for name in shared_l_frame['filename']]

    def notdelta(name):
        return not ('delta' in name or 'Delta' in name)

    shared_l_frame = shared_l_frame.sort_values('filename')

    shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(notdelta)]
    shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(notdelta)]
    #shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(lambda name: not 'dead' in name)]
    #shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(lambda name: not 'dead' in name)]
    shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(lambda name : not 'unsuitable' in name)]
    shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(lambda name : not 'unsuitable' in name)]
    

    pos_s_frame = shared_s_frame[
        (shared_s_frame['submergence'] >= numpy.float64(0)) & (shared_s_frame['rate'] >= numpy.float(0))]
    pos_s_frame = pos_s_frame[pos_s_frame['rate'] >= 0]
    names = list(pandas.unique(pos_s_frame['filename']))
    ids_pos = [names.index(name) for name in pos_s_frame['filename']]
    ##pyplot.scatter(pos_s_frame['submergence'], pos_s_frame['rate'],c=ids_pos)
    ##pyplot.title("submergence vs ingest rate w/ SVM")
    ##pyplot.xlabel("submergence mm")
    ##pyplot.ylabel("rate mL/s")
    ##pyplot.show()

    sub_s = shared_l_frame['submergence'] >= 0
    drink_s = shared_l_frame['rate'] >= 0
    sub_proportion = sum(shared_l_frame['rate'] < 0) / len(shared_l_frame)
    sup_proportion = sum(shared_l_frame['rate'] >= 0) / len(shared_l_frame)
    #print(sub_proportion)
    #print(sup_proportion)
    #print(sub_proportion + sup_proportion)
    sub_percent = sub_proportion * 100
    sup_percent = sup_proportion * 100
    print(round(sub_percent, 1))
    print(round(sup_percent, 1))
    print(round(sub_percent, 1) + round(sup_percent, 1))
    pos_l_frame = shared_l_frame[sub_s & drink_s]
    # pos_l_frame = pos_l_frame[pos_l_frame['rate'] >= 0]
    names = list(pandas.unique(pos_l_frame['filename']))
    ids_posl = [names.index(name) for name in pos_l_frame['filename']]
    ##pyplot.scatter(pos_l_frame['submergence'], pos_l_frame['rate'],c=ids_posl)
    ##pyplot.title("submergence vs ingest rate w/ LOESS")
    ##pyplot.xlabel("submergence mm")
    ##pyplot.ylabel("rate mL/s")
    ##pyplot.show()

    pos_l_frame['mothname'] = pos_l_frame['filename'].map(get_moth_name)
    pos_l_frame['date'] = pos_l_frame['filename'].map(get_date)
    #pos_s_frame['mothname'] = pos_s_frame['filename'].map(get_moth_name)

    pos_l_frame = pos_l_frame.sort_values('mothname')
    #pos_s_frame = pos_s_frame.sort_values('mothname')

    #pos_l_frame['moth ID'] = pos_l_frame['mothname'].factorize(
    l_labels, l_levels = pos_l_frame['mothname'].factorize()
    pos_l_frame['moth ID'] = l_labels.astype(str)
    pos_l_frame['moth ID'] = pos_l_frame['moth ID'].astype(int).astype(str)

    #s_labels, s_levels = pos_s_frame['mothname'].factorize()
    #pos_s_frame['moth ID'] = s_labels

    l_files = pos_l_frame.drop(columns=['mothname', 'date',]).groupby(by='filename')
    #s_files = pos_s_frame.drop(columns=['mothname']).groupby(by='filename')

    meds_l = l_files.median()
    meds_l['date'] = meds_l.index.map(get_date)
    #meds_s = s_files.median()

    #meds_s['mothname'] = meds_s.index.map(get_moth_name)
    meds_l['mothname'] = meds_l.index.map(get_moth_name)

    meds_l['moth ID'] = meds_l['moth ID'].astype(int).astype(str)

    # seaborn.violinplot(pos_l_frame, x='mothname', y='submergence', inner=None)
    # pyplot.show()

    # seaborn.violinplot(pos_l_frame, x='mothname', y='rate', inner=None)
    # pyplot.show()

    #raise Exception("ah hell")

    perchstatus = pandas.read_excel("postures.ods", engine='odf')
    perchstatus = perchstatus.iloc[:43]
    perchstatus['perched'] = perchstatus['isPerched'].map(lambda s : "yes" in s)
    perchstatus['filename'] = perchstatus['filename'].map(lambda x : "data2/{}".format(x))

    perching = meds_l.merge(perchstatus, how='inner', on='filename')
    perching['flying'] = ~perching['perched']
    seaborn.violinplot(perching, x='flying', y='rate', cut=0)
    pyplot.ylabel(RATE_NAME)
    pyplot.xlabel("Moth is flying in recording?")
    pyplot.show()

    fly = perching[perching['flying']]
    perch = perching[~perching['flying']]

    print("is fly-rate significantly greater than perch rate?")
    sig_test = scipy.stats.mannwhitneyu(fly.rate, perch.rate, alternative="greater")
    print(sig_test)
    if sig_test.pvalue <= 0.05:
        print("yes")

    #kruskal = scipy.stats.kruskall(fly, perch)
    #print(kruskal)

    meds_l['date'] = pandas.to_datetime(meds_l['date'])

    seaborn.lineplot(meds_l, x='date', y='rate', hue='moth ID')
    pyplot.xticks(rotation=90)
    pyplot.ylabel(RATE_NAME)
    pyplot.xlabel("Date")
    pyplot.show()

    seaborn.stripplot(meds_l, x='date', y='rate', hue='moth ID')
    pyplot.xticks(rotation=90)
    pyplot.ylabel(RATE_NAME)
    pyplot.xlabel("Date")
    pyplot.show()

    meds_K = pos_l_frame.drop(columns=['filename', 'date']).groupby(by="mothname")
    meds_KK = meds_K.median()
    #meds_KK['mothname'] = meds_KK.index
    meds_KK['moth ID'] = meds_KK['moth ID'].astype(int).astype(str)
    seaborn.scatterplot(meds_KK, x='submergence', y='rate', hue='moth ID')
    #pyplot.title("median moth LOESS sub v rate")
    pyplot.ylabel(RATE_NAME)
    pyplot.xlabel(SUB_NAME)
    pyplot.show()

##    meds_K2 = pos_s_frame.drop(columns=['filename']).groupby(by="mothname")
##    meds_KK2 = meds_K2.median()
##    meds_KK2['mothname'] = meds_KK2.index
##    seaborn.scatterplot(meds_KK2, x='submergence', y='rate', hue='mothname')
##    pyplot.title("median moth SVM sub v rate")
##    pyplot.ylabel(RATE_NAME)
##    pyplot.xlabel(SUB_NAME)
##    pyplot.show()

    pl = seaborn.lmplot(meds_l, x='submergence', y='rate', markers='.')
    ax = pyplot.gca()
    seaborn.scatterplot(meds_l, x='submergence', y='rate', hue='moth ID', ax=ax)
    for item in range(len(meds_l)):
        pyplot.text(meds_l.submergence[item]+0.01, meds_l.rate[item],
                    meds_l['moth ID'][item],
                horizontalalignment='left', size='small', color='black',
                weight='light')
    pyplot.xlabel(SUB_NAME)
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

    seaborn.scatterplot(meds_l, x='submergence', y='rate', hue='moth ID')
    #pyplot.title("median submergence vs rate with LOESS")
    pyplot.xlabel(SUB_NAME)
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

##    seaborn.scatterplot(meds_s, x='submergence', y='rate', hue='mothname')
##    pyplot.title("median submergence vs rate with SVM")
##    pyplot.xlabel(SUB_NAME)
##    pyplot.ylabel(RATE_NAME)
##    pyplot.show()

##    seaborn.scatterplot(pos_s_frame, x='submergence', y='rate', hue='mothname')
##    pyplot.title("submergence vs ingest rate w/ SVM")
##    pyplot.xlabel("submergence mm")
##    pyplot.ylabel("rate mL/s")
##    pyplot.show()

    seaborn.scatterplot(pos_l_frame, x='submergence', y='rate', hue='moth ID')
    #pyplot.title("submergence vs ingest rate w/ LOESS")
    pyplot.xlabel(SUB_NAME)
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

    seaborn.boxplot(pos_l_frame, x='moth ID', y='submergence')
    pyplot.title("submergence depth distribution LOESS")
    pyplot.xlabel("moth ID")
    pyplot.ylabel(SUB_NAME)
    #pyplot.xticks(rotation=90)
    pyplot.show()

    seaborn.boxplot(pos_l_frame, x='moth ID', y='rate')
    #pyplot.title("drinking rate distribution LOESS")
    pyplot.xlabel("moth ID")
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

##    seaborn.boxplot(pos_s_frame, x='mothname', y='submergence')
##    pyplot.title("submergence depth distribution SVM")
##    pyplot.xlabel("moth name")
##    pyplot.ylabel("submergence mm")
##    pyplot.show()

##    seaborn.boxplot(pos_s_frame, x='mothname', y='rate')
##    pyplot.title("drinking rate distribution SVM")
##    pyplot.xlabel("moth name")
##    pyplot.ylabel("rate mL/s")
##    pyplot.show()

    seaborn.lineplot(pos_l_frame, x='elapsed_time', y='submergence', hue='filename')
    pyplot.title('submergence over time LOESS')
    pyplot.xlabel('elapsed time (s)')
    pyplot.ylabel(SUB_NAME)
    pyplot.show()

##    seaborn.lineplot(pos_s_frame, x='elapsed_time', y='submergence', hue='filename')
##    pyplot.title('submergence over time SVM')
##    pyplot.xlabel('elapsed time s')
##    pyplot.ylabel('submergence mm')
##    pyplot.show()

    seaborn.lineplot(pos_l_frame, x='elapsed_time', y='rate', hue='filename')
    pyplot.title('rate over time LOESS')
    pyplot.xlabel("elapsed time (s)")
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

    seaborn.scatterplot(pos_l_frame, x='elapsed_time', y='rate', hue='filename')
    pyplot.title('rate over time LOESS')
    pyplot.xlabel("elapsed time s")
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

##    seaborn.lineplot(pos_s_frame, x='elapsed_time', y='rate', hue='filename')
##    pyplot.title('rate over time SVM')
##    pyplot.xlabel("elapsed time s")
##    pyplot.ylabel("rate mL/s")
##    pyplot.show()

    seaborn.lmplot(meds_l, x='submergence', y='rate', hue='moth ID', markers='.')
    pyplot.title("rate vs submergence median w/ LOESS")
    pyplot.xlabel("submergence mm")
    pyplot.ylabel(RATE_NAME)
    pyplot.show()

##    seaborn.lmplot(meds_s, x='submergence', y='rate', hue='mothname', markers='.')
##    pyplot.title("rate vs submergence median w/ SVM")
##    pyplot.xlabel("submergence mm")
##    pyplot.ylabel(RATE_NAME)
##    pyplot.show()

