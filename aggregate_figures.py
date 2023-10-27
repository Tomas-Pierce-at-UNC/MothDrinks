import json
import re
import scipy
import numpy
from sklearn import ensemble, svm, neighbors, decomposition
from loess import loess_1d
from matplotlib import pyplot
import matplotlib
import scipy
import pandas
import seaborn
import real_unit_convert

RATE_NAME = "nectar ingestion rate (mL/s)"
SUB_NAME = "proboscis submergence depth (mm)"
DATE_EXPR = re.compile(r"\d{4}.\d{2}.\d{2}")

font = {'family': 'normal',
        'weight': 'normal',
        'size': 25}

matplotlib.rc('font', **font)
matplotlib.rc('lines', markersize=20)

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

def notdelta(name):
    return not ('delta' in name or 'Delta' in name)


if __name__ == '__main__':
    s_frames, l_frames = load_all()
    convert_units(s_frames, l_frames)
    shared_s_frame = pandas.concat(s_frames)
    shared_l_frame = pandas.concat(l_frames)

    s_names = list(pandas.unique(shared_s_frame['filename']))
    ids_s = [s_names.index(name) for name in shared_s_frame['filename']]
    l_names = list(pandas.unique(shared_l_frame['filename']))
    ids_l = [l_names.index(name) for name in shared_l_frame['filename']]
    shared_l_frame = shared_l_frame.sort_values('filename')
    shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(notdelta)]
    shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(notdelta)]

    ALL_LIVING = False #True #False #True #False

    if ALL_LIVING:
        shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(lambda name: not 'dead' in name)]
        shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(lambda name: not 'dead' in name)]
    else:
        shared_s_frame = shared_s_frame[shared_s_frame['filename'].map(lambda name : not 'unsuitable' in name)]
        shared_l_frame = shared_l_frame[shared_l_frame['filename'].map(lambda name : not 'unsuitable' in name)]

    print(len(shared_l_frame))

    pos_s_frame = shared_s_frame[
        (shared_s_frame['submergence'] >= numpy.float64(0)) & (shared_s_frame['rate'] >= numpy.float(0))]
    pos_s_frame = pos_s_frame[pos_s_frame['rate'] >= 0]
    names = list(pandas.unique(pos_s_frame['filename']))
    ids_pos = [names.index(name) for name in pos_s_frame['filename']]

    sub_s = shared_l_frame['submergence'] >= 0
    drink_s = shared_l_frame['rate'] >= 0
    sub_proportion = sum(shared_l_frame['rate'] < 0) / len(shared_l_frame)
    sup_proportion = sum(shared_l_frame['rate'] >= 0) / len(shared_l_frame)
    sub_percent = sub_proportion * 100
    sup_percent = sup_proportion * 100
    print(round(sub_percent, 1))
    print(round(sup_percent, 1))
    print(round(sub_percent, 1) + round(sup_percent, 1))
    pos_l_frame = shared_l_frame[sub_s & drink_s]
    names = list(pandas.unique(pos_l_frame['filename']))
    ids_posl = [names.index(name) for name in pos_l_frame['filename']]
    pos_l_frame['mothname'] = pos_l_frame['filename'].map(get_moth_name)
    pos_l_frame['date'] = pos_l_frame['filename'].map(get_date)

    pos_l_frame = pos_l_frame.sort_values('mothname')
    l_labels, l_levels = pos_l_frame['mothname'].factorize()
    pos_l_frame['moth ID'] = l_labels.astype(str)
    pos_l_frame['moth ID'] = pos_l_frame['moth ID'].astype(int).astype(str)
    l_files = pos_l_frame.drop(columns=['mothname', 'date',]).groupby(by='filename')
    meds_l = l_files.median()
    meds_l['date'] = meds_l.index.map(get_date)
    meds_l['mothname'] = meds_l.index.map(get_moth_name)
    meds_l['moth ID'] = meds_l['moth ID'].astype(int).astype(str)
    perchstatus = pandas.read_excel("postures.ods", engine='odf')
    perchstatus = perchstatus.iloc[:44]
    perchstatus['perched'] = perchstatus['isPerched'].map(lambda s : "yes" in s)
    perchstatus['filename'] = perchstatus['filename'].map(lambda x : "data2/{}".format(x))
    perchstatus['curling'] = perchstatus['curled'] == 'yes'
    perching = meds_l.merge(perchstatus,
                            how='inner',
                            left_index = True,
                            right_on='filename'
                            )
    perching['flying'] = ~perching['perched']
    unc = perching[~perching.curling]
    curling = perching[perching.curling]
    bigshared = pos_l_frame.merge(perchstatus, how='left', on='filename')
    bigshared['categories'] = bigshared.curling.astype(int) + (bigshared.perched).astype(int)
    perching['categories'] = perching.curling.astype(int) + (~perching.flying).astype(int)
    print("fly v curl")
    s = scipy.stats.mannwhitneyu(unc[unc.flying].rate,
                             curling.rate,
                             alternative='greater'
                             )
    print(s)
    print("fly v perch (unroll)")
    s = scipy.stats.mannwhitneyu(unc[unc.flying].rate,
                                 unc[~unc.flying].rate,
                                 alternative='greater')
    print(s)
    print("perch unroll vs perch curl")
    s = scipy.stats.mannwhitneyu(unc[~unc.flying].rate,
                                 curling.rate,
                                 alternative='greater'
                                 )
    print(s)
    
    gax = seaborn.violinplot(perching, x='categories', y='rate', cut=0)
    gax.set(xticklabels=['flying,\nproboscis unrolled',
                         'perched,\nproboscis unrolled',
                         'perched,\nproboscis curled'
                         ]
            )
    pyplot.ylabel(RATE_NAME)
    pyplot.xlabel("posture of moth and proboscis when feeding")
    pyplot.show()

    matplotlib.rc('font', **font)

    regr_all = scipy.stats.linregress(unc.submergence, unc.rate, alternative='greater')
    print(regr_all)
    print(len(unc))
    ax1 = seaborn.regplot(unc, x='submergence', y='rate',scatter=False)
    ax1 = seaborn.scatterplot(unc,
                              x='submergence',
                              y='rate',
                              hue='moth ID',
                              style='flying',
                              ax=ax1)
    pyplot.xlim(0, 65)
    pyplot.xlabel(SUB_NAME, fontdict={'fontsize' : 30})
    pyplot.ylabel(RATE_NAME, fontdict={'fontsize': 30})
    pyplot.show()
    print("Sub 10")
    low = perching[perching.submergence <= 10]
    low = low[~low.curling]
    seaborn.regplot(low, x='submergence', y='rate',
                    scatter=False)
    ax = pyplot.gca()
    seaborn.scatterplot(low,
                        x='submergence',
                        y='rate',
                        hue='moth ID',
                        style='flying',
                        ax=ax)
    print("sub 10")
    lr = scipy.stats.linregress(unc.submergence, unc.rate)
    print(lr)
    pyplot.xlabel(SUB_NAME)
    pyplot.ylabel(RATE_NAME)
    pyplot.xlim(right=12)
    pyplot.ylim(bottom=0)
    pyplot.show()
    seaborn.residplot(low[~low.curling], x='submergence', y='rate')
    pyplot.xlabel(SUB_NAME)
    pyplot.ylabel(RATE_NAME)
    pyplot.show()
    fly = unc[unc['flying']]
    perch = unc[~unc['flying']]
    print("is fly-rate significantly greater than perch rate?")
    sig_test = scipy.stats.mannwhitneyu(fly.rate, perch.rate, alternative="greater")
    print(sig_test)
    if sig_test.pvalue <= 0.05:
        print("yes")
    

    meds_l['date'] = pandas.to_datetime(meds_l['date'])

    meds_l['mindate'] = meds_l['moth ID'].map(meds_l.groupby('moth ID')['date'].min())

    meds_l['delay'] = meds_l['date'] - meds_l['mindate']

    big_unc = bigshared[~bigshared.curling]
    flierprops = dict(marker='o',
                      markerfacecolor='None',
                      markersize=2,
                      markeredgecolor='black')
    seaborn.boxplot(big_unc, x='rate', y='moth ID', flierprops=flierprops)
    pyplot.xlabel(RATE_NAME)
    pyplot.show()

    seaborn.boxplot(big_unc, x='submergence', y='moth ID', flierprops=flierprops)
    pyplot.xlabel(SUB_NAME)
    pyplot.show()

    big_unc['date'] = big_unc.filename.map(meds_l.date)

    # arbitary date between the two data collection phases
    # when no measurements were collected.
    splitwhen = pandas.Timestamp(year=2022, month=6, day=1)

    before = big_unc[big_unc.date < splitwhen]
    after = big_unc[big_unc.date > splitwhen]

    myfig, myaxes = pyplot.subplots(nrows=2,
                                    ncols=2,
                                    sharey=True)

    seaborn.lineplot(before, x='date', y='rate', hue='moth ID', ax=myaxes[0,0])
    seaborn.stripplot(before, x='date', y='rate', hue='moth ID', ax=myaxes[1,0])

    seaborn.lineplot(after, x='date', y='rate', hue='moth ID', ax=myaxes[0,1])
    seaborn.stripplot(after, x='date', y='rate', hue='moth ID', ax=myaxes[1,1])
    

    myaxes[0,0].set_ylabel(RATE_NAME)
    myaxes[1,0].set_ylabel(RATE_NAME)

    myaxes[1,0].set_xlabel("Date")
    myaxes[1,1].set_xlabel("Date")
    myaxes[0,0].set_xlabel("Date")
    myaxes[0,1].set_xlabel("Date")

    myaxes[0,0].tick_params('x', labelrotation=90)
    myaxes[0,1].tick_params('x', labelrotation=90)
    myaxes[1,0].tick_params('x', labelrotation=90)
    myaxes[1,1].tick_params('x', labelrotation=90)


    #myfig.tight_layout()

    #line = pyplot.Line2D((0.5, 0.5), (0.1, 0.9), color='k', linewidth=3)
    #myfig.add_artist(line)

    pyplot.show()

    before['dayspast'] = (before.date - before.date.min()).map(
        lambda elapsed: elapsed.days
        )
    after['dayspast'] = (after.date - after.date.min()).map(
        lambda elapsed: elapsed.days
        )

    big_unc['dayspast'] = (big_unc.date - big_unc.date.min()).map(
        lambda elapsed: elapsed.days
        )

    #effectively oversampling
    b_regress = scipy.stats.linregress(before.dayspast, before.rate)
    a_regress = scipy.stats.linregress(after.dayspast, after.rate)
    all_regress = scipy.stats.linregress(big_unc.dayspast, big_unc.rate)

    bf = before[['filename','rate', 'submergence', 'dayspast']].groupby('filename').median()
    af = after[['filename','rate', 'submergence', 'dayspast']].groupby('filename').median()
    bg = big_unc[['filename','rate', 'submergence', 'dayspast']].groupby('filename').median()
    #fig, axes = pyplot.subplots(ncols=2)
    #axb = seaborn.regplot(before, x='dayspast', y='rate', scatter=False)
    #seaborn.scatterplot(before, x='dayspast', y='rate', hue='moth ID', ax=axb)
    #pyplot.show()

    ax_all = seaborn.regplot(big_unc, x='dayspast', y='rate', scatter=False)
    seaborn.scatterplot(big_unc, x='dayspast', y='rate', hue='moth ID')
    pyplot.xlabel("Days since initial recording")
    pyplot.ylabel(RATE_NAME)
    pyplot.show()
    

    print("total")
    print(len(l_names))
    print("viable")
    print(len(shared_l_frame.filename.unique()))

    print("perched")
    print(len(perching[perching.perched]))
    print("flying")
    print(len(perching[perching.flying]))
    print("uncurled")
    print(len(unc))
    print("curled")
    print(len(curling))
