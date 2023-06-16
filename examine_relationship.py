
import pathlib

import meniscus_rate2
from loess import loess_1d
import real_unit_convert as ruc
import numpy as np
import pandas
from matplotlib import pyplot
import statistics

def list_filenames():
    p_folder = pathlib.Path("./proboscis_measurements")
    m_folder = pathlib.Path("./meniscusTrackNN")

    p_sheets = list(p_folder.glob("*.tsv"))
    m_sheets = list(m_folder.glob('*.csv'))
    p_stems = [p.stem for p in p_sheets]
    m_stems = [m.stem for m in m_sheets]

    meniscus_names = []
    proboscis_names = []

    for i,p_stem in enumerate(p_stems):
        index = m_stems.index(p_stem)
        #pair = (m_sheets[index], p_sheets[i])
        #paired.append(pair)
        meniscus_names.append(m_sheets[index])
        proboscis_names.append(p_sheets[i])

    return meniscus_names, proboscis_names

def main():
    mlist, plist = list_filenames()
    uc = ruc.UnitConversion()

    r_vals = []

    for i in range(len(mlist)):
        mname = mlist[i]
        pname = plist[i]
        corresponding = meniscus_rate2.get_corresponding_name(mname)
        mystem = pathlib.Path(corresponding).stem
        lin = uc.get_lin_factor(corresponding)
        vol = uc.get_vol_factor(corresponding)
        framerate = uc.get_framerate(corresponding)
        m_table = np.loadtxt(mname, delimiter=' ')
        p_table = np.loadtxt(pname, delimiter='\t')
        m_table_filt = meniscus_rate2.apply_isolation_forest(m_table)
        #m_table_filt[:,1] - m_table_filt[:,1].min()
        predx,predy,predw = meniscus_rate2.form_model(m_table_filt)
        p_x, l_deriv = meniscus_rate2.calc_derivative(predx, predy)
        deriv_table = np.c_[p_x, l_deriv]
        meniscus_table = pandas.DataFrame(data=m_table_filt,
                                          columns=('frame',
                                                   'pos',
                                                   'area',
                                                   'bbox0',
                                                   'bbox1',
                                                   'bbox2',
                                                   'bbox3')
                                          )
        derivative_table = pandas.DataFrame(data=deriv_table,
                                            columns=('frame',
                                                     'derivative_raw'))
        proboscis_table = pandas.DataFrame(data=p_table,
                                           columns=('frame', 'position_raw'))
        
        proboscis_table['position_abs_p'] = proboscis_table['position_raw'] * lin

        derivative_table['rate'] = derivative_table['derivative_raw'] * vol * framerate

        meniscus_table['position_m'] = meniscus_table['pos'] * lin

        merged1 = pandas.merge(
            proboscis_table,
            meniscus_table,
            how='inner',
            on='frame')

        merged1['submergence'] = merged1['position_abs_p'] - merged1['position_m']

        merged2 = pandas.merge(merged1, derivative_table, how='inner', on='frame')
        #merged2 = pandas.merge(merged1, meniscus_table,on='frame'
        plausible = merged2[merged2['submergence'] > -0.1]

        if len(plausible) > 0:
            correlation = statistics.correlation(plausible['submergence'],
                                             plausible['rate'])
            r_vals.append(correlation)
        

        pyplot.scatter(plausible['submergence'], plausible['rate'], marker='.')
        pyplot.title(corresponding)
        if len(plausible) > 0:
            pyplot.annotate(f"R = {correlation}",
                            (0.5,0.7),
                            xycoords="figure fraction")
        pyplot.xlabel('proboscis submergence depth (mm)')
        pyplot.ylabel('nectar ingestion rate (mL/s)')
        pyplot.savefig('final_plots/{}.png'.format(mystem))
        pyplot.close()
    return r_vals


if __name__ == '__main__':
    r_vals = main()
    pyplot.hist(r_vals)
    pyplot.title("correlation strengths")
    pyplot.ylabel("count")
    pyplot.xlabel("R value")
    pyplot.savefig("final_plots/correlation_strengths.png")
    pyplot.show()


