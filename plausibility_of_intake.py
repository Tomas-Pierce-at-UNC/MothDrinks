
import json
from loess import loess_1d
from sklearn.ensemble import IsolationForest
import real_unit_convert as ruc
import aggregate_figures as af
import pandas

s_tables, l_tables = af.load_all()
united_s_table = pandas.concat(s_tables)
united_l_table = pandas.concat(l_tables)

l_groups = united_l_table.groupby('filename')
l_maxima = l_groups.max()
l_minima = l_groups.min()

convert = ruc.UnitConversion()

if all(l_maxima.index == l_minima.index):
    movement_pix = l_maxima['centroid_row_m'] - l_minima['centroid_row_m']
    conversions = l_maxima.index.map(convert.get_vol_factor)
    max_intake_possible = movement_pix * conversions
    max_intake_possible.to_csv("max_intake_possible_by_recording.csv")
    #max_intake_possible.
