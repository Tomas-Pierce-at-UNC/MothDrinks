# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun  6 10:16:37 2023

# @author: tomas
# """

# import numpy as np
# from matplotlib import pyplot
# import pathlib
# import glob

# from sklearn.ensemble import IsolationForest
# from loess import loess_1d

# #from meniscus_rate2 import calc_derivative
# import meniscus_rate2

# import statistics

# from cine import Cine

# import real_unit_convert

# meniscus_folder = pathlib.Path("meniscusTrackNN")
# proboscis_folder = pathlib.Path("proboscis_canidates")

# video_folder = pathlib.Path("data2")
# video_folder2 = pathlib.Path("data2/unsuitableVideos")

# video_names = list(video_folder.glob("*.cine"))
# video_names.extend(video_folder2.glob("*.cine"))

# v_stems = [vid.stem for vid in video_names]


# m_sheets = list(meniscus_folder.glob("*.csv"))
# p_sheets = list(proboscis_folder.glob("*.tsv"))

# m_stems = [sheet.stem for sheet in m_sheets]
# p_stems = [sheet.stem for sheet in p_sheets]

# crossover = [p_stems.index(m_stem) for m_stem in m_stems]

# v_crossover = [v_stems.index(m_stem) for m_stem in m_stems]

# cines = [Cine(name) for name in video_names]
# framerates = [cine.framerate for cine in cines]
# [cine.close() for cine in cines]

# #print(m_sheets[5])
# #print(p_sheets[crossover[5]])

# m_tables = [np.loadtxt(name, dtype=float, delimiter=' ') for name in m_sheets]
# p_tables = [np.loadtxt(name, dtype=float, delimiter='\t') for name in p_sheets]

# unit_converter = real_unit_convert.UnitConversion()

# for i, m_table in enumerate(m_tables):
#     name = m_sheets[i]
#     vname = meniscus_rate2.get_corresponding_name(name)
    
#     p_table = p_tables[crossover[i]]
#     framerate = framerates[v_crossover[i]]
#     try:
#         m_table = meniscus_rate2.apply_isolation_forest(m_table)
#     except Exception as e:
#         print(e)
#         print(m_sheets[i])
#         continue
#     try:
#         predx,predy,predw = loess_1d.loess_1d(m_table[:,0], m_table[:,1])
#     except Exception as e:
#         print(e)
#         print(m_sheets[i])
#         continue
    
#     p_x, deriv = meniscus_rate2.calc_derivative(predx, predy)
#     deriv = deriv * framerate
#     dtable = np.c_[p_x, deriv]
    
#     rates = []
#     subs = []
    
#     for i, x_row in enumerate(p_x):
#         rate = deriv[i]
#         sub_proboscis = p_table[p_table[:,0] == x_row]
#         try:
#             tip_pos = max(sub_proboscis[:,3])
#         except ValueError as e:
#             print(e)
#             continue
#         rates.append(rate)
#         subs.append(tip_pos)
    
#     pyplot.scatter(subs, rates)
#     #covar = statistics.covariance(subs, rates)
#     #print(covar)
#     try:
#         corr = statistics.correlation(subs, rates)
#         print(corr)
#     except ZeroDivisionError as e:
#         print(e)
#     except statistics.StatisticsError as e:
#         print(e)
#     pyplot.show()
    