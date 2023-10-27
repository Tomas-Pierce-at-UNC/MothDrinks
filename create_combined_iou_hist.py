# coding: utf-8
get_ipython().run_line_magic('cd', 'proboscis_utils')
get_ipython().run_line_magic('ls', '')
#get_ipython().run_line_magic('runfile("test_proboscis_model.py")', '')
#runfile
#get_ipython().system('runfile')
get_ipython().run_line_magic('cat', 'test_proboscis_model.py')
import test_proboscis_model as prob
prob_iou = prob.measure_IoU_on_test_set()
#prob_iou = iou
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', 'meniscus_utils')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cat', 'evaluate_meniscus_tracker.py')
dir()
get_ipython().run_line_magic('run', 'evaluate_meniscus_tracker.py')
men_iou = iou
import seaborn
from matplotlib import pyplot
import matplotlib
font = {'family': 'normal',
        'weight': 'normal',
        'size': 30}
matplotlib.rc('font', **font)
#get_ipython().run_line_magic('pinfo', 'pyplot.subplots')
fig, axes = pyplot.subplots(ncols=2, sharey=True)
seaborn.histplot(prob_iou, ax=axes[1])
seaborn.histplot(men_iou, ax=axes[0])

fig.set_tight_layout(True)
#dir(axes[0])
#get_ipython().run_line_magic('pinfo', 'axes[0].set_title')
axes[1].set_title("proboscis")
axes[0].set_title("meniscus")
axes[0].set_xlabel("Intersection \nOver Union")
axes[1].set_xlabel("Intersection \nOver Union")
pyplot.show()
#get_ipython().run_line_magic('save', 'create_combined_iou_hist 1-29')
