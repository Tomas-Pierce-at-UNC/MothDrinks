
import pandas as pd
import re
from matplotlib import pyplot
import seaborn
import statistics

EXPR = re.compile("\d{4}-\d{2}-\d{2}")
EXPR2 = re.compile("\d{4}_\d{2}_\d{2}")



def get_moth_name(filename: str)->str:
    start = filename.lower().index("moth")
    end = filename.index("_", start)
    return filename[start+4:end]

def get_date(filename: str) -> str:
    dates = EXPR.findall(filename)
    if len(dates) < 1:
        dates = EXPR2.findall(filename)
        return dates[0].replace("_", "-")
    return dates[0]


max_vid = pd.read_csv("max_intake_possible_by_recording.csv")
max_vid['intake_vol'] = max_vid['0']
del max_vid['0']
moth_maxes = pd.read_excel("MassTable.ods", engine="odf")
max_vid['moth ID'] = max_vid.filename.map(get_moth_name)
max_vid['date'] = max_vid.filename.map(get_date)

max_vid['moth ID'] = max_vid['moth ID'].astype('|S')
moth_maxes['moth ID'] = moth_maxes['moth ID'].astype('|S')

moth_maxes['date'] = moth_maxes['date'].astype(str)

bydaymass = moth_maxes.groupby(by=['moth ID', 'date']).sum()
bydayvol = max_vid.groupby(by=['moth ID', 'date']).sum()

merged = bydaymass.merge(bydayvol, how='inner', left_index=True, right_index=True)
# remove points with missing data that causes errors
merged = merged[merged['initial mass'] > 0]

corr = statistics.correlation(merged['difference'], merged['intake_vol'])
print(corr)

seaborn.lmplot(merged, x='difference', y='intake_vol')
ax = pyplot.gca()
maxdif = max(merged['difference'])
mindif = min(merged['difference'])
pyplot.axline((0,0), slope=1, linestyle='dotted')
#pyplot.plot(0, 0, maxdif, maxdif, color='black', linestyle='dotted')
#pyplot.plot(merged['difference'], merged['difference'], color='black', linestyle='dashed')
pyplot.xlabel("Mass Difference (g)")
pyplot.ylabel("Intake Volume (mL)")
#pyplot.title("verify correctness of measurement")
pyplot.show()
#merged = moth_maxes.merge(max_vid, how='outer', on=('moth ID', 'date'))
