
import meniscus
from pathlib import Path
import cine
import numpy as np

datapath = Path("data2/")
files = sorted(datapath.glob("*.cine"))

for file in files:
    try:
        video = cine.Cine(str(file))
        data = meniscus.find_meniscus(video, 0, video.image_count - 1)
        out = Path("men_data/")
        outfile = out / (str(file.stem) + ".tsv")
        np.savetxt(outfile, data, delimiter='\t')
        video.close()
    except Exception as e:
        print(e)
        continue
