
import tube2
import cine
import glob
from pathlib import Path
from skimage import io as skio

cine_names = glob.glob("data2/*.cine")

for name in cine_names:
    video = cine.Cine(name)
    med = video.get_video_median()
    p = Path(name)
    newname = "medtubes/" + p.stem + ".png"
    tube = tube2.tube_crop1(med)
    skio.imsave(newname, tube)
