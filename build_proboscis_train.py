
import glob
import random

from skimage import transform, util
from skimage import io as skio

import cine
import tube2

names = glob.glob("data2/**/*.cine", recursive=True)

for i, name in enumerate(names):
    video = cine.Cine(name)
    for j in range(10):
        idx = random.randint(0, video.image_count - 1)
        frame = video.get_ith_image(idx)
        crop1 = tube2.tube_crop1(frame)
        try:
            img = util.img_as_ubyte(transform.resize(crop1, (600,100)))
        except Exception as e:
            print(e)
            continue
        name = f"proboscis_data/img{i}_{j}.png"
        skio.imsave(name, img)
    video.close()
