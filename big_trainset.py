import tube2
import cine
import random
import glob
from skimage import transform, util
from skimage import io as skio

videonames = glob.glob("data2/*.cine")

fcount = 0

for name in videonames:

    vid = cine.Cine(name)

    count = vid.image_count

    frame_indexes = random.sample(range(0, count), 20)

    for idx in frame_indexes:

        image = vid.get_ith_image(idx)

        tb1 = tube2.tube_crop1(image)
        
        shaped = util.img_as_ubyte(transform.resize(tb1, (600, 100)))

        skio.imsave("big/example{:07}.png".format(fcount), shaped)

        fcount += 1

    vid.close()
