
import random
import glob

from cine import Cine
import tube2
import align

from skimage.io import imsave

videonames = glob.glob("data2/*.cine")
videonames2 = glob.glob("videos_surface_bug/*.cine")
videonames.extend(videonames2)

for i, name in enumerate(videonames):
    vid = Cine(name)
    try:
        image_count = vid.image_count
        index1 = random.randint(0, image_count - 1)
        index2 = random.randint(0, image_count - 1)
        index3 = random.randint(0, image_count - 1)
        img1 = vid.get_ith_image(index1)
        img2 = vid.get_ith_image(index2)
        img3 = vid.get_ith_image(index3)

        tb1 = tube2.get_tube(img1)
        tb2 = tube2.get_tube(img2)
        tb3 = tube2.get_tube(img3)

        res1 = tube2.apply_bounds(img1, tb1)
        res2 = tube2.apply_bounds(img2, tb2)
        res3 = tube2.apply_bounds(img3, tb3)

        name = "training/img{}_{}.png"
        imsave(name.format(i, 1), res1)
        imsave(name.format(i, 2), res2)
        imsave(name.format(i, 3), res3)
    except:
        pass
    finally:
        vid.close()