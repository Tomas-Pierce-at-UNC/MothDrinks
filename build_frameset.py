
import random
import glob

from cine import Cine
import tube2
import align

from skimage.io import imsave
from numpy import int16

videonames = glob.glob("data2/*.cine")
videonames2 = glob.glob("videos_surface_bug/*.cine")
videonames.extend(videonames2)

for i, name in enumerate(videonames):
    vid = Cine(name)
    med = vid.get_video_median()
    med = med.astype(int16)
    tb = tube2.get_tube(med)
    res_med = tube2.apply_bounds(med, tb)
    try:
        image_count = vid.image_count
        index1 = random.randint(0, image_count - 1)
        index2 = random.randint(0, image_count - 1)
        index3 = random.randint(0, image_count - 1)
        index4 = random.randint(0, image_count - 1)
        index5 = random.randint(0, image_count - 1)
        img1 = vid.get_ith_image(index1)
        img2 = vid.get_ith_image(index2)
        img3 = vid.get_ith_image(index3)
        img4 = vid.get_ith_image(index4)
        img5 = vid.get_ith_image(index5)

        res1 = tube2.constrain_to_tube_refwidth(img1, res_med)
        res2 = tube2.constrain_to_tube_refwidth(img2, res_med)
        res3 = tube2.constrain_to_tube_refwidth(img3, res_med)
        res4 = tube2.constrain_to_tube_refwidth(img4, res_med)
        res5 = tube2.constrain_to_tube_refwidth(img5, res_med)

        # dif1 = res1.astype(int16) - res_med.astype(int16)
        # dif2 = res2.astype(int16) - res_med.astype(int16)
        # dif3 = res3.astype(int16) - res_med.astype(int16)

        name = "frameset/img{}_{}.png"
        
        imsave(name.format(i, 1), res1)
        imsave(name.format(i, 2), res2)
        imsave(name.format(i, 3), res3)
        imsave(name.format(i, 4), res4)
        imsave(name.format(i, 5), res5)

        # imsave(name.format(i, 1), dif1)
        # imsave(name.format(i, 2), dif2)
        # imsave(name.format(i, 3), dif3)
    finally:
        vid.close()
