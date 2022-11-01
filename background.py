import numpy as np
from skimage import util, restoration


def subtract_background(image: np.ndarray):
    inverted = util.invert(image)
    inv_bg = restoration.rolling_ball(inverted, radius=45)
    # imshow(inv_bg)
    # pyplot.show()
    inv_dif = inverted - inv_bg
    dif = util.invert(inv_dif)
    return dif


if __name__ == '__main__':
    import cine
    from skimage.io import imshow
    from matplotlib import pyplot
    c = cine.Cine("data2/mothM6_2022-09-27_Cine1.cine")
    z = c.get_ith_image(0)
    c.close()
    inv = util.invert(z)
    inv_bg = restoration.rolling_ball(inv, radius=50)
    inv_dif = inv - inv_bg
    dif = util.invert(inv_dif)
    bg = util.invert(inv_bg)
    imshow(bg)
    pyplot.show()