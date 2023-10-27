
import cine
from matplotlib import pyplot
import tube2
import keras
import numpy as np
from skimage import io

model = keras.models.load_model("meniscus_utils/meniscus_track_4")
FILENAME = "data2/unsuitableVideos/moth22_2022_02_04_Cine1.cine"
PAUSE = 0.5
#fig, ax = pyplot.subplots()
try:
    video = cine.Cine(FILENAME)
    print(video.image_count)
    f_rate = video.framerate
    pause = 1 / f_rate
    for i in range(0, video.image_count - 16, 16):
        batch = []
        for j in range(i, i + 16):
            f = video.get_ith_image(j)
            frame = tube2.tube_crop1(f)
            batch.append(frame)
        batch = np.array(batch)[..., np.newaxis]
        preds = model(batch)
        p = preds.numpy()
        above = p > 0.5
##        f = video.get_ith_image(i)
##        frame = tube2.tube_crop1(f)
##        ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
##        fig.show()
##        pyplot.pause(pause)
##        fig.clear()
##        fig.add_axes(ax)
        for k in range(p.shape[0]):
            if above[k,:,:,0].sum() > 0:
                io.imshow(above[k,:,:,0])
                pyplot.title(f"{i + k}")
                io.show()
            #pyplot.pause(PAUSE)
            #fig.clear()
            #fig.add_axes(ax)

finally:
    video.close()
    pyplot.close()
