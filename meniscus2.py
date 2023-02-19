
# this file exists for the sake of working around a
# bug that occurs with some videos regarding their length

# may have fixed it by addition of line in cine.py
import glob
import meniscus


vidnames = glob.glob("videos_surface_bug/*.cine")

for vidname in vidnames:
    cine = meniscus.Cine(vidname)
    meniscus.additive_measure_video(cine)
    print(vidname)