import cython
from cython.view cimport array as cvarray


from cine import Cine

import ctypes

from cine_profiler import print_time

siftlib = ctypes.cdll.LoadLibrary("./libsift2.so")


class DescribedKeypoint(ctypes.Structure):
    _fields_ = [("octave", ctypes.c_uint32),
                ("scale", ctypes.c_uint32),
                ("x", ctypes.c_double),
                ("y", ctypes.c_double),
                ("sigma", ctypes.c_double),
                ("orientation", ctypes.c_double),
                ("features", ctypes.c_double * 128)]


class KeyVector(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint64),
                ("keys", ctypes.POINTER(DescribedKeypoint))]


siftlib.find_image_matches.restype = KeyVector


vid = Cine("data2/mothDelta3_2022-09-16_Cine1.cine")

med = vid.get_video_median()

ith = vid.get_ith_image(vid.image_count // 2)

h1, w1 = med.shape
h2, w2 = ith.shape

m_flat = med.flatten(order="C")
i_flat = ith.flatten(order="C")

mbuf = ctypes.create_string_buffer(bytes(m_flat))
ibuf = ctypes.create_string_buffer(bytes(i_flat))


@print_time
def example():
    return siftlib.find_image_matches(w1, h1, w2, h2, mbuf, ibuf)


kv = example()
