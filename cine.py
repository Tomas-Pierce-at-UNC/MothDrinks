# Module to support opening .cine video files

import array
import ctypes
import numpy

# need to check whether optimization attempts helpful
from cine_profiler import print_time

_cine_median = ctypes.cdll.LoadLibrary("./libcine_median.so")
BytePtr = ctypes.POINTER(ctypes.c_ubyte)
_cine_median.video_median.restype = BytePtr
_cine_median.image_size.restype = ctypes.c_int32
_cine_median.image_width.restype = ctypes.c_int32
_cine_median.image_height.restype = ctypes.c_int32
_cine_median.restricted_video_median.restype = BytePtr
_cine_median.read_frame_interop.restype = BytePtr
_cine_median.image_count.restype = ctypes.c_uint32


class Cine:
    # represents a .cine file

    ENDIAN: str = "little"

    def __init__(self, filename):

        self.filename: str = filename
        self.handle = open(filename, "rb")
        self.image_count: int = self.__image_count()
        self.image_width: int = self.__get_image_width()
        self.image_height: int = self.__get_image_height()
        self.image_size: int = self.__get_image_size()
        self.__image_offsets = self.__get_image_offsets()
        self.__to = self.__to_images_offset()

    def __image_count(self) -> int:
        fn = self.handle.fileno()
        fnmbr = ctypes.c_uint32(fn)
        count = _cine_median.image_count(fnmbr)
        return count

    def __to_images_offset(self) -> int:
        self.handle.seek(0x20)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __get_image_offsets(self) -> array.array:
        to = self.__to_images_offset()
        self.handle.seek(to)
        offsets = array.array('Q')
        offsets.fromfile(self.handle, self.image_count)
        return offsets

    def __get_image_width(self) -> int:
        self.handle.seek(0x30)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=True)

    def __get_image_height(self) -> int:
        self.handle.seek(0x34)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __get_image_size(self) -> int:
        self.handle.seek(0x40)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __get_ith_bytes(self, i) -> array.array:
        offset = self.__image_offsets[i]
        self.handle.seek(offset)
        annote_sz_bytes = self.handle.read(4)
        annot_size = int.from_bytes(annote_sz_bytes, self.ENDIAN, signed=False)
        offset += annot_size
        self.handle.seek(offset)
        mybytes = array.array('B')
        mybytes.fromfile(self.handle, self.image_size)
        return mybytes

    def __get_ith_bytes_Rust(self, i) -> array.array:
        fd = self.get_fileno()
        offset = self.__image_offsets[i]
        my_offset = ctypes.c_uint64(offset)
        bptr = _cine_median.read_frame_interop(fd, my_offset)
        data = bptr[:self.image_size]
        sz = ctypes.c_uint64(self.image_size)
        _cine_median.liberate_frame(bptr, sz)
        return data

    def __get_ith_image(self, i) -> numpy.ndarray:
        mybytes = self.__get_ith_bytes_Rust(i)
        data = numpy.array(mybytes, dtype=numpy.uint8)
        shape = (self.image_height, self.image_width)
        shaped = numpy.reshape(data, shape)
        flip = numpy.flip(shaped, 0)
        return flip

    def frames_between(self, start, end, step=1):
        for i in range(start, end, step):
            yield (i, self.__get_ith_image(i))

    def close(self):
        self.handle.close()

    def get_fileno(self) -> int:
        self.handle.close()
        self.handle = open(self.filename, "rb")
        return self.handle.fileno()

    def get_ith_image(self, i):
        if i < 0 or i >= self.image_count:
            raise ValueError("image index out of range")
        return self.__get_ith_image(i)

    def get_video_median(self) -> numpy.ndarray:
        fd = self.get_fileno()
        mybytes = _cine_median.video_median(fd)
        imsize = self.image_size
        data = mybytes[:imsize]
        sz = ctypes.c_uint64(self.image_size)
        _cine_median.liberate_frame(mybytes, sz)
        data = numpy.array(data, dtype=numpy.uint8)
        frame = numpy.reshape(data, (self.image_height, self.image_width))
        frame = numpy.flip(frame, 0)
        return frame

    def get_restricted_video_median(self, start_frame: int, count: int) -> numpy.ndarray:
        fd = self.get_fileno()
        if start_frame >= self.image_count or start_frame+count >= self.image_count:
            ertxt = f"frames go OOB: {start_frame}:{start_frame+count}"
            raise Exception(ertxt)
        mybytes = _cine_median.restricted_video_median(
            fd,
            ctypes.c_uint64(start_frame),
            ctypes.c_uint64(count)
            )
        imsize = self.image_size
        data = mybytes[:imsize]
        sz = ctypes.c_uint64(self.image_size)
        _cine_median.liberate_frame(mybytes, sz)
        data = numpy.array(data, dtype=numpy.uint8)
        frame = numpy.reshape(data, (self.image_height, self.image_width))
        frame = numpy.flip(frame, 0)
        return frame


def test_main():

    from skimage.io import imshow
    from matplotlib import pyplot

    exfile = "data/moth23_2022-02-15_Cine1.cine"
    cin = Cine(exfile)
    import time
    t0 = time.time()
    median_img = cin.get_video_median()
    t1 = time.time()
    print(t1 - t0)

    imshow(median_img)
    pyplot.show()
    cin.close()


if __name__ == '__main__':
    #test_main()
    pass
