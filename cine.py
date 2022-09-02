# Module to support opening .cine video files
import array
import ctypes
import numpy


_cine_median = ctypes.cdll.LoadLibrary("./libcine_median.so")
BytePtr = ctypes.POINTER(ctypes.c_ubyte)
_cine_median.video_median.restype = BytePtr
_cine_median.image_size.restype = ctypes.c_int32
_cine_median.image_width.restype = ctypes.c_int32
_cine_median.image_height.restype = ctypes.c_int32
_cine_median.restricted_video_median.restype = BytePtr


class Histogram:

    # min and max val are both inclusive
    def __init__(self, minval: int, maxval: int):
        self.minval = minval
        self.maxval = maxval
        self.count = 0
        self.data = {}
        for i in range(minval, maxval+1):
            self.data[i] = 0

    def add(self, value: int):
        if value > self.maxval or value < self.minval:
            raise ValueError(f"Value of {value} is outside bounds")
        self.data[value] += 1
        self.count += 1

    def median(self):
        halfpoint = (self.count + 1) // 2
        total = 0
        for value in sorted(self.data.keys()):
            total += self.data[value]
            if total > halfpoint:
                return value
    # nog642 (https://math.stackexchange.com/users/517892/nog642),
    # How to find median from a histogram?, URL (version: 2018-01-04):
    # https://math.stackexchange.com/q/2591986


class Cine:
    # represents a .cine file

    ENDIAN = "little"

    def __init__(self, filename):

        self.filename = filename
        self.handle = open(filename, "rb")
        self.image_count = self.__image_count()
        self.image_width = self.__get_image_width()
        self.image_height = self.__get_image_height()
        self.image_size = self.__get_image_size()
        self.__image_offsets = self.__get_image_offsets()
        self.__to = self.__to_images_offset()

    def __image_count(self):

        self.handle.seek(0x14)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __to_images_offset(self):
        self.handle.seek(0x20)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __get_image_offsets(self):
        to = self.__to_images_offset()
        self.handle.seek(to)
        offsets = array.array('Q')
        offsets.fromfile(self.handle, self.image_count)
        return offsets

    def __get_image_width(self):
        self.handle.seek(0x30)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=True)

    def __get_image_height(self):
        self.handle.seek(0x34)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __get_image_size(self):
        self.handle.seek(0x40)
        mybytes = self.handle.read(4)
        return int.from_bytes(mybytes, self.ENDIAN, signed=False)

    def __get_ith_bytes(self, i):
        offset = self.__image_offsets[i]
        self.handle.seek(offset)
        annote_sz_bytes = self.handle.read(4)
        annot_size = int.from_bytes(annote_sz_bytes, self.ENDIAN, signed=False)
        offset += annot_size
        self.handle.seek(offset)
        mybytes = array.array('B')
        mybytes.fromfile(self.handle, self.image_size)
        return mybytes

    def __get_ith_image(self, i):
        mybytes = self.__get_ith_bytes(i)
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

    def get_fileno(self):
        return self.handle.fileno()

    def get_ith_image(self, i):
        if i < 0 or i >= self.image_count:
            raise ValueError("image index out of range")
        return self.__get_ith_image(i)

    def get_video_median_slow(self):
        z = self.get_ith_image(0)
        shape = z.shape
        histograms = numpy.ndarray(shape=shape, dtype=Histogram)
        for i in range(len(histograms)):
            for j in range(len(histograms[0])):
                histograms[i, j] = Histogram(0, 255)
        for i in range(self.image_count):
            img = self.get_ith_image(i)
            for row, vec in enumerate(img):
                for col, val in enumerate(vec):
                    histograms[row, col].add(val)
        out = numpy.ndarray(shape=shape, dtype=z.dtype)
        for row, vec in enumerate(histograms):
            for col, hist in enumerate(vec):
                val = hist.median()
                out[row, col] = val
        return out

    def get_video_median(self):
        fd = self.get_fileno()
        mybytes = _cine_median.video_median(fd)
        imsize = self.image_size
        data = mybytes[:imsize]
        data = numpy.array(data, dtype=numpy.uint8)
        frame = numpy.reshape(data, (self.image_height, self.image_width))
        frame = numpy.flip(frame, 0)
        return frame

    def get_restricted_video_median(self, start_frame: int, count: int):
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
        data = numpy.array(data, dtype=numpy.uint8)
        frame = numpy.reshape(data, (self.image_height, self.image_width))
        frame = numpy.flip(frame, 0)
        return frame


if __name__ == '__main__':

    from skimage.io import imshow
    from matplotlib import pyplot

    exfile = "data/moth23_2022-02-15_Cine1.cine"
    cin = Cine(exfile)
    median_img = cin.get_video_median()

    imshow(median_img)
    pyplot.show()
