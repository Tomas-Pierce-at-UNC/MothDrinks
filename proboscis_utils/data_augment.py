
from tensorflow import keras
import math
import numpy as np
from skimage import transform, util


class ImageSeq(keras.utils.Sequence):

    def __init__(self, ins: np.ndarray, outs: np.ndarray, batch=32):
        assert ins.shape == outs.shape
        self.inputs = np.copy(ins)
        self.outputs = np.copy(outs)
        self.batch = batch

    def __len__(self):
        return math.ceil(len(self.inputs) / self.batch)

    def __getitem__(self, idx):
        i = self.inputs[idx * self.batch: (idx + 1) * self.batch]
        o = self.outputs[idx * self.batch: (idx + 1) * self.batch]
        return i.astype(np.uint8), o


class RepeatSeq(keras.utils.Sequence):

    def __init__(self, wrapped, repeats: int):
        self.wrapped = wrapped
        self.repeats = repeats

    def __len__(self):
        return len(self.wrapped) * self.repeats

    def __getitem__(self, idx):
        rel_idx = idx % len(self.wrapped)
        i, o = self.wrapped[rel_idx]
        return np.copy(i), np.copy(o)


class RandomFlip(keras.utils.Sequence):

    def __init__(self, wrapped, thresh=0.4):
        self.wrapped = wrapped
        self.thresh = thresh
        self.should_flip = np.random.random(len(wrapped))

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, idx):
        i, o = self.wrapped[idx]
        give_i = i[:]
        give_o = o[:]
        if self.should_flip[idx] > self.thresh:
            give_i = np.flip(give_i, axis=1)
            give_o = np.flip(give_o, axis=1)
        return give_i, give_o


class RandomXShift(keras.utils.Sequence):

    def __init__(self, wrapped, thresh=0.4):
        self.wrapped = wrapped
        self.move_likely = np.random.random(len(wrapped))
        self.thresh = thresh
        self.move_amount = np.random.randint(-20, 20, len(wrapped))

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, idx):
        i, o = self.wrapped[idx]
        give_i = i[:]
        give_o = o[:]
        if self.move_likely[idx] > self.thresh:
            move = self.move_amount[idx]
            transl = transform.EuclideanTransform(translation=[move, 0])
            for j in range(len(give_i)):
                give_i[j] = util.img_as_ubyte(
                    transform.warp(give_i[j], transl.inverse))
                give_o[j] = util.img_as_ubyte(
                    transform.warp(give_o[j], transl.inverse))
        return give_i, give_o


class RandomYShift(keras.utils.Sequence):

    def __init__(self, wrapped, thresh=0.4):
        self.wrapped = wrapped
        self.move_likely = np.random.random(len(wrapped))
        self.thresh = thresh
        self.move_amount = np.random.randint(-200, 200, len(wrapped))

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, idx):
        i, o = self.wrapped[idx]
        give_i = i[:]
        give_o = o[:]
        if self.move_likely[idx] > self.thresh:
            move = self.move_amount[idx]
            transl = transform.EuclideanTransform(translation=[0, move])
            for j in range(len(give_i)):
                give_i[j] = util.img_as_ubyte(
                    transform.warp(give_i[j], transl.inverse))
                give_o[j] = util.img_as_ubyte(
                    transform.warp(give_o[j], transl.inverse))
        return give_i, give_o
