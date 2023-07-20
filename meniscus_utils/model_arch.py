
from tensorflow import keras
import tensorflow as tf
import sklearn
from skimage import draw
from skimage.io import imshow, imread
from skimage import morphology
from skimage import transform
from skimage import util
import numpy as np
from matplotlib import pyplot
# from tensorflow import keras
# import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model
from tensorflow.keras.utils import plot_model
tf.keras.backend.set_image_data_format('channels_last')

# from https://blog.paperspace.com/unet-architecture-image-segmentation/


def convolution_block(input_lay, filters=64):
    conv1 = layers.Conv2D(filters, kernel_size=(3, 3),
                          padding="same")(input_lay)
    bn1 = layers.BatchNormalization()(conv1)
    act1 = layers.ReLU()(bn1)

    conv2 = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(act1)
    bn2 = layers.BatchNormalization()(conv2)
    act2 = layers.ReLU()(bn2)

    return act2


def encoder(input_lay, filters=64):
    enc1 = convolution_block(input_lay, filters)
    max_pool1 = layers.MaxPooling2D()(enc1)
    return enc1, max_pool1


def decoder(input_lay, skip_lay, filters=64):
    upsample = layers.Conv2DTranspose(
        filters, (2, 2), strides=2, padding="same")(input_lay)
    connect_skip = layers.Concatenate()([upsample, skip_lay])
    out = convolution_block(connect_skip, filters)
    return out


def u_net(image_size):
    input1 = layers.Input(image_size)

    skip1, enc1 = encoder(input1, 64)
    skip2, enc2 = encoder(enc1, 64 * 2)
    skip3, enc3 = encoder(enc2, 64 * 4)
    skip4, enc4 = encoder(enc3, 64 * 8)

    conv_block = convolution_block(enc4, 64*16)

    dec1 = decoder(conv_block, skip4, 64 * 8)
    dec2 = decoder(dec1, skip3, 64 * 4)
    dec3 = decoder(dec2, skip2, 64 * 2)
    dec4 = decoder(dec3, skip1, 64)

    out = layers.Conv2D(filters=1, kernel_size=(
        1, 1), padding="same", activation="sigmoid")(dec4)

    model = Model(input1, out)

    return model


def our_model():
    inp = layers.Input(shape=[600, 100, 1])
    res = layers.Rescaling(1.0/255)(inp)
    # x = layers.Conv2D(filters=4, kernel_size=(3,3))(res)

    skip1, enc1 = encoder(res, 64)
    skip2, enc2 = encoder(enc1, 64 * 2)
    # skip3, enc3 = encoder(enc2, 64 * 4)

    conv_block = convolution_block(enc2, 64 * 4)

    dec1 = decoder(conv_block, skip2, 64 * 2)
    dec2 = decoder(dec1, skip1, 64)
    # dec3 = decoder(dec2, skip1, 64)

    f = layers.Flatten()(dec2)

    # out = layers.Dense(1)(f)
    out = layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(dec2)

    model = Model(inp, out)
    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer=optimizers.Adam(use_ema=True,
                  weight_decay=0.0005)) # see Krizhevsky ImageNet
    #plot_model(model)
    return model
