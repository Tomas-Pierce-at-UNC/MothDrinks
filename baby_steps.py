
import tensorflow as tf

FILENAME = "/home/tomas/Projects/DrinkMoth2/annotatedMeniscus/MeniscusTracks/default.tfrecord"

filename_queue = tf.train.string_input_producer([FILENAME], num_epochs=1)

reader = tf.TFRecordReader()

key, serialized_example = reader.read(filename_queue)

