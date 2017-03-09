import glob
import os
import tensorflow as tf


def get_all_image_training():
    dico = {}

    for filename in glob.iglob('images/**/*.png', recursive=True):
        directory_name = os.path.basename(os.path.dirname(filename))
        if directory_name not in dico:
            dico[directory_name] = []

        dico[directory_name].append(filename)
    return dico


def read_all_file(filenames):
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_png(content)

    image_batch = tf.train.batch(image, batch_size=8)
    print(image_batch)
