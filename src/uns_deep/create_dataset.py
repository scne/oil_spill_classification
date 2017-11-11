import tensorflow as tf
import os
import numpy as np
import matplotlib.image as mpimg

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    return image_decoded, label


def _read_image(filename):
    # image_string = tf.read_file(filename)
    # image_decoded = tf.image.decode_png(image_string, channels=1)
    return mpimg.imread(filename)

def _load_image(filenames):
    images = [*map(lambda x: np.asarray(_read_image(x)), filenames)]
    return images

def _one_hot_label(labels):
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 3
        one_hot[num-2] = 1.0
        one_hot_labels.append(one_hot)
    return np.array(one_hot_labels).astype(np.float32)


def _iterate_path(path):
    filenames_array = []
    labels_array = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filenames_array.append(os.path.join(root, name))
            #labels_array.append(int(os.path.basename(os.path.normpath(root))))
    #labels_array = _one_hot_label(labels_array)
    return filenames_array, labels_array


def _generate_dataset(base_path):
    filenames_array, labels_array = _iterate_path(base_path)
    # images = [*map(lambda x: _read_image(x), filenames_array)]
    # filenames = tf.constant(filenames_array)
    # labels = tf.constant(labels_array)
    # dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    # return dataset.map(_parse_function)
    return filenames_array, labels_array


# _generate_dataset('/home/claudio/Dropbox/ImageProcessing_project/crop')
