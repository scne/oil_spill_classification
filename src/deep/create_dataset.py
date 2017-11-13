import tensorflow as tf
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools



def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    return image_decoded, label


def _read_image(filename):
    # image_string = tf.read_file(filename)
    # image_decoded = tf.image.decode_png(image_string, channels=1)
    return mpimg.imread(filename)

def _load_image(filenames):
    images = [*map(lambda x: np.asarray([_read_image(x)]).reshape((32,32,1)), filenames)]
    # images = [*map(lambda x: np.asarray(_read_image(x)), filenames)]
    return images

def _one_hot_label(labels):
    one_hot_labels = []
    for num in labels:
        i = round(num)
        one_hot = [0] * 3
        one_hot[i-2] = 1
        one_hot_labels.append(one_hot)
    return np.array(one_hot_labels).astype(np.int32)


def _iterate_path(path):
    filenames_array = []
    labels_array = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filenames_array.append(os.path.join(root, name))
            labels_array.append(int(os.path.basename(os.path.normpath(root))))
    # labels_array = _one_hot_label(labels_array)
    return filenames_array, labels_array


def _generate_dataset(base_path):
    filenames_array, labels_array = _iterate_path(base_path)
    # images = [*map(lambda x: _read_image(x), filenames_array)]
    # filenames = tf.constant(filenames_array)
    # labels = tf.constant(labels_array)
    # dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    # return dataset.map(_parse_function)
    return filenames_array, labels_array


def _extract_class(labels, name_class):
    labels = np.array(labels)
    labels_select = np.zeros(len(labels))
    # idx = np.where(labels != name_class)[0]
    # labels_select[idx] = 0
    idx = np.where(labels == name_class)[0]
    labels_select[idx] = 1
    return labels_select

def _plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# _generate_dataset('/home/claudio/Dropbox/ImageProcessing_project/crop')