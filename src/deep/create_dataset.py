import itertools
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _read_image(filename):
    """
    load image from disk
    :param filename: path of images
    :return: image match to file name path
    """
    return mpimg.imread(filename)


def _load_image(filenames):
    """
    load images from and array of path
    :param filenames: array of file names to load
    :return: an array of images
    """
    images = [*map(lambda x: np.asarray([_read_image(x)]).reshape((32, 32, 1)), filenames)]
    return images


def _one_hot_label(labels):
    """
    generate an array of one hot labels based on original labels class
    :param labels: original array of labels
    :return: one hot array
    """
    one_hot_labels = []
    for num in labels:
        i = round(num)
        one_hot = [0] * 3
        one_hot[i-2] = 1
        one_hot_labels.append(one_hot)
    return np.array(one_hot_labels).astype(np.int32)


def _iterate_path(path):
    """
    iterate path and sub path to generate an two array of images and labels
    :param path: base path to investigate
    :return: two array of file names and labels
    """
    filenames_array = []
    labels_array = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filenames_array.append(os.path.join(root, name))
            labels_array.append(int(os.path.basename(os.path.normpath(root))))
    # labels_array = _one_hot_label(labels_array)
    return filenames_array, labels_array


def _generate_dataset(base_path):
    """
    produce a dataset for neural network from path
    :param base_path: base path of dataset images
    :return: two array with file name and labels
    """
    filenames_array, labels_array = _iterate_path(base_path)
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
    """
    plot confusion matrix
    :param cm: confusion matrix
    :param classes: classes of evaluation
    :param normalize: flag to enable normalization step
    :param title: title of plot
    :param cmap: color map
    :return: a chart of confusion matrix
    """
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
