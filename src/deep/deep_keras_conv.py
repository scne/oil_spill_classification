# base libraries
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import signal
import skimage.measure
import matplotlib.cm as cm

# keras import
from keras import backend as K
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, Embedding
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
# sklearn import
from sklearn.model_selection import train_test_split

# utility function
from src.deep.create_dataset import _generate_dataset, _load_image
from mpl_toolkits.axes_grid1 import make_axes_locatable

# path to save files
path_dataset = './deep/img_cluster'  # dataset path
path_model = './deep/model/'  # path to save model
path_board = './deep/tensorboard'

# convolutional network parmas
batch_size = 64  # training cases batch
num_epochs = 500  # max number of epochs
kernel_size = 3  # kernel size dimension
pool_size = 2  # max pooling size
seed = 42  # base random seed

# dataset image parameters
height = 32
width = 32
depth = 1


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.get_cmap('jet')
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)


def load_data_cnn():

    filenames, labels = _generate_dataset(path_dataset)
    x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=seed, stratify=labels)
    X_train = np.asarray(_load_image(x_train))
    X_test = np.asarray(_load_image(x_test))
    num_classes = np.unique(y_train).shape[0]
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)

    return X_train, X_test, Y_train, Y_test


def evaluete_cnn(X_test, Y_test):

    model = load_model(path_model+'model.h5')

    model.evaluate(X_test, Y_test, verbose=1, sample_weight=None)  # Evaluate the trained model on the test set!

    score = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

    y_predict = np.asarray(model.predict(X_test, verbose=0))
    Y_predict = np.argmax(y_predict, axis=1)
    y_test = np.argmax(Y_test, axis=1)
    confmatrix = confusion_matrix(y_test, Y_predict)
    print("\n Confusion Matrix :")
    print(confmatrix)
    class_names = ["0", "1", '2']
    # plt.figure()
    # _plot_confusion_matrix(confmatrix, classes=class_names, title='Confusion matrix')

    print("\nMetrics => ", model.metrics_names, score)
    print('\nClassification Report : ')
    print(classification_report(y_test, Y_predict, target_names=class_names))
    # plt.show()



def start_cnn ():
    print('STARTING FITTING CONVOLUTIONAL DEEP NEURAL NETWORK')
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    if os.path.exists(path_board):
        shutil.rmtree(path_board)
    os.mkdir(path_board)

    X_train, X_test, Y_train, Y_test = load_data_cnn()
    print('loading data .......')
    #DEFINIZIONE DEL MODELLO DELLA CONVOLUTIONAL DEEP NEURAL NETWORK
    # inp = Input(shape=(height, width, depth))
    # conv_1 = Convolution2D(32, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    # pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    # conv_2 = Convolution2D(64, (kernel_size, kernel_size), padding='same', activation='relu')(pool_1)
    # pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    # drop_1 = Dropout(0.25)(pool_2)
    # flat = Flatten()(drop_1)
    # # hidden_1 = Dense(4096, activation='relu')(flat)
    # hidden_2 = Dense(1024, activation='relu')(flat)
    # drop_2 = Dropout(0.4)(hidden_2)
    # out = Dense(3, activation='softmax')(drop_2)

    model = Sequential()
    conv_1 = Conv2D(32, kernel_size=(kernel_size, kernel_size), padding='same', input_shape=(height, width, depth),
                    name='conv1')
    model.add(conv_1)
    convout1 = Activation('elu')
    model.add(convout1)
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(kernel_size, kernel_size), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    #CREAZIONE DEL MODELLO E COMPILAZIONE
    # model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    #STARTING FITTING
    print('start .... ')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_split=0.1,
              callbacks=[TensorBoard(log_dir='./deep/tensorboard/', histogram_freq=1, write_images=True,
                                     write_grads=True, embeddings_layer_names=['conv1'], embeddings_freq=1),
                         EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2,
                                       verbose=1, mode='auto')])

    # convout1_f = K.function(model.inputs, [conv_1.output])

    # weight_all = model.get_weights()

    # plt.figure()
    # plt.title('input')
    # nice_imshow(plt.gca(), np.squeeze(X), vmin=0, vmax=1, cmap='binary')

    # W = model.layers[0]
    # W = W.get_value(borrow=True)
    # W = np.squeeze(W)
    # print("W shape : ", W.shape)
    #
    # plt.figure(figsize=(15, 15))
    # plt.title('conv1 weights')
    # nice_imshow(plt.gca(), make_mosaic(W, 6, 6), cmap=cm.get_cmap('binary'))

    # C1 = convout1_f([X])
    # C1 = np.squeeze(C1)
    # print("C1 shape : ", C1.shape)
    #
    # plt.figure(figsize=(32, 32))
    # plt.suptitle('convout1')
    # nice_imshow(plt.gca(), make_mosaic(C1, 6, 6), cmap=cm.get_cmap('binary'))


    #SALVO IL MODELLO APPENA ADDESTRATO
    print("Saving model")
    model.save(path_model+'model.h5', overwrite=True)

    #PASSO ALLA FASE DI VALUTAZIONE DEL MODELLO SUI TEST CASES
    evaluete_cnn(X_test, Y_test)


    # Visualize the first layer of convolutions on an input image

    i = 500
    img = np.reshape(X_test[i], [32, 32])

    plt.figure()
    plt.title('Img Original 32x32')
    plt.imshow(img, cmap=cm.get_cmap('binary'))

    weight = np.asarray(model.get_weights()[0]).reshape(32, 3, 3)
    plt.figure()
    plt.title('32 Conv1 Kernels 3x3')
    nice_imshow(plt.gca(), make_mosaic(weight, 4, 8), cmap=cm.get_cmap('binary'))

    weight1 = np.asarray(model.get_weights()[2]).reshape(32, 64, 3, 3)
    plt.figure()
    plt.title('64 Conv2 Kernels 3x3')
    nice_imshow(plt.gca(), make_mosaic(weight1[0], 8, 8), cmap=cm.get_cmap('binary'))

    conv_imgs = []
    for i in range(32):
        conv_imgs.append(signal.convolve2d(img, weight[i], mode='same'))
    conv_imgs = np.asarray(conv_imgs)

    plt.figure()
    plt.title('Output Conv1 Layer')
    nice_imshow(plt.gca(), make_mosaic(conv_imgs, 4, 8), cmap=cm.get_cmap('binary'))

    conv_imgs_pool = []
    # maxpooling
    for i in range(32):
        conv_imgs_pool.append(skimage.measure.block_reduce(conv_imgs[i], (2, 2), np.max))
    conv_imgs = np.asarray(conv_imgs_pool)
    plt.figure()
    plt.title('Img Reduced 16x16')
    plt.imshow(conv_imgs[0], cmap=cm.get_cmap('binary'))

    conv2_imgs = []
    for i in range(64):
        conv2_imgs.append(signal.convolve2d(conv_imgs[0], weight1[0, i], mode='same'))

    conv2_imgs = np.asarray(conv2_imgs)
    plt.figure()
    plt.title('Output Conv2 Layer')
    nice_imshow(plt.gca(), make_mosaic(conv2_imgs, 8, 8), cmap=cm.get_cmap('binary'))
    plt.show()

    print("The End")
