# base libraries
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
# keras import
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
# sklearn import
from sklearn.model_selection import train_test_split

# utility function
from src.deep.create_dataset import _generate_dataset, _load_image

# path to save files
path_dataset = './deep/img_cluster'  # dataset path
path_model = './deep/model/'  # path to save model

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


def _load_data_cnn():

    filenames, labels = _generate_dataset(path_dataset)
    x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=seed, stratify=labels)
    X_train = np.asarray(_load_image(x_train))
    X_test = np.asarray(_load_image(x_test))
    num_classes = np.unique(y_train).shape[0]
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)

    return X_train, X_test, Y_train, Y_test


def _evaluete_cnn(X_test, Y_test):

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
    plt.figure()
    # _plot_confusion_matrix(confmatrix, classes=class_names, title='Confusion matrix')

    print("\nMetrics => ", model.metrics_names, score)
    print('\nClassification Report : ')
    print(classification_report(y_test, Y_predict, target_names=class_names))
    # plt.show()



def _start_cnn ():
    print('STARTING FITTING CONVOLUTIONAL DEEP NEURAL NETWORK')
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    X_train, X_test, Y_train, Y_test = _load_data_cnn()
    print('loading data .......')
    #DEFINIZIONE DEL MODELLO DELLA CONVOLUTIONAL DEEP NEURAL NETWORK
    inp = Input(shape=(height, width, depth))
    conv_1 = Convolution2D(32, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    conv_2 = Convolution2D(64, (kernel_size, kernel_size), padding='same', activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(0.25)(pool_2)
    flat = Flatten()(drop_1)
    # hidden_1 = Dense(4096, activation='relu')(flat)
    hidden_2 = Dense(1024, activation='relu')(flat)
    drop_2 = Dropout(0.4)(hidden_2)
    out = Dense(3, activation='softmax')(drop_2)

    #CREAZIONE DEL MODELLO E COMPILAZIONE
    model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # print(model.summary())
    #STARTING FITTING
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1,
              callbacks=[TensorBoard(log_dir='./deep/tensorboard/'),
                         EarlyStopping(monitor='loss', min_delta=0.1, patience=10,
                                       verbose=0, mode='auto')])

    #SALVO IL MODELLO APPENA ADDESTRATO
    print("Saving model")
    model.save(path_model+'model.h5', overwrite=True)
    #PASSO ALLA FASE DI VALUTAZIONE DEL MODELLO SUI TEST CASES
    _evaluete_cnn(X_test, Y_test)
    print("The End")
