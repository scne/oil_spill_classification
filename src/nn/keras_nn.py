# standard libraries
import os
import shutil
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
# keras libraries
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
# sklearn libraries
from sklearn.model_selection import train_test_split

# convolutional network parmas
path_dataset = './nn/dataset/dataset_total.txt'  # datasetpath
path_best = './nn/best_model/'  # kfolds model path
path_thebest = './nn/thebetter_model/'  # bset models path

# neural network params
batch_size = 32  # training cases batch
num_epochs = 500  # max number of epochs
num_classes = 3  # number of class in dataset
seed = 42  # base random seed
n_splits = 10  # number of kfold
n_input_layer = 31

def _load_data_nn():
    """
    generate dataset based on data in dataset folder
    :return: train and test dataset based on stratification strategy
    """
    dataset = np.loadtxt(path_dataset, delimiter='\t')

    y = np.array(np.ceil(dataset[:, -1])).astype(np.str)
    X = np.array(dataset[:, :-1]).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)

    scaler = preprocessing.StandardScaler().fit(x_train)

    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)

    y_train = np.subtract(y_train.reshape((len(y_train), 1)).astype(np.float32), np.asarray(2.0))
    y_test = np.subtract(y_test.reshape((len(y_test), 1)).astype(np.float32), np.asarray(2.0))

    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    return X_train, X_test, Y_train, Y_test


def _evaluete_nn(X_test, Y_test, best_model):
    """
    Evaluate best model after kfold training
    :param X_test: example images to test best model after kfold
    :param Y_test: labels matching truth to example
    :param best_model: index which identify best model after kfold
    :return: evaluation of best model trough dataset test and save it with loss and accuracy metrics
    """

    # load best model and evaluate it with accuracy and loss
    print('Load best model and test it ')
    model = load_model(path_best+'checkpoint-%d.h5' %(best_model))
    score = model.evaluate(X_test, Y_test, verbose=0)  # evaluate model
    print('Metrics => ', model.metrics_names, score)
    y_predict = np.asarray(model.predict(X_test, verbose=0))
    Y_predict = np.argmax(y_predict, axis=1)
    y_test = np.argmax(Y_test, axis=1)
    confmatrix = confusion_matrix(y_test, Y_predict)
    print("\nConfusion Matrix :")
    print(confmatrix)
    class_names = ["0", "1", '2']
    print("\nMetrics => ", model.metrics_names, score)
    print('\nClassification Report : ')
    print(classification_report(y_test, Y_predict, target_names=class_names))
    # save model tested with loss and accuracy
    model.save(path_thebest+'model-'+'{:.4f}'.format(score[0])+'-'+'{:.4f}'.format(score[1])+'.h5')


def _start_nn():
    print('STARTING FITTING NEURAL NETWORK')
    if os.path.exists(path_best):
        shutil.rmtree(path_best)
    os.mkdir(path_best)
    if not os.path.exists(path_thebest):
        os.mkdir(path_thebest)

    X_train, X_test, Y_train, Y_test = _load_data_nn()

    print('loading data .......')

    def baseline_model():
        """
        Definition of neural network base model
        :return: model
        """
        base_model = Sequential()
        base_model.add(Dense(n_input_layer, activation='elu',  input_shape=(X_train.shape[1],)))
        #hidden
        base_model.add(Dense(31, activation='elu'))
        base_model.add(Dropout(0.1))
        base_model.add(Dense(31, activation='elu'))
        base_model.add(Dropout(0.1))
        base_model.add(Dense(31, activation='elu'))
        base_model.add(Dropout(0.1))
        base_model.add(Dense(31, activation='elu'))
        base_model.add(Dropout(0.3))
        base_model.add(Dense(3, activation='softmax'))
        base_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        return base_model

    # generation kfolds to cross validation process
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    i = 0

    # start cross validation
    for train, test in kfold.split(X_train, Y_train):
        model = baseline_model()
        model.fit(X_train[train], Y_train[train], epochs=num_epochs, batch_size=batch_size, verbose=0,
                  callbacks=[TensorBoard(log_dir='./nn/tensorboard/'),
                             ModelCheckpoint(path_best+'checkpoint-%d.h5' %(i), monitor='acc', verbose=0,
                                             save_best_only=True, mode='max'),
                             EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=2, mode='min')])
        scores = model.evaluate(X_train[test], Y_train[test], verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        i += 1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # evaluate best model based on higher accuracy
    vect_max = np.argmax(cvscores)
    _evaluete_nn(X_test, Y_test, vect_max)
