#librerie standard
import numpy as np
import matplotlib.pyplot as plt
#librerie keras
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model
from keras.utils import np_utils
#librerie sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#funzioni
from src.deep.create_dataset import _generate_dataset, _load_image, _plot_confusion_matrix

#PATH PER IL SALVATAGGIO DEI DATI
path_dataset = './deep/img_cluster' #PERCORSO DOVE SONO SALVATE LE IMMAGINI DI INPUT PER LA RETE
path_model = './deep/model/model.h5'  #PERCORSO DOVE VIENE SALVATO IL MODELLO DOPO L'ADDESTRAMENTO DELLA RETE

#PARAMETRI PER LA CONVOLUTIONAL DEEP NEURAL NETWORK
batch_size = 64 #quantità di trainig cases elaborati per esecuzione
num_epochs = 500 #numero massimo di epoche per la quale la rete può essere addestrata
kernel_size = 3 #dimensione del kernel per l'operazione di convolusione della rete
pool_size = 2 #dimensione dello strato di pooling per la rete
seed = 42 #seme per la generazione random dei pesi

#PARAMETRI DIMENSIONE IMMAGINI
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

def _evaluete_cnn(X_train, X_test, Y_train, Y_test):

    model = load_model(path_model)

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
    model.save(path_model, overwrite=True)
    #PASSO ALLA FASE DI VALUTAZIONE DEL MODELLO SUI TEST CASES
    _evaluete_cnn(X_train, X_test, Y_train, Y_test)
    print("The End")
