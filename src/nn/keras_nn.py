#librerie standard
import shutil
import os
import numpy as np
#librerie keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
#liberie sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report


#PATH PER IL SALVATAGGIO DEI DATI
path_dataset = './nn/dataset/dataset_total.txt' #PERCORSO DOVE RISIEDE IL DATASET
path_best = './nn/best_model/' #PERCORSO DOVE VENGONO SALVATI I MODELLI DELLE RETI ADDESTRATE PER TUTTI I KFOLD
path_thebest = './nn/thebetter_model/' #PERCORSO DOVE VENGONO SALVATE LE RETI CHE HANNO OTTENUTO GLI SCORE MIGLIORI
                                       #DOPO IL TEST
#PARAMETRI PER LA RETE NEURALE
batch_size = 32 #quantità di trainig cases elaborati per esecuzione
num_epochs = 500 #numero massimo di epoche per la quale la rete può essere addestrata
num_classes = 3 #numero di classi che dipendo dal dataset di input
seed = 42 #seme per la generazione random dei pesi
n_splits = 10 #numero di kfold per la cross validation



def _load_data_nn():
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


def _evaluete_nn (X_train, X_test, Y_train, Y_test, vect_max=None):

    #CARICO IL MODELLO CON LA MIGLIORE ACCURANCY E VERIFICO L'ANDAMENTO SUI TEST CASES
    print('Load best model and test it ')
    model = load_model(path_best+'checkpoint-%d.h5' %(vect_max))
    score = model.evaluate(X_test, Y_test, verbose=0) #valuto il modello
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
    #SALVO IL MODELLO APPENA TESTATO CON LOSS E ACCURANCY
    model.save(path_thebest+'model-'+'{:.4f}'.format(score[0])+'-'+'{:.4f}'.format(score[1])+'.h5')


def _start_nn ():
    print('STARTING FITTING NEURAL NETWORK')
    if os.path.exists(path_best):
        shutil.rmtree(path_best)
    os.mkdir(path_best)
    if not os.path.exists(path_thebest):
        os.mkdir(path_thebest)

    X_train, X_test, Y_train, Y_test = _load_data_nn()

    print('loading data .......')

    #DEFINIZIONE DEL MODELLO PER LA NEURAL NETWORK
    def baseline_model ():
        model = Sequential()
        model.add(Dense(5, activation='elu',  input_shape=(X_train.shape[1],)))
        #hidden
        model.add(Dense(31, activation='elu'))
        model.add(Dense(31, activation='elu'))
        model.add(Dense(31, activation='elu'))
        model.add(Dense(31, activation='elu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    #GENERAZIONE DEGLI SPLIT PER L'ADDESTRAMENTO TRAMITE CROSS VALIDATION
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cvscores = []
    i = 0

    #STARTING CROSS VALIDATION
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

    #PASSO ALLA FUNZIONE DI VALUTAZIONE IL MODELLO CHE HA OTTENUTO L'ACCURANCY MIGLIORE TRA QUELLI GENERATI
    vect_max = np.argmax(cvscores)
    _evaluete_nn(X_train, X_test, Y_train, Y_test, vect_max)
