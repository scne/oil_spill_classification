#librerie standard
import numpy as np
from scipy.misc import imsave
import os
import shutil
import pandas as pd
#librerie keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
#librerie sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix, classification_report
#funzioni
from src.deep.create_dataset import _generate_dataset, _load_image, _extract_class

#PATH PER IL SALVATAGGIO DEI DATI
path_dataset = './uns_deep/crop_r' #PERCORSO DOVE SONO SALVATE LE IMMAGINI DI INPUT PER LA RETE
path_img_out = './uns_deep/img_cluster' #PERSOCOSO DOVE VENGONO SALVATE LE IMMAGINI CLUSTERIZZATE
path_model = './uns_deep/model/' #PERCORSO DOVE VERRA' SALVATO IL MODELLO DELL'AUTOENCODER

#PARAMETRI PER L'UNSUPERVISED NEURAL NETWORK
batch_size = 16 #quantità di trainig cases elaborati per esecuzione
num_epochs = 500 #numero massimo di epoche per la quale la rete può essere addestrata
out_encoder = 3 #numero di neuroni in uscita dall'encoder
seed = 42 #seme per la generazione random dei pesi

#PARAMETRI DIMENSIONE IMMAGINI
height = 32
width = 32
depth = 1


def _save_clustering(X, kmeans, path, filenames, predict=False):

    names = []
    for i in range(len(filenames)):
        names.append(filenames[i][20:-4])
    if predict:
        idx0 = np.where(kmeans == 0)
        idx1 = np.where(kmeans == 1)
        idx2 = np.where(kmeans == 2)
    else:
        idx0 = np.where(kmeans.labels_ == 0)
        idx1 = np.where(kmeans.labels_ == 1)
        idx2 = np.where(kmeans.labels_ == 2)

    image0 = X[idx0]
    image1 = X[idx1]
    image2 = X[idx2]

    if os.path.exists(path):
        shutil.rmtree(path)
    i = 0
    os.mkdir(path)
    os.mkdir(path + "/0/")
    for im in image0:
        p = path + "/0/" + names[idx0[0][i]] + ".png"
        image = np.asarray(im).reshape(height, width)
        imsave(p, image)
        i = i + 1
    i = 0
    os.mkdir(path + "/1/")
    for im in image1:
        p = path + "/1/" + names[idx1[0][i]] + ".png"
        image = np.asarray(im).reshape(height, width)
        imsave(p, image)
        i = i + 1
    i = 0
    os.mkdir(path + "/2/")
    for im in image2:
        p = path + "/2/" + names[idx2[0][i]] + ".png"
        image = np.asarray(im).reshape(height, width)
        imsave(p, image)
        i = i + 1


def _load_data_uns():

    filenames, labels = _generate_dataset(path_dataset)
    X_train = np.asarray(_load_image(filenames))
    Y_train = labels

    return X_train, Y_train, filenames

def _start_uns():
    print('STARTING FITTING UNSUPERVISED NEURAL NETWORK')
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    X_train, Y_train, filenames= _load_data_uns()
    print('loading data .........')
    num_classes = np.unique(Y_train).shape[0]

    #DEFINIZIONE DEL MODELLO PER L'UNSUPERVIDE NEURAL NETWORK
    input_img = Input(shape=(height*width,))
    encoded = Dense(9, activation='sigmoid')(input_img)
    # encoded = Dense(9, activation='sigmoid')(encoded)
    # encoded = Dropout(0.5)(encoded)
    encoded = Dense(out_encoder, activation='sigmoid')(encoded)
    # decoded = Dense(9, activation='sigmoid')(encoded)
    decoded = Dense(9, activation='sigmoid')(encoded)
    decoded = Dropout(0.5)(decoded)
    decoded = Dense(height*width, activation='sigmoid')(encoded)

    #CREAZIONE DEL MODELLO E COMPILAZIONE
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder = Model(input_img, encoded)
    # print(autoencoder.summary())

    # reshape input
    X_train_r = np.asarray(X_train).reshape(-1, height*width)

    #STARTING FITTING
    autoencoder.fit(X_train_r, X_train_r,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[TensorBoard(log_dir='./uns_deep/tensorboard'),
                               EarlyStopping(monitor='loss', min_delta=0.01, patience=10,
                                             verbose=0, mode='auto')]
    )

    autoencoder.save(path_model+'autoencoder.h5')

    encoded_img = encoder.predict(X_train_r)

    kmeans_all = KMeans(n_clusters=num_classes, init='k-means++', random_state=seed).fit_predict(np.asarray(encoded_img).reshape(-1, 3))
    unique_all, counts_all = np.unique(kmeans_all, return_counts=True)

    order_class = np.sort(counts_all)[::-1]
    ordered_v = np.empty(shape=kmeans_all.shape)
    if (counts_all != order_class).all:
        idx_0 = np.asscalar(np.asarray(np.where(counts_all == order_class[0])))
        idx_1 = np.asscalar(np.asarray(np.where(counts_all == order_class[1])))
        idx_2 = np.asscalar(np.asarray(np.where(counts_all == order_class[2])))
        ordered_v[np.asarray(np.where(kmeans_all == idx_0))] = 0
        ordered_v[np.asarray(np.where(kmeans_all == idx_1))] = 1
        ordered_v[np.asarray(np.where(kmeans_all == idx_2))] = 2

    unique_ord, counts_ord = np.unique(ordered_v, return_counts=True)
    _save_clustering(X=X_train_r, kmeans=ordered_v, path=path_img_out, filenames=filenames,
                     predict=True)
    print('\nOrdered Count : ', counts_ord)

    confmatrix = confusion_matrix(Y_train, ordered_v)
    print("\nConfusion Matrix :")
    print(confmatrix)

    class_names = ["0", "1", '2']
    # plt.figure()
    # _plot_confusion_matrix(confmatrix, classes=class_names, title='Confusion matrix')

    print('\nClassification Report : ')
    print(classification_report(Y_train, ordered_v, target_names=class_names))

    # out = np.matrix([Y_train, ordered_v])
    # out = {'Original': pd.Series(Y_train).astype(int),
    #        'New Tag': pd.Series(ordered_v).astype(int)}
    # df_out =pd.DataFrame(out)
    # # print(out.shape)
    # pd.set_option('display.max_rows', len(df_out))
    # print(df_out)

    # plt.show()
    print('The end')