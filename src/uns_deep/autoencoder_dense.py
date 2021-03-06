# standard libraries
import os
import shutil

import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
# keras libraries
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from scipy.misc import imsave
# sklearn libraries
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import mixture
# utility functions
from src.deep.create_dataset import _generate_dataset, _load_image

# path to save files
path_dataset = './uns_deep/crop_r'  # dataset base path
path_img_out = './uns_deep/img_cluster'  # path to save clustered images
path_model = './uns_deep/model/'  # path to save autoencoder model
path_board = './uns_deep/tensorboard/'

# unsupervised neural network params
batch_size = 32  # training cases batch
num_epochs = 500  # max number of epochs
out_encoder = 4  # number of neuron of encoder layer
seed = 42   # base random seed

# dataset image parameters
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
    """
    load dataset from path
    :return: train dataset and file names
    """
    filenames, labels = _generate_dataset(path_dataset)
    X_train = np.asarray(_load_image(filenames))
    Y_train = labels

    return X_train, Y_train, filenames


def start_uns():
    print('STARTING FITTING UNSUPERVISED NEURAL NETWORK')
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    if os.path.exists(path_board):
        shutil.rmtree(path_board)
        os.mkdir(path_board)
    X_train, Y_train, filenames= _load_data_uns()
    print('loading data .........')
    num_classes = np.unique(Y_train).shape[0]

    # unsupervised neural network base model
    # input_img = Input(shape=(height*width,))
    # # encoded = Dense(9, activation='sigmoid')(input_img)
    # # encoded = Dense(9, activation='sigmoid')(encoded)
    # # encoded = Dropout(0.5)(encoded)
    # encoded = Dense(out_encoder, activation='sigmoid')(input_img)
    # # decoded = Dense(9, activation='sigmoid')(encoded)
    # # decoded = Dense(9, activation='sigmoid')(encoded)
    # # decoded = Dropout(0.5)(decoded)
    # decoded = Dense(height*width, activation='sigmoid')(encoded)

    autoencoder = Sequential()
    # autoencoder.add(Dense(1024, activation='sigmoid', input_shape=(height*width,)))
    # autoencoder.add(Dropout(0.1))
    autoencoder.add(Dense(8, activation='sigmoid', input_shape=(height*width,)))
    autoencoder.add(Dense(out_encoder))
    out_e = Activation('sigmoid')  # output encoder
    autoencoder.add(out_e)
    autoencoder.add(Dense(8, activation='sigmoid'))
    autoencoder.add(Dropout(0.5, seed=seed))
    autoencoder.add(Dense(height*width, activation='sigmoid'))

    # model generation
    # autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # encoder = Model(input_img, encoded)
    encoder = Model(autoencoder.input, out_e.output)
    # print(autoencoder.summary())

    # reshape input
    X_train_r = np.asarray(X_train).reshape(-1, height*width)
    # print(X_train.shape[0])
    # batch_size=1
    # start fitting process
    autoencoder.fit(X_train_r, X_train_r,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=2,
                    callbacks=[TensorBoard(log_dir='./uns_deep/tensorboard', histogram_freq=1, write_images=True,
                                           write_grads=True),
                               EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,
                                             verbose=0, mode='auto')]
    )

    autoencoder.save(path_model+'autoencoder.h5')

    encoded_img = encoder.predict(X_train_r)

    kmeans_all = KMeans(n_clusters=num_classes, init='k-means++', random_state=seed)\
        .fit_predict(np.asarray(encoded_img).reshape(-1, out_encoder))
    unique_all, counts_all = np.unique(kmeans_all, return_counts=True)

    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(np.asarray(encoded_img).reshape(-1, out_encoder))
    class_gmm = gmm.predict(np.asarray(encoded_img).reshape(-1, out_encoder))
    unique_gmm, counts_gmm = np.unique(class_gmm, return_counts=True)
    print('\nGMM Count : ', counts_gmm)

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

    _save_clustering(X=X_train_r, kmeans=class_gmm, path=path_img_out+'_gmm', filenames=filenames,
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
