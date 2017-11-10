from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from scipy.misc import imsave
import os
import shutil
from sklearn.cluster import KMeans, MiniBatchKMeans
from keras.callbacks import TensorBoard, EarlyStopping

from src.deep.create_dataset import _generate_dataset, _load_image, _extract_class
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



def _save_clustering(X, kmeans, counts, path, predict=False):
    print("Count : ", counts)

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
        p = path + "/0/" + str(i) + ".png"
        image = np.asarray(im).reshape(height, width)
        imsave(p, image)
        i = i + 1
    i = 0
    os.mkdir(path + "/1/")
    for im in image1:
        p = path + "/1/" + str(i) + ".png"
        image = np.asarray(im).reshape(height, width)
        imsave(p, image)
        i = i + 1
    i = 0
    os.mkdir(path + "/2/")
    for im in image2:
        p = path + "/2/" + str(i) + ".png"
        image = np.asarray(im).reshape(height, width)
        imsave(p, image)
        i = i + 1


filenames, labels = _generate_dataset('./crop_r')

# labels_0 = _extract_class(labels, name_class=0)

x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=42)

X_train = np.asarray(_load_image(x_train))
X_test = np.asarray(_load_image(x_test))
Y_train = y_train
Y_test = y_test

num_train = len(X_train)
num_test = len(X_test)
num_classes = np.unique(y_train).shape[0]

batch_size = 16
num_epochs = 500
height = 32
width = 32
depth = 1
out_encoder = 3

input_img = Input(shape=(height*width,))
encoded = Dense(9, activation='sigmoid')(input_img)
# encoded = Dense(9, activation='sigmoid')(encoded)
# encoded = Dropout(0.5)(encoded)
encoded = Dense(out_encoder, activation='sigmoid')(encoded)
# decoded = Dense(9, activation='sigmoid')(encoded)
decoded = Dense(9, activation='sigmoid')(encoded)
decoded = Dropout(0.5)(decoded)
decoded = Dense(height*width, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# reshape input

X_train_r = np.asarray(X_train).reshape(-1, height*width)
X_test_r = np.asarray(X_test).reshape(-1, height*width)

print(autoencoder.summary())

autoencoder.fit(X_train_r, X_train_r,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test_r, X_test_r),
                callbacks=[TensorBoard(log_dir='./board2'),
                           EarlyStopping(monitor='loss', min_delta=0.01, patience=10,
                                         verbose=0, mode='auto')]
)

encoded_train_img = encoder.predict(X_train_r)

encoded_test_img = encoder.predict(X_test_r)

kmeans = KMeans(n_clusters=num_classes, init='k-means++', random_state=42).fit(np.asarray(encoded_train_img).reshape(-1, out_encoder))
unique, counts_fit = np.unique(kmeans.labels_, return_counts=True)

predict_test = kmeans.predict(np.asarray(encoded_test_img).reshape(-1, 3))
unique, counts_pre = np.unique(predict_test, return_counts=True)


_save_clustering(X=X_train, kmeans=kmeans, counts=counts_fit, path='./img_train_autoe', predict=False)
_save_clustering(X=X_test, kmeans=predict_test, counts=counts_pre, path='./img_test_autoe', predict=True)


X = np.concatenate((X_train_r, X_test_r), axis=0)
encoded_img = encoder.predict(X)

kmeans_all = KMeans(n_clusters=num_classes, init='k-means++', random_state=42).fit_predict(np.asarray(encoded_img).reshape(-1, 3))
unique_all, counts_all = np.unique(kmeans_all, return_counts=True)
_save_clustering(X=X, kmeans=kmeans_all, counts=counts_all, path='./img_all_autoe', predict=True)

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
print('Ordered Count : ', counts_ord)

confmatrix = confusion_matrix(labels, ordered_v)
print("\n Confusion Matrix :")
print(confmatrix)

class_names = ["0", "1", '2']
# plt.figure()
# _plot_confusion_matrix(confmatrix, classes=class_names, title='Confusion matrix')

print('\nClassification Report : ')
print(classification_report(labels, ordered_v, target_names=class_names))
# plt.show()
print('The end')