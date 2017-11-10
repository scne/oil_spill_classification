from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from scipy.misc import imsave
import os
import shutil
from sklearn.cluster import KMeans, MiniBatchKMeans


from src.deep.create_dataset import _generate_dataset, _load_image, _extract_class
from sklearn.model_selection import train_test_split

filenames, labels = _generate_dataset('./crop_c')

labels_0 = _extract_class(labels, name_class=0)

x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=42, stratify=labels)

X_train = np.asarray(_load_image(x_train))
X_test = np.asarray(_load_image(x_test))
Y_train = y_train
Y_test = y_test


num_train = len(X_train)
num_test = len(X_test)
num_classes = np.unique(y_train).shape[0]

batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 10 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

height = 32
width = 32
depth = 1

input_img = Input(shape=(height, width, depth)) # adapt this if using `channels_first` image data format

x = Conv2D(6, (kernel_size, kernel_size), activation='relu', padding='same')(input_img)
x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
x = Conv2D(16, (kernel_size, kernel_size), activation='relu', padding='same')(x)
x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
x = Conv2D(8, (kernel_size, kernel_size), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((pool_size, pool_size), padding='same')(x)
x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
x = Conv2D(4, (kernel_size, kernel_size), activation='relu', padding='same')(x)
# x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
# x = Conv2D(2, (kernel_size, kernel_size), activation='relu', padding='same')(x)
encoded = MaxPooling2D((pool_size, pool_size), padding='same')(x)


# at this point the representation is (4, 4, 8) i.e. 128-dimensional
# x = Conv2D(2, (kernel_size, kernel_size), activation='relu', padding='same')(encoded)
# x = UpSampling2D((pool_size, pool_size))(x)
x = Conv2D(4, (kernel_size, kernel_size), activation='relu', padding='same')(encoded)
x = UpSampling2D((pool_size, pool_size))(x)
x = Conv2D(8, (kernel_size, kernel_size), activation='relu', padding='same')(x)
x = UpSampling2D((pool_size, pool_size))(x)
x = Conv2D(16, (kernel_size, kernel_size), activation='relu', padding='same')(x)
x = UpSampling2D((pool_size, pool_size))(x)
x = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same')(x)
x = UpSampling2D((pool_size, pool_size))(x)
decoded = Conv2D(1, (kernel_size, kernel_size), activation='sigmoid', padding='same')(x)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print(autoencoder.summary())

autoencoder.fit(X_train, X_train,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[TensorBoard(log_dir='./board/autoencoder')]
                )

encoded_imgs = encoder.predict(X_test)
# decoded_imgs = autoencoder.predict(X_test)

# n = 9
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(X_test[i].reshape(32, 32))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n+1)
#     plt.imshow(decoded_imgs[i].reshape(32, 32))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(n):
#     ax = plt.subplot(1, n, i+1)
#     plt.imshow(encoded_imgs[i].reshape(1, 1 * 2).T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(np.asarray(encoded_imgs).reshape(-1, 16))
unique, counts = np.unique(kmeans.labels_, return_counts=True)


print("Count : ", counts)

idx0 = np.where(kmeans.labels_ == 0)
idx1 = np.where(kmeans.labels_ == 1)
idx2 = np.where(kmeans.labels_ == 2)

image0 = X_test[idx0]
image1 = X_test[idx1]
image2 = X_test[idx2]

p = "./auto_k_img"
if os.path.exists(p):
    shutil.rmtree(p)
i = 0
os.mkdir(p)
os.mkdir(p+"/0/")
for im in image0:
    patch = p+"/0/"+str(i)+".png"
    image = np.asarray(im).reshape(height, width)
    imsave(patch, image)
    i = i + 1
i = 0
os.mkdir(p+"/1")
for im in image1:
    patch = p+"/1/"+str(i)+".png"
    image = np.asarray(im).reshape(height, width)
    imsave(patch, image)
    i = i + 1
i = 0
os.mkdir(p+"/2")
for im in image2:
    patch = p+"/2/"+str(i)+".png"
    image = np.asarray(im).reshape(height, width)
    imsave(patch, image)
    i = i + 1