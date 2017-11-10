from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from scipy.misc import imsave
import os
import shutil

from src.deep.create_dataset import _generate_dataset, _load_image
from sklearn.model_selection import train_test_split


height = 32
width = 32

filenames, labels = _generate_dataset('./crop_r')

x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=42, stratify=labels)

X_train = np.asarray(_load_image(x_train))
X_test = np.asarray(_load_image(x_test))

num_train = len(X_train)
num_test = len(X_test)
num_classes = np.unique(y_train).shape[0]

X_train_r = np.asarray(X_train).reshape(num_train, height*width)

kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(X_train_r)

unique, counts = np.unique(kmeans.labels_, return_counts=True)

print("Count : ", counts)

idx0 = np.where(kmeans.labels_ == 0)
idx1 = np.where(kmeans.labels_ == 1)
idx2 = np.where(kmeans.labels_ == 2)

image0 = X_train[idx0]
image1 = X_train[idx1]
image2 = X_train[idx2]

p = "./img_k"
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
for im in image1:
    patch = p+"/2/"+str(i)+".png"
    image = np.asarray(im).reshape(height, width)
    imsave(patch, image)
    i = i + 1
