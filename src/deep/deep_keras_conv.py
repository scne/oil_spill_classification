from keras.models import Model, Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping

from src.deep.create_dataset import _generate_dataset, _load_image, _extract_class, _plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing


batch_size = 64# in each iteration, we consider 32 training examples at once
num_epochs = 500 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons


filenames, labels = _generate_dataset('./img_autoe')


x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=42, stratify=labels)

X_train = np.asarray(_load_image(x_train))
X_test = np.asarray(_load_image(x_test))
Y_train = y_train
Y_test = y_test


height = 32
width = 32
depth = 1
num_train = len(X_train)
# num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = len(X_test)
# num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes


print("ciaoooo")
# Y_train = _one_hot_label(y_train)
Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

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

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1,
          callbacks=[TensorBoard(log_dir='./board2'), EarlyStopping(monitor='loss', min_delta=0.1, patience=10,
                                                                    verbose=0, mode='auto')])

model.evaluate(X_test, Y_test, verbose=1, sample_weight=None)  # Evaluate the trained model on the test set!

print("Saving model")
model.save("./model/model.h5", overwrite= True)

score = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

y_predict = np.asarray(model.predict(X_test, verbose=0))
Y_predict = np.argmax(y_predict, axis=1)
confmatrix = confusion_matrix(y_test, Y_predict)
print("\n Confusion Matrix :")
# print(confmatrix)

class_names = ["0", "1", '2']
plt.figure()
_plot_confusion_matrix(confmatrix, classes=class_names, title='Confusion matrix')


print("\nMetrics => ")
print(model.metrics_names)
print(score)
print('\nClassification Report : ')
print(classification_report(y_test, Y_predict, target_names=class_names))
# plt.show()
print("The End")