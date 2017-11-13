import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential  # basic class for specifying and training a neural network
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from src.deep.create_dataset import _generate_dataset, _load_image, _plot_confusion_matrix

batch_size = 100  # in each iteration, we consider 32 training examples at once
num_epochs = 500  # we iterate 200 times over the entire training set
kernel_size = 3  # we will use 3x3 kernels throughout
pool_size = 2  # we will use 2x2 pooling throughout
conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64  # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
hidden_size = 512  # the FC layer will have 512 neurons


filenames, labels = _generate_dataset('./img_autoe')

# labels_0 = _extract_class(labels, name_class=0)


x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=42, stratify=labels)

X_train = np.asarray(_load_image(x_train))
X_test = np.asarray(_load_image(x_test))
Y_train = y_train
Y_test = y_test


# (X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

height = 32
width = 32
depth = 1
num_train = len(X_train)
# num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = len(X_test)
# num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes


# Y_train = _one_hot_label(y_train)
Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels


model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1,
          callbacks=[TensorBoard(log_dir='./board2'), EarlyStopping(monitor='loss', min_delta=0.1, patience=10,
                                                                    verbose=0, mode='min')])

model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

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


print("\n Metrics => ")
print(model.metrics_names)
print(score)
print('\nClassification Report : ')
print(classification_report(y_test, Y_predict, target_names=class_names))
# plt.show()
print("The End")