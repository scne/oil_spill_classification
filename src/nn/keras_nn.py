from keras.models import Model, Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, LSTM, Embedding
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from src.deep.create_dataset import _generate_dataset, _load_image, _extract_class, _plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


batch_size = 32
num_epochs = 500
num_classes = 3
seed = 42

print('= LOADING Training Cases ==================')

dataset = np.loadtxt('./dataset/dataset_total.txt', delimiter='\t')
path_best = './best_model/'
path_thebest = './thebetter_model/'
if os.path.exists(path_best):
    shutil.rmtree(path_best)
os.mkdir(path_best)
# if os.path.exists(path_thebest):
#     shutil.rmtree(path_thebest)
# os.mkdir(path_thebest)

dim_dataset = dataset.shape

# print(dim_dataset)

y = np.array(np.ceil(dataset[:, -1])).astype(np.str)
X = np.array(dataset[:, :-1]).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# print(x_train.shape)
# print(y_train.shape)

scaler = preprocessing.StandardScaler().fit(x_train)

X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

y_train = np.subtract(y_train.reshape((len(y_train), 1)).astype(np.float32), np.asarray(2.0))
y_test = np.subtract(y_test.reshape((len(y_test), 1)).astype(np.float32),  np.asarray(2.0))

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
print('loaded.................')
print('= STARTING TRAINING =======================')
timesteps = 8
def baseline_model ():
    model = Sequential()
    model.add(Dense(5, activation='relu',  input_shape=(X_train.shape[1],)))
    #hidden
    model.add(Dense(31, activation='relu'))
    model.add(Dense(31, activation='relu'))
    model.add(Dense(31, activation='relu'))
    model.add(Dense(31, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return model

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

vect_max = []
cvscores = []
i = 0
for train, test in kfold.split(X_train, Y_train):
    model = baseline_model()
    history = model.fit(X_train[train], Y_train[train], epochs=num_epochs, batch_size=batch_size, verbose=0,
              callbacks=[ModelCheckpoint('./best_model/checkpoint-%d.h5' %(i), monitor='acc', verbose=1,
                                         save_best_only=True, mode='max'),
                         EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=2, mode='min')])
    vect_max.append(np.max(history.history.get('acc')))
    i = i + 1
    scores = model.evaluate(X_train[test], Y_train[test], verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print('Load best model and test it ')
max_acc = 0

model = load_model('./best_model/checkpoint-%d.h5' %(np.argmax(vect_max)))
score = model.evaluate(X_train, Y_train, verbose=0)
print('Metrics => ', model.metrics_names, score)

# for root, dirs, files in os.walk('./best_model/'):
#     print('Starting retrainig for %d models' %(len(files)))
#     for name in files:
#         filenames = os.path.join(root, name)
#         print('Test for :', filenames)
#         model = load_model(filenames, custom_objects=None)
#         model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=0, validation_split=0.1,
#                   callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=2, mode='auto')])
#         score = model.evaluate(X_train, Y_train, verbose=0)
#         print('Metrics => ', model.metrics_names, score)
#         if score[1] > max_acc:
#             model_max = filenames
#             model.save('./thebetter_model/thebetter.h5')
#             max_acc = score[1]

# print('\nThe better model is ', model_max)
# model = load_model('./thebetter_model/thebetter.h5')
y_predict = np.asarray(model.predict(X_test, verbose=0))
Y_predict = np.argmax(y_predict, axis=1)
confmatrix = confusion_matrix(y_test, Y_predict)
print("\nConfusion Matrix :")
# print(confmatrix)
class_names = ["0", "1", '2']
plt.figure()
_plot_confusion_matrix(confmatrix, classes=class_names, title='Confusion matrix')
print("\nMetrics => ", model.metrics_names, score)
print('\nClassification Report : ')
print(classification_report(y_test, Y_predict, target_names=class_names))
model.save(path_thebest+'model-'+'{:.4f}'.format(score[0])+'-'+'{:.4f}'.format(score[1])+'.h5')
# plt.show()
print("The End")
