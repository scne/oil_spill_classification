import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import itertools

print('= LOADING Training Cases ==================')

dataset = np.loadtxt('./dataset/dataset_total.txt', delimiter='\t')

dim_dataset = dataset.shape

print(dim_dataset)

y = np.array(np.ceil(dataset[:, -1])).astype(np.str)
X = np.array(dataset[:, :-1]).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(x_train.shape)
print(y_train.shape)

scaler = preprocessing.StandardScaler().fit(x_train)

X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

y_train = y_train.reshape((len(y_train), 1))
y_test = y_test.reshape((len(y_test), 1))

# Parameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50

feature_columns = [tf.feature_column.numeric_column("x", shape=[31])]

#get unique labels
uniqueTrain = set()
for l in y:
    uniqueTrain.add(l)
uniqueTrain = list(uniqueTrain)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[31, 31, 31, 31],
                                        n_classes=3,
                                        model_dir="./model/oil_nn_model",
                                        label_vocabulary=uniqueTrain)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=None,
    shuffle=True)

# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)


# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_test)},
    y=y_test,
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

predictions_iter = classifier.predict(input_fn=test_input_fn)

predictions = [*map(lambda x: x['classes'], list(itertools.islice(predictions_iter, len(y_test))))]

confusion_matrix = confusion_matrix(y_test, np.asarray(predictions).astype(np.str))

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
print("\nConfusion matrix: \n", confusion_matrix)

