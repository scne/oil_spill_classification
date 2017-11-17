import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from src.deep.create_dataset import _generate_dataset, _load_image

tf.logging.set_verbosity(tf.logging.INFO)

# path to save files
path_dataset = './img_cluster'  # dataset path
path_model = './model/'  # path to save model

# convolutional network parmas
batch_size = 64  # training cases batch
num_epochs = 500  # max number of epochs
kernel_size = 3  # kernel size dimension
pool_size = 2  # max pooling size
seed = 42  # base random seed

# dataset image parameters
height = 32
width = 32
depth = 1

filenames, labels = _generate_dataset(path_dataset)
x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=seed, stratify=labels)
X_train = np.asarray(_load_image(x_train))
X_test = np.asarray(_load_image(x_test))
num_classes = np.unique(y_train).shape[0]
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

def model_fn(features, labels, mode):
    input_layer = features["x"]

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.elu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    drop1 = tf.layers.dropout(inputs=pool1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    conv2 = tf.layers.conv2d(
        inputs=drop1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.elu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    drop2 = tf.layers.dropout(inputs=pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    drop2_flat = tf.reshape(drop2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=drop2_flat, units=1024, activation=tf.nn.elu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=logits, name="acc")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels, logits=logits)
    logging_hook_inner = tf.train.LoggingTensorHook([loss], every_n_iter=10)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=logging_hook_inner)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(input=logits, axis=1))}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir="./model_tensor/model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=np.asarray(Y_train).astype(np.float32),
    batch_size=64,
    num_epochs=None,
    shuffle=True)
classifier.train(
    input_fn=train_input_fn,
    steps=3000,
    hooks=[logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=Y_test,
    num_epochs=1,
    shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)