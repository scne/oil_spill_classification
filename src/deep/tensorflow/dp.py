import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from math import sqrt


class Network:
    def __init__(self, train_batch_size, test_batch_size, pooling_scale,
                 dropout_rate, base_learning_rate, decay_rate,
                 optimizeMethod='adadelta', save_path='./model/default.ckpt'):
        self.optimizeMethod = optimizeMethod
        self.dropout_rate = dropout_rate
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Hyper Parameters
        self.conv_config = []
        self.fc_config = []
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.pooling_scale = pooling_scale
        self.pooling_stride = pooling_scale

        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None

        self.writer = None
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        self.saver = None
        self.save_path = save_path

    def add_conv(self, patch_size, in_depth, out_depth, activation='relu', pooling=False, name='conv'):
        self.conv_config.append({
            'patch_size': patch_size,
            'in_depth': in_depth,
            'out_depth': out_depth,
            'activation': activation,
            'pooling': pooling,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=0.1), name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_depth]), name=name + '_biases')
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, in_num_nodes, out_num_nodes, activation='relu', name='fc'):
        self.fc_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes]))
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)
            self.train_summaries.append(tf.summary.histogram(str(len(self.fc_weights)) + '_weights', weights))
            self.train_summaries.append(tf.summary.histogram(str(len(self.fc_biases)) + '_biases', biases))

    def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        # 1e5
        return _lambda * regularization

    # should make the definition as an exposed API, instead of implemented in the function
    def define_inputs(self, train_samples_shape, train_labels_shape, test_samples_shape):
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')

    def define_model(self):
        def model(data_flow, train=True):
            # Define Convolutional Layers
            for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
                with tf.name_scope(config['name'] + '_model'):
                    with tf.name_scope('convolution'):
                        # default 1,1,1,1 stride and SAME padding
                        data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                        data_flow = data_flow + biases
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i + 1), name=config['name'] + '_conv')
                        else:
                            if i == 0:
                                # save kernel image
                                grid = self.put_kernels_on_grid(weights)
                                self.train_summaries.append(tf.summary.image('_conv', grid, max_outputs=1))

                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i + 1), name=config['name'] + '_relu')
                    elif config['activation'] == 'elu':
                        data_flow = tf.nn.elu(data_flow)
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i + 1), name=config['name'] + '_elu')
                    else:
                        raise Exception('Activation Func can only be Relu or Elu right now. You passed', config['activation'])
                    if config['pooling']:
                        data_flow = tf.nn.max_pool(
                            data_flow,
                            ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                            strides=[1, self.pooling_stride, self.pooling_stride, 1],
                            padding='SAME')
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i + 1) // 2,
                                                      name=config['name'] + '_pooling')
                    data_flow = tf.nn.dropout(data_flow, 0.75, seed=4926)

            # Define Fully Connected Layers
            for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
                if i == 0:
                    shape = data_flow.get_shape().as_list()
                    data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
                with tf.name_scope(config['name'] + 'model'):

                    ### Dropout
                    if train and i == len(self.fc_weights) - 1:
                        data_flow = tf.nn.dropout(data_flow, 0.6, seed=4926)
                    ###

                    data_flow = tf.matmul(data_flow, weights) + biases
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                    elif config['activation'] == 'elu':
                        data_flow = tf.nn.elu(data_flow)
                    elif config['activation'] == 'softmax':
                        data_flow = tf.nn.softmax(data_flow)
                    elif config['activation'] is None:
                        pass
                    else:
                        raise Exception('Activation Func can only be Relu or None right now. You passed',
                                        config['activation'])
            return data_flow

        # Training computation.
        logits = model(self.tf_train_samples)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=self.tf_train_labels))
            self.loss += self.apply_regularization(_lambda=5e-4)
            self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

        # learning rate decay
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.base_learning_rate,
            global_step=global_step * self.train_batch_size,
            decay_steps=100,
            decay_rate=self.decay_rate,
            staircase=True
        )

        # Optimizer.
        with tf.name_scope('optimizer'):
            if (self.optimizeMethod == 'gradient'):
                self.optimizer = tf.train \
                    .GradientDescentOptimizer(learning_rate) \
                    .minimize(self.loss)
            elif (self.optimizeMethod == 'momentum'):
                self.optimizer = tf.train \
                    .MomentumOptimizer(learning_rate, 0.5) \
                    .minimize(self.loss)
            elif (self.optimizeMethod == 'adam'):
                self.optimizer = tf.train \
                    .AdamOptimizer(learning_rate) \
                    .minimize(self.loss)
            elif (self.optimizeMethod == 'adadelta'):
                self.optimizer = tf.train \
                    .AdadeltaOptimizer(learning_rate=learning_rate) \
                    .minimize(self.loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope('train'):
            self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
            tf.add_to_collection("prediction", self.train_prediction)
        with tf.name_scope('test'):
            self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')
            tf.add_to_collection("prediction", self.test_prediction)

        self.merged_train_summary = tf.summary.merge(self.train_summaries)
        self.merged_test_summary = tf.summary.merge(self.test_summaries)

        self.saver = tf.train.Saver(tf.global_variables())

    def run(self, train_samples, train_labels, test_samples, test_labels, train_data_iterator, iteration_steps,
            test_data_iterator):

        with tf.Session(graph=tf.get_default_graph()) as session:
            self.writer = tf.summary.FileWriter('./board', session.graph)
            tf.global_variables_initializer().run()

            print('Start Training')
            for i, samples, labels in train_data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                          chunkSize=self.train_batch_size):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels},
                    options=run_options,
                    run_metadata=run_metadata
                )
                self.writer.add_run_metadata(run_metadata, 'step%d' % i)
                self.writer.add_summary(summary, i)

                # labels is True Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)

            import os
            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

            accuracies = []
            confusionMatrices = []
            for i, samples, labels in test_data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples},
                    options=run_options,
                    run_metadata=run_metadata
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices), len(test_labels))
        ###

    def train(self, train_samples, train_labels, data_iterator, iteration_steps):
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.writer = tf.summary.FileWriter('./board', session.graph)
            tf.global_variables_initializer().run()

            print('Start Training')
            # batch 1000
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                    chunkSize=self.train_batch_size):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels},
                    options=run_options,
                    run_metadata=run_metadata
                )
                self.writer.add_run_metadata(run_metadata, 'step%d' % i)
                self.writer.add_summary(summary, i)

                # labels is True Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)

            import os
            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

    def test(self, test_samples, test_labels, data_iterator):
        if self.saver is None:
            self.define_model()
        if self.writer is None:
            self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())

        print('Before session')
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.saver.restore(session, self.save_path)

            accuracies = []
            confusionMatrices = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples},
                    options=run_options,
                    run_metadata=run_metadata
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices, len(test_labels)))
        ###

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm

    def visualize_filter_map(self, tensor, how_many, display_size, name):
        # print(tensor.get_shape)
        filter_map = tensor[-1]
        # print(filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        # print(filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        # print(how_many)
        self.test_summaries.append(tf.summary.image(name, tensor=filter_map, max_outputs=how_many))

    def print_confusion_matrix(self, confusionMatrix, population):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            # recall index
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / population)
            print(column[i] / np.sum(column), )
        print('\n', np.sum(confusionMatrix), a)

    def put_kernels_on_grid(self, kernel, pad=1):
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
        print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + 2 * pad
        X = kernel.get_shape()[1] + 2 * pad

        channels = kernel.get_shape()[2]

        # put NumKernels to the 1st dimension
        x = tf.transpose(x, (3, 0, 1, 2))
        # organize grid on Y axis
        x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x = tf.transpose(x, (0, 2, 1, 3))
        # organize grid on X axis
        x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x = tf.transpose(x, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x = tf.transpose(x, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x
