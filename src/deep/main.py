

if __name__ == '__main__':
    from src.deep.create_dataset import _generate_dataset, _one_hot_label, _load_image
    from src.deep.dp import Network
    from sklearn.model_selection import train_test_split
    import numpy as np

    filenames, labels = _generate_dataset('./crop')

    x_train, x_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.30, random_state=42, stratify=labels)

    filenames_size = len(filenames)
    num_labels = 3
    num_channels = 1
    image_size = 32

    X_test = _load_image(x_test)
    X_train = _load_image(x_train)


    def train_data_iterator(samples, labels, iteration_steps, chunkSize):
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while i < iteration_steps:
            stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
            yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
            i += 1


    def test_data_iterator(samples, labels, chunkSize):
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while stepStart < len(samples):
            stepEnd = stepStart + chunkSize
            if stepEnd < len(samples):
                yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
                i += 1
            stepStart = stepEnd


    net = Network(
        train_batch_size=40, test_batch_size=500, pooling_scale=2,
        dropout_rate=0.9,
        base_learning_rate=0.001, decay_rate=0.99)
    net.define_inputs(
        train_samples_shape=(40, image_size, image_size, num_channels),
        train_labels_shape=(40, num_labels),
        test_samples_shape=(500, image_size, image_size, num_channels),
    )
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32, activation='relu', pooling=False, name='conv1')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv2')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')

    net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=128, activation='relu',
               name='fc1')
    net.add_fc(in_num_nodes=128, out_num_nodes=3, activation=None, name='fc2')

    net.define_model()
    net.run(X_train, y_train, X_test, y_test, train_data_iterator=train_data_iterator,
             iteration_steps=3000, test_data_iterator=test_data_iterator)


else:
    raise Exception('main.py: Should Not Be Imported!!! Must Run by "python main.py"')
