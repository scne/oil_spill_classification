from src.uns_deep.keras_dec.keras_dec import DeepEmbeddingClustering
# from keras.datasets import mnist
import numpy as np

from src.deep.create_dataset import _generate_dataset, _one_hot_label, _load_image


def get_dataset():
    # np.random.seed(1234)  # set seed for deterministic ordering
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x, y = _generate_dataset("./crop")
    x_all = np.array(_load_image(x))
    # x_all = np.concatenate((x_train, x_test), axis=0)
    # Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])

    Y = y
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]
    return X, Y


X, Y = get_dataset()

c = DeepEmbeddingClustering(n_clusters=3, input_dim=1024)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
c.cluster(X, y=Y)