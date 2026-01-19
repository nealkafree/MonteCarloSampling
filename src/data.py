import random

from scipy.io import loadmat
import numpy as np


def prepare_data(dataset_path, random_state=42):
    """
    Loads data, prepares it for training and evaluation and balances classes. All in one.
    :param dataset_path: path to .mat file with data.
    :param random_state:
    :return: list of (image, label)
    """
    dataset = loadmat(dataset_path)

    # Function that transforms image to greyscale
    grey = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # Dimensions in this dataset are weird
    data = np.transpose(dataset['X'], [3, 0, 1, 2])
    # Greyscale data allows us to save on computation (and kind of how it is always done)
    data = grey(data)

    # Normalizing data and making it eatable for pytorch
    data = np.expand_dims(np.float32(data), 0) / 255
    #  Changing labels '10' for '0'
    y = dataset['y']
    y[y == 10] = 0

    # Classes in the dataset are not balanced
    # I use the size of the smallest class as size for all classes
    unique = np.unique(y, return_counts=True)
    class_size = np.min(unique[1])

    # And that's how I balance classes
    x_train = []
    y_train = []
    label_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for image, label in zip(data, y):
        if label_count[label] < class_size:
            label_count[label] += 1
            x_train.append(image)
            y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # I need to shuffle here, because currently dataset is not homogenic in different parts
    rng = np.random.default_rng(random_state)
    p = rng.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]

    return x_train, y_train
