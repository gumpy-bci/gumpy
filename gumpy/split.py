import sklearn.model_selection
import numpy as np


def normal(X, labels, test_size):
    """Split a dataset into training and test parts.

    """
    Y = labels
    X_train, X_test, Y_train, Y_test = \
        sklearn.model_selection.train_test_split(X, Y,
                                                 test_size=test_size,
                                                 random_state=0)
    return X_train, X_test, Y_train, Y_test


def time_series_split(features, labels, n_splits):
    """Split a dataset into n splits.

    """
    xx = sklearn.model_selection.TimeSeriesSplit(n_splits)
    for train_index, test_index in xx.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    return X_train, X_test, y_train, y_test
