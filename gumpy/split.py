import sklearn.model_selection
import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, cross_val_score, StratifiedKFold


def normal(X, labels, test_size):
    """Split a dataset into training and test parts.
     Args:
        X (numpy.ndarray): 2D features matrix 
        labels: labels vector 
        test_size: size of the split
        
    Returns:
        A 2D CSP features matrix 
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


def  stratified_KFold(features, labels, n_splits):

    """Stratified K-Folds cross-validator
     Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole
     and by also keeping the balance of classes
    """
    skf = StratifiedKFold(n_splits)
    skf.get_n_splits(features, labels)
    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
    return X_train, X_test, Y_train, Y_test

#Stratified ShuffleSplit cross-validator
def  stratified_shuffle_Split(features, labels, n_splits,test_size,random_state):

    """Stratified ShuffleSplit cross-validator
    """
    cv = StratifiedShuffleSplit(n_splits, test_size, random_state=random_state)
    for train_index, test_index in cv.split(features,labels):
        X_train = features[train_index]
        X_test = features[test_index]
        Y_train = labels[train_index]
        Y_test = labels[test_index]
    return X_train, X_test, Y_train, Y_test


#Random permutation cross-validator
def  shuffle_Split(features, labels, n_splits,test_size,random_state):

    """ShuffleSplit: Random permutation cross-validator
    """
    cv = ShuffleSplit(n_splits, test_size, random_state=random_state)
    for train_index, test_index in cv.split(features):
        X_train = features[train_index]
        X_test = features[test_index]
        Y_train = labels[train_index]
        Y_test = labels[test_index]
    return X_train, X_test, Y_train, Y_test
