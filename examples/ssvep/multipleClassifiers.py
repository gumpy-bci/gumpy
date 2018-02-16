from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_recordings, load_preprocessed_data
import os, numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

preprocess_recordings()
features, labels, _, _ = load_preprocessed_data()

classifiers = [LinearDiscriminantAnalysis(), MLPClassifier(), QuadraticDiscriminantAnalysis(), RandomForestClassifier()]
clfNames = ['LDA', 'MLP', 'QDA', 'RF']

training_labels, training_features = labels, features
test_labels, test_features = labels, features

count = 0

for classifier in classifiers:
    classifier.fit(training_features, training_labels)
    joblib.dump(classifier, clfNames[count] + '.pkl')
    out_test = classifier.predict(test_features)
    ax = plt.subplot(1, len(classifiers) + 1, count + 1)
    # print(np.shape(features2))
    accuracy = 100 - 100 * np.sqrt(np.sum(np.power(test_labels - out_test, 2))) / len(test_labels)
    print(clfNames[count] + " accuracy: " + str(accuracy))
    count += 1
