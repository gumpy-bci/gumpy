from sklearn.externals import joblib
import numpy as np

def getData(data):
    labeled_features = data.item(0)["labeled_features"]
    frequencies = data.item(0)["frequencies"]
    labels = []
    features = []
    for labeled_feature in labeled_features:
        labels.append(labeled_feature['label'])
        features.append(labeled_feature['features'].flatten())
    return labels, features

model_file = 'QDA.pkl' #You can replace this by any model produced by multipleClassifiers file
#preprocess_recordings('19_06_05_07_2017_freq_19.mat')
data_test2 = np.load('19_06_05_07_2017_freq_19.mat.npy')
labels2, features2 = getData(data_test2) #Replace this with online data
labels2 = np.array(labels2)

classifier = joblib.load(model_file)
out_test = classifier.predict(features2)
accuracy = 100 - 100 * np.sqrt(np.sum(np.power(labels2 - out_test, 2))) / len(labels2)
print("Classifier accuracy: " + str(accuracy))
