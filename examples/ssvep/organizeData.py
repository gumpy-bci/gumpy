# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:48:44 2017

This function shuffles the data, in order to have a balanced training

@author: Sebastian
"""
import copy
import numpy as np

def organize_data(data):
    features = data.item(0).get('features')
    labels = data.item(0).get('labels')
    newFeatures = copy.deepcopy(features)
    newLabels = copy.deepcopy(labels)
    counter = [0,0,0,0]
        
    for i in range(0,len(features)):
        if labels[i] == 13.0:
            j = 0
        if labels[i] == 15.0:
            j = 1
        if labels[i] == 17.0:
            j = 2
        if labels[i] == 19.0:
            j= 3
        newFeatures[counter[j]*4 - j] = features[i]
        newLabels[counter[j]*4 - j] = labels[i]
        counter[j] = counter[j] + 1
    newData = {'features':newFeatures , 'labels':newLabels}
    return newData

data = np.load('preprocessed_data.npy');
print(organize_data(data))
#print(newLabels)
