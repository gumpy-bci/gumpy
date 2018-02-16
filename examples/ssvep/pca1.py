# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 20:27:33 2017

@author: Sebastian
"""
import numpy as np
#from Pre_processing import preprocess_recordings
import copy



def add_data(data,arrays):
    features = data.item(0).get('features')
    labels = data.item(0).get('labels')
    for i in range (0,len(features)):
        newData = features[i].reshape(44,1)
        if i == 0:
            arrays[str(int(labels[i]))] = newData
            arrays['general'] = newData
        arrays[str(int(labels[i]))] = np.append(arrays.get(str(int(labels[i]))), newData,axis=1)
        arrays['general'] = np.append(arrays.get('general'), newData,axis=1)

def create_model_pca(data):
    arrays = {"general": np.zeros(44).reshape(44,1) , "13" : np.zeros(44).reshape(44,1), "15": np.zeros(44).reshape(44,1),"17":np.zeros(44).reshape(44,1),"19":np.zeros(44).reshape(44,1)}
    meanArrays = {"general": np.zeros(44).reshape(44,1) , "13" : np.zeros(44).reshape(44,1), "15": np.zeros(44).reshape(44,1),"17":np.zeros(44).reshape(44,1),"19":np.zeros(44).reshape(44,1)}
    add_data(data,arrays)
    ZMarrays = copy.deepcopy(arrays)
    for key,value in arrays.items():
        #np.delete(arrays[key], 0, 1)
        meanArrays[key] = np.mean(arrays.get(key), axis=1)
        for i in range (0,arrays.get(key).shape[1]):  #substracting the mean to each sample
            ZMarrays[key][:,i] = arrays.get(key)[:,i] - meanArrays.get(key)
    
    covMatrix = np.cov(ZMarrays.get('general'))
    eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
    
    D = 23
    #for D in range (2,arrays.get(key).shape[0]-4):
    basis = eigenvectors[:,0:D]
    newClasses = copy.deepcopy(arrays)
    del newClasses['general']
    meanClasses = copy.deepcopy(newClasses)
    covClasses = copy.deepcopy(newClasses)
    
    for key,value in newClasses.items():
        newClasses[key] = np.dot(np.transpose(basis),arrays.get(key))
        meanClasses[key] = np.mean(newClasses.get(key), axis=1)
        covClasses[key] = np.cov(newClasses.get(key))
    model = {'newClasses':newClasses , 'meanClasses':meanClasses, 'covClasses':covClasses,'basis':basis,'D':[D]}
    return model

def predict_label(data , model):
    newClasses = model.get('newClasses')
    meanClasses = model.get('meanClasses')
    covClasses = model.get('covClasses')
    basis = model.get('basis')
    D = model.get('D')[0]
    testProjection = np.dot(np.transpose(basis),test)
    highestLikelihood = 0;
    keyL = '';
    for key,value in newClasses.items():
        likelihood = (1/(np.power(2*np.pi , D/2)*np.power(np.linalg.det(covClasses.get(key)) , 0.5)))*(np.exp(-(1/2*np.dot(np.dot(np.subtract(np.transpose(testProjection) , meanClasses.get(key)) , np.linalg.inv(covClasses.get(key))) , np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key)))))))
        if key == str(19):
            # print(testProjection)
            # print(np.subtract(np.transpose(testProjection) , meanClasses.get(key)) , np.linalg.inv(covClasses.get(key)))
            # print(np.dot(np.subtract(np.transpose(testProjection) , meanClasses.get(key)) , np.linalg.inv(covClasses.get(key))) , np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key))))
            print(1/2*np.dot(np.dot(np.subtract(np.transpose(testProjection) , meanClasses.get(key)) , np.linalg.inv(covClasses.get(key))) , np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key)))))
            print(1./2.)
            # print(np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key))))
            # print(np.linalg.inv(covClasses.get(key)))
            # print(np.linalg.inv(covClasses.get(key))) , np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key)))
            # print(np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key))))
            # print((np.dot(np.subtract(np.transpose(testProjection) , meanClasses.get(key)) , np.linalg.inv(covClasses.get(key))) , np.transpose(np.subtract(np.transpose(testProjection) , meanClasses.get(key)))))
            print(likelihood[0][0])
        #print(likelihood[0])
        #print(key, likelihood)
        if likelihood > highestLikelihood:
            highestLikelihood = likelihood
            keyL = key
    return keyL
 
"""I always initialize the variables; i m not sure if i am supossed to.... anyway
arrays = dictionary with arrays (44 x N) where N is the number of samples and 44 is the aligned data of each sample. One is general and the other are one for each requency
meanArrays = dictionary with arrays (44 X 1) with the mean of the 'arrays' data
ZMarrays = dictionary with arrays of the Zero Mean Arrays (substract the mean to each sample)"""

data = np.load('preprocessed_data.npy');
model = create_model_pca(data);

    
   
#Test: It goes through all data again. When trying on real time data, for loop is changed for a infinite loop
score = 0
for i in [0, 4]:
    test = data.item(0).get('features')[i].reshape(44,1) #this is the input variable
    keyL = predict_label(test , model)
    #print(int(data.item(0).get('labeled_features')[i].get('label')))
    # print("Frequency:  " + keyL + " Hz")#keyL is the result of the classification, given as a string. 
    if int(keyL) == int(data.item(0).get('labels')[i]):
        score = score + 1
print("success rate:" ,score*100/len(data.item(0).get('features')),"%")
