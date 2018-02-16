import sys, os, os.path
sys.path.append('../../gumpy')


import numpy as np
import gumpy


# First specify the location of the data and some 
# identifier that is exposed by the dataset (e.g. subject)
base_dir = '../Data/NST-EMG'
subject = 'S1'

# The next line first initializes the data structure. 
# Note that this does not yet load the data! In custom implementations
# of a dataset, this should be used to prepare file transfers, 
# for instance check if all files are available, etc.
data_low = gumpy.data.NST_EMG(base_dir, subject, 'low')
data_high = gumpy.data.NST_EMG(base_dir, subject, 'high') 

# Finally, load the dataset
data_low.load()
data_high.load()

# Printing Informations About the dataset
data_low.print_stats()
data_high.print_stats()


# Filtering the Signals
#bandpass
lowcut=20
highcut=255
#notch
f0=50
Q=50

flt_low = gumpy.signal.butter_bandpass(data_low, lowcut, highcut)
flt_low = gumpy.signal.notch(flt_low, cutoff=f0, Q=Q)

trialsLow = gumpy.utils.getTrials(data_low, flt_low)
trialsLowBg = gumpy.utils.getTrials(data_low, flt_low, True)

flt_high = gumpy.signal.butter_bandpass(data_high, lowcut, highcut)
flt_high = gumpy.signal.notch(flt_high, cutoff=f0, Q=Q)

trialsHigh = gumpy.utils.getTrials(data_high, flt_high)
trialsHighBg = gumpy.utils.getTrials(data_high, flt_high, True)


# Creating an RMS feature extraction function
def RMS_features_extraction(data, trialList, window_size, window_shift):
    if window_shift > window_size:
        raise ValueError("window_shift > window_size")

    fs = data.sampling_freq
    
    n_features = int(data.duration/(window_size-window_shift))
    
    X = np.zeros((len(trialList), n_features*4))
    
    t = 0
    for trial in trialList:
        # x3 is the worst of all with 43.3% average performance
        x1=gumpy.signal.rms(trial[0], fs, window_size, window_shift)
        x2=gumpy.signal.rms(trial[1], fs, window_size, window_shift)
        x3=gumpy.signal.rms(trial[2], fs, window_size, window_shift)
        x4=gumpy.signal.rms(trial[3], fs, window_size, window_shift)
        x=np.concatenate((x1, x2, x3, x4))
        X[t, :] = np.array([x])
        t += 1
    return X


# Retrieving the features
window_size = 0.2
window_shift = 0.05

highRMSfeatures = RMS_features_extraction(data_high, trialsHigh, window_size, window_shift)
highRMSfeaturesBg = RMS_features_extraction(data_high, trialsHighBg, window_size, window_shift)
lowRMSfeatures = RMS_features_extraction(data_high, trialsLow, window_size, window_shift)
lowRMSfeaturesBg = RMS_features_extraction(data_high, trialsLowBg, window_size, window_shift)



# Constructing Classification arrays
X_tot = np.vstack((highRMSfeatures, lowRMSfeatures))
y_tot = np.hstack((np.ones((highRMSfeatures.shape[0])),
                     np.zeros((lowRMSfeatures.shape[0]))))
  
X_totSig = np.vstack((highRMSfeatures, highRMSfeaturesBg, lowRMSfeatures, lowRMSfeaturesBg))
X_totSig = X_totSig/np.linalg.norm(X_totSig)

#pHigh.labels = np.hstack((self.labels, 3*np.ones(self.trials.shape[0]/3)))
y_totSig = np.hstack((data_high.labels, 
                     data_low.labels))

  

# Posture Classification
(clf, sfs) = gumpy.features.sequential_feature_selector(X_totSig, y_totSig, 'SVM', (10,25), 3, 'SFFS')

# Force Level Classification
(clfF, sfsF) = Sequential_Feature_Selector(X_tot, y_tot, 'SVM', (10,25), 3, 'SFFS')