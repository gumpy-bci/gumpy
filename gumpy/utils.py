"""Utility functions that may be used during data processing.

Because many datasets differ slightly, not all utility functions may work with
each dataset. However, the modifications are typically only minor, and thus the
functions provided within this module can be adapted easily.
"""

from .data.dataset import Dataset
import signal
import numpy as np

# TODO: documentation


def extract_trials(data, filtered=None, trials=None, labels=None, sampling_freq=0):

    if isinstance(data, Dataset) or (filtered is not None):
        # extract all necessary information from the dataset
        fs = data.sampling_freq
        labels = data.labels
        trial_len = data.trial_len
        trial_offset = data.trial_offset
        trials = data.trials

        # determine if to work on raw_data or if filtered information was passed
        # along
        if filtered is None:
            _data = data.raw_data
        else:
            _data = filtered
    else:
        _data = data
        fs = sampling_freq
        trial_len=8
        trial_offset=0


    # Indices of class 1 and 2
    c1_idxs = np.where(labels == 0)[0]  # 1 means left
    c2_idxs = np.where(labels == 1)[0]  # 2 means right
    c1_trials = trials[c1_idxs]
    c2_trials = trials[c2_idxs]

    # Init arrays (#trials, length_trial)
    raw_c3_c1_a = np.zeros((len(c1_idxs), fs*(trial_len+trial_offset)))
    raw_c4_c1_a = np.zeros((len(c1_idxs), fs*(trial_len+trial_offset)))
    raw_cz_c1_a = np.zeros((len(c1_idxs), fs*(trial_len+trial_offset)))
    raw_c3_c2_a = np.zeros((len(c2_idxs), fs*(trial_len+trial_offset)))
    raw_c4_c2_a = np.zeros((len(c2_idxs), fs*(trial_len+trial_offset)))
    raw_cz_c2_a = np.zeros((len(c2_idxs), fs*(trial_len+trial_offset)))

    # Add eeg trial data to array
    for i,(idx_c1, idx_c2) in enumerate(zip(c1_trials, c2_trials)):
        raw_c3_c1_a[i,:] = _data[idx_c1-(trial_offset*fs) : idx_c1+(trial_len*fs), 0]
        raw_c4_c1_a[i,:] = _data[idx_c1-(trial_offset*fs) : idx_c1+(trial_len*fs), 2]
        raw_cz_c1_a[i,:] = _data[idx_c1-(trial_offset*fs) : idx_c1+(trial_len*fs), 1]

        raw_c3_c2_a[i,:] = _data[idx_c2-(trial_offset*fs) : idx_c2+(trial_len*fs), 0]
        raw_c4_c2_a[i,:] = _data[idx_c2-(trial_offset*fs) : idx_c2+(trial_len*fs), 2]
        raw_cz_c2_a[i,:] = _data[idx_c2-(trial_offset*fs) : idx_c2+(trial_len*fs), 1]

    return np.array((raw_c3_c1_a, raw_c4_c1_a, raw_cz_c1_a, raw_c3_c2_a, raw_c4_c2_a, raw_cz_c2_a))



def _retrieveTrialSlice(data, trialIndex, type='signal'):
    if type=='signal':
        return slice(int(data.trials[trialIndex] +
                            data.trialSignalOffset[0]*data.sampling_freq), int(data.trials[trialIndex] +
                            data.trialSignalOffset[1]*data.sampling_freq))

    elif type=='force':
        return slice(int(data.trials[trialIndex] +
                            data.trialForceOffset[0]*data.sampling_freq), int(data.trials[trialIndex] +
                            data.trialForceOffset[1]*data.sampling_freq))

    elif type=='background':
        return slice(int(data.trials[trialIndex] +
                            data.trialBgOffset[0]*data.sampling_freq), int(data.trials[trialIndex] +
                            data.trialBgOffset[1]*data.sampling_freq))

    else:
        raise AttributeError('type should be "signal" or "force".')



def _processData(data, type='signal'):
    if type=='signal':
        return data
    elif type=='force':
        try:
            return data/max(data)
        except ValueError:
            return data



def getTrials(data, filtered=None, background=False):
    data.channel = []

    raw_data = data.raw_data
    if filtered is not None:
        raw_data = filtered

    for pair in data.electrodePairList:
        data.channel.append(_processData(raw_data[:, pair[0]]-
                                        raw_data[:, pair[1]]))

    processedForces = _processData(data.forces, 'force')

    if background:
        return [(data.channel[0][_retrieveTrialSlice(data, i%3, 'background')],
                 data.channel[1][_retrieveTrialSlice(data, i%3, 'background')],
                 data.channel[2][_retrieveTrialSlice(data, i%3, 'background')],
                 data.channel[3][_retrieveTrialSlice(data, i%3, 'background')],
                 processedForces[_retrieveTrialSlice(data, i%3, 'force')])
                  for i in range(int(data.trials.shape[0]/3))]
    else:
        return [(data.channel[0][_retrieveTrialSlice(data, i, 'signal')],
                 data.channel[1][_retrieveTrialSlice(data, i, 'signal')],
                 data.channel[2][_retrieveTrialSlice(data, i, 'signal')],
                 data.channel[3][_retrieveTrialSlice(data, i, 'signal')],
                 processedForces[_retrieveTrialSlice(data, i, 'force')])
                  for i in range(data.trials.shape[0])]

