from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io


class GrazB(Dataset):
    """An NST dataset.

    An NST dataset usually consists of three files that are within a specific
    subdirectory. The implementation follows this structuring, i.e. the user
    needs to pass a base-directory as well as the identifier upon instantiation.

    """

    def __init__(self, base_dir, identifier, **kwargs):
        """Initialize a GrazB dataset without loading it.

        Args:
            base_dir (str): The path to the base directory in which the GrazB dataset resides.
            identifier (str): String identifier for the dataset, e.g. `B01`
            **kwargs: Arbitrary keyword arguments (unused).

        """

        super(GrazB, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = base_dir
        self.data_type = 'EEG'
        self.data_name = 'GrazB'

        # parameters of the GrazB dataset
        # length of a trial (in seconds)
        self.trial_len = 8
        # motor imagery appears in interval (in seconds)
        self.mi_interval = [4, 7]
        # idle perior prior to start of signal (in seconds)
        self.trial_offset = 0
        # total length of a trial (in seconds)
        self.trial_total = self.trial_len
        # sampling frequency (in Hz)
        self.expected_freq_s = 250

        # the graz dataset is split into T and E files
        self.fT = os.path.join(self.data_dir, "{id}T.mat".format(id=self.data_id))
        self.fE = os.path.join(self.data_dir, "{id}E.mat".format(id=self.data_id))

        for f in [self.fT, self.fE]:
            if not os.path.isfile(f):
                raise DatasetError("GrazB Dataset ({id}) file '{f}' unavailable".format(id=self.data_id, f=f))

        # variables to store data
        self.raw_data = None
        self.labels = None
        self.trials = None
        self.sampling_freq = None


    def load(self, **kwargs):
        """Load a dataset.

        Args:
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Instance to the dataset (i.e. `self`).

        """


        mat1 = scipy.io.loadmat(self.fT)['data']
        #mat2 = scipy.io.loadmat(folder_dir + file_dir2)['data']
        # dict_keys(['__header__', '__globals__', '__version__', 'data'])

        # Load Test Data
        data_bt = []
        labels_bt = []
        trials_bt = []
        n_experiments = 3
        for i in range(n_experiments):
            data      = mat1[0,i][0][0][0]
            trials    = mat1[0,i][0][0][1]
            labels    = mat1[0,i][0][0][2] - 1
            # TODO: fs shadows self.fs? do we need to store this somewhere?
            fs        = mat1[0,i][0][0][3].flatten()[0]
            if fs != self.expected_freq_s:
                raise DatasetError("GrazB Dataset ({id}) Sampling Frequencies don't match (expected {f1}, got {f2})".format(id=self.data_id, f1=self.expected_freq_s, f2=fs))
            artifacts = mat1[0,i][0][0][5]
            # remove artivacts
            artifact_idxs = np.where(artifacts == 1)[0]
            trials = np.delete(trials, artifact_idxs)
            labels = np.delete(labels, artifact_idxs)
            # add data to files
            data_bt.append(data)
            labels_bt.append(labels)
            trials_bt.append(trials)

        # add length of previous data set to adjust trial start points
        trials_bt[1] += data_bt[0].shape[0]
        trials_bt[2] += data_bt[0].shape[0] + data_bt[1].shape[0]

        # concatenate all data mat, trials, and labels
        data_bt = np.concatenate((data_bt[0], data_bt[1], data_bt[2]))
        trials_bt = np.concatenate((trials_bt[0], trials_bt[1], trials_bt[2]))
        labels_bt = np.concatenate((labels_bt[0], labels_bt[1], labels_bt[2]))

        self.raw_data = data_bt[:,:3]
        self.trials = trials_bt
        self.labels = labels_bt
        self.sampling_freq = self.expected_freq_s

        return self

