from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io


class NST_EMG(Dataset):
    """An NST_EMG dataset.

    An NST_EMG dataset usually consists of three files that are within a specific
    subdirectory. The implementation follows this structuring, i.e. the user
    needs to pass a base-directory as well as the identifier upon instantiation.

    If you require a copy of the data, please contact one of the gumpy authors.

    """

    def __init__(self, base_dir, identifier, force_level, **kwargs):
        """Initialize an NST_EMG dataset without loading it.

        Args:
            base_dir (str): The path to the base directory in which the NST_EMG dataset resides.
            identifier (str): String identifier for the dataset, e.g. ``S1``
            **kwargs: Additional keyword arguments: n_classes (int, default=3): number of classes to fetch.

        """

        super(NST_EMG, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.force_level = force_level
        self.data_dir = os.path.join(self.base_dir, self.data_id)
        self.data_type = 'EMG'
        self.data_name = 'NST_EMG'

        self.electrodePairList = [(0, 2), (1, 3), (4, 6), (5,7)]
        self.channel = []
        self.trialSignalOffset = (0.5,5.5)
        self.trialBgOffset = (5.5,10.5)
        self.trialForceOffset = (5,10)
        self.duration = 5

        # number of classes in the dataset
        self.n_classes = kwargs.pop('n_classes', 3)

        # all NST_EMG datasets have the same configuration and parameters
        # length of a trial after trial_sample (in seconds)
        self.trial_len = 5
        # idle period prior to trial start (in seconds)
        self.trial_offset = 5
        # total time of the trial
        self.trial_total = self.trial_offset + self.trial_len
        # interval of motor imagery within trial_t (in seconds)
        self.mi_interval = [self.trial_offset, self.trial_offset + self.trial_len]

        # additional variables to store data as expected by the ABC
        self.raw_data = None
        self.trials = None
        self.labels = None
        self.sampling_freq = None

        file_list_highForce = []
        file_list_lowForce = []

        # S1
        if self.data_id == 'S1':
            file_list_highForce = ['session_14_26_15_01_2018.mat', 'session_14_35_15_01_2018.mat', 'session_14_43_15_01_2018.mat']
            file_list_lowForce = ['session_15_00_15_01_2018.mat', 'session_15_08_15_01_2018.mat', 'session_15_16_15_01_2018.mat']

        # S2
        elif self.data_id == 'S2':
            file_list_highForce = ['session_14_51_10_01_2018.mat', 'session_15_10_10_01_2018.mat', 'session_15_10_10_01_2018.mat']
            file_list_lowForce = ['session_15_25_10_01_2018.mat', 'session_15_32_10_01_2018.mat', 'session_15_45_10_01_2018.mat']

        # S3
        elif self.data_id == 'S3':
            file_list_highForce = ['session_13_04_16_01_2018.mat', 'session_13_10_16_01_2018.mat', 'session_13_18_16_01_2018.mat']
            file_list_lowForce = ['session_13_26_16_01_2018.mat', 'session_13_31_16_01_2018.mat', 'session_13_35_16_01_2018.mat']

        # S4
        elif self.data_id == 'S4':
            file_list_highForce = ['session_13_36_09_03_2018', 'session_13_39_09_03_2018']
            file_list_lowForce = ['session_13_42_09_03_2018', 'session_13_44_09_03_2018']

        # S4
        if self.force_level == 'high':
            self.fileList = file_list_highForce
        elif self.force_level == 'low':
            self.fileList = file_list_lowForce


    def load(self, **kwargs):
        """Load an NST_EMG dataset.

        For more information about the returned values, see
        :meth:`gumpy.data.Dataset.load`
        """

        trial_len = 5   # sec (length of a trial after trial_sample)
        trial_offset = 5    # idle period prior to trial start [sec]
        self.trial_total = trial_offset + trial_len    # total length of trial
        self.mi__interval = [trial_offset, trial_offset+trial_len] # interval of motor imagery within trial_t [sec]

        matrices = []
        raw_data_ = []
        labels_ = []
        trials_ = []
        forces_ = []

        for file in self.fileList:
            try:
                fname = os.path.join(self.data_dir, file)
                if fname.exists():
                    matrices.append(scipy.io.loadmat(fname))
            except Exception as e:
                print('An exception occured while reading file {}: {}'.format(file, e))


        # read matlab data
        for matrix in matrices:
            raw_data_.append(matrix['X'][:,:])
            labels_.append(matrix['Y'][:])
            trials_.append(matrix['trial'][:])

            #forces_.append(matrix['force'][:].T)

            size_X = len(matrix['X'][:,0])
            size_force = np.shape(matrix['force'][:])[1]

            #print(size_X)
            #print(size_force)

            Zero = size_X-size_force
            f = np.zeros((1, size_X))
            f[0, Zero:] = matrix['force'][:]
            forces_.append(f.T)


            #forces_.append(matrix['force'][:])

        # to get the correct values of the trials
        for i in range(1, len(trials_)):
            trials_[i] += raw_data_[i-1].T.shape[1]

        #combine matrices together
        self.raw_data = np.concatenate(tuple(raw_data_))
        self.labels = np.concatenate(tuple(labels_))
        self.trials = np.concatenate(tuple(trials_))
        self.forces = np.concatenate(tuple(forces_))

        # Resetting points higher than max intensity of force to 0
        self.forces[self.forces > 20] = 0
        # Resetting points lower than 0 to 0
        self.forces[self.forces < 0] = 0

        # Remove class 3
        c3_idxs = np.where(self.labels==3)[0]
        self.labels = np.delete(self.labels, c3_idxs)
        self.trials = np.delete(self.trials, c3_idxs)

        self.labels = np.hstack((self.labels, 3*np.ones(int(self.trials.shape[0]/3))))


        self.sampling_freq = matrices[0]['Fs'].flatten()[0]

        return self
