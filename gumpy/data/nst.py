from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io


class NST(Dataset):
    """An NST dataset.

    An NST dataset usually consists of three files that are within a specific
    subdirectory. The implementation follows this structuring, i.e. the user
    needs to pass a base-directory as well as the identifier upon instantiation.

    """

    def __init__(self, base_dir, identifier, **kwargs):
        """Initialize an NST dataset without loading it.

        Args:
            base_dir (str): The path to the base directory in which the NST dataset resides.
            identifier (str): String identifier for the dataset, e.g. ``S1``
            **kwargs: Additional keyword arguments: n_classes (int, default=3): number of classes to fetch.

        """

        super(NST, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = os.path.join(self.base_dir, self.data_id)
        self.data_type = 'EEG'
        self.data_name = 'NST'

        # number of classes in the dataset
        self.n_classes = kwargs.pop('n_classes', 3)

        # all NST datasets have the same configuration and parameters
        # length of a trial after trial_sample (in seconds)
        self.trial_len = 4
        # idle period prior to trial start (in seconds)
        self.trial_offset = 4
        # total time of the trial
        self.trial_total = self.trial_offset + self.trial_len+2
        # interval of motor imagery within trial_t (in seconds)
        self.mi_interval = [self.trial_offset, self.trial_offset + self.trial_len]

        # additional variables to store data as expected by the ABC
        self.raw_data = None
        self.trials = None
        self.labels = None
        self.sampling_freq = None

        # TODO: change the files on disk, don't check in here...
        # the first few sessions had a different file type
        self.f0 = os.path.join(self.data_dir, 'Run1.mat')
        self.f1 = os.path.join(self.data_dir, 'Run2.mat')
        self.f2 = os.path.join(self.data_dir, 'Run3.mat')

        # check if files are available
        for f in [self.f0, self.f1, self.f2]:
            if not os.path.isfile(f):
                raise DatasetError("NST Dataset ({id}) file '{f}' unavailable".format(id=self.data_id, f=f))


    def load(self, **kwargs):
        """Loads an NST dataset.

        For more information about the returned values, see
        :meth:`gumpy.data.Dataset.load`
        """
        mat1 = scipy.io.loadmat(self.f0)
        mat2 = scipy.io.loadmat(self.f1)
        mat3 = scipy.io.loadmat(self.f2)

        fs = mat1['Fs'].flatten()[0]
        # read matlab data
        raw_data1 = mat1['X'][:,0:3]
        raw_data2 = mat2['X'][:,0:3]
        raw_data3 = mat3['X'][:,0:3]
        trials1 = mat1['trial'][0]
        trials2 = mat2['trial'][0]
        trials3 = mat3['trial'][0]
        raw_data3 = mat3['X'][:,0:3]

        # extract labels
        labels1 = mat1['Y'].flatten() - 1
        labels2 = mat2['Y'].flatten() - 1
        labels3 = mat3['Y'].flatten() - 1

        # prepare trial data
        trials1 = mat1['trial'].flatten() -  fs*self.trial_offset
        trials2 = mat2['trial'].flatten() -  fs*self.trial_offset
        trials3 = mat3['trial'].flatten() -  fs*self.trial_offset
        trials2 += raw_data1.T.shape[1]
        trials3 += raw_data1.T.shape[1] + raw_data2.T.shape[1]

        # concatenate matrices
        self.raw_data = np.concatenate((raw_data1, raw_data2, raw_data3))
        self.labels = np.concatenate((labels1, labels2, labels3))
        self.trials = np.concatenate((trials1, trials2, trials3))
        self.sampling_freq = fs
        if self.n_classes == 2: # Remove class 3 if desired
            c3_idxs = np.where(self.labels==2)[0]
            self.labels = np.delete(self.labels, c3_idxs)
            self.trials = np.delete(self.trials, c3_idxs)

        return self
