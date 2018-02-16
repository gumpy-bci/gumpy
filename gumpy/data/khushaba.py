from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io


# TODO: BROKEN!
class Khushaba(Dataset):
    """A Khushaba dataset.

    An Khushaba dataset usually consists of three files that are within a specific
    subdirectory. The implementation follows this structuring, i.e. the user
    needs to pass a base-directory as well as the identifier upon instantiation.

    """

    def __init__(self, base_dir, identifier, class_labels=[], **kwargs):
        """Initialize a Khushaba dataset without loading it.

        Args:
            base_dir (str): The path to the base directory in which the Khushaba dataset resides.
            identifier (str): String identifier for the dataset, e.g. `S1`
            class_labels (list): A list of class labels
            **kwargs: Additional keyword arguments (unused)

        """

        super(Khushaba, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = os.path.join(self.base_dir, self.data_id)
        self.data_type = 'EMG'
        self.data_name = 'Khushaba'

        self._class_labels = ['Ball', 'ThInd', 'ThIndMid', 'Ind', 'LRMI', 'Th']
        self._force_levels = ['high', 'low', 'med']
        # number of classes in the dataset
        if not isinstance(class_labels, list):
            raise ValueError('Required list of class labels (`class_labels`)')

        self.class_labels = class_labels

        # all Khushaba datasets have the same configuration and parameters

        # length of a trial after trial_sample (in seconds)
        self.trial_len = None
        # idle period prior to trial start (in seconds)
        self.trial_offset = None
        # total time of the trial
        self.trial_total = None #self.trial_offset + self.trial_len
        # interval of motor imagery within trial_t (in seconds)
        self.mi_interval = [self.trial_offset, self.trial_offset + self.trial_len]

        # additional variables to store data as expected by the ABC
        self.raw_data = None
        self.trials = None
        self.labels = None
        self.sampling_freq = 2000


    def load(self, **kwargs):
        """Loads a Khushaba dataset.

        For more information about the returned values, see
        :meth:`gumpy.data.Dataset.load`
        """

        self.trials = ()
        self.labels = ()

        for class_name in self.class_labels:
            classTrials, label_list = self.getClassTrials(class_name)
            self.trials = self.trials + (classTrials,)

            for trial in self.trials:
                if self.raw_data is None:
                    self.raw_data = trial
                else:
                    self.raw_data = np.concatenate((self.raw_data, trial))

            self.labels = self.labels + (label_list,)

        return self


    def getClassTrials(self, class_name):
        """Return all class trials and labels.

        Args:
            class_name (str): The class name for which the trials should be returned

        Returns:
            A 2-tuple containing

            - **trials**: A list of all trials of `class_name`
            - **labels**: A list of corresponding labels for the trials

        """
        Results = []
        label_list = []

        for force_level in self._force_levels:
            path = base_dir+'{}_Force Exp/{}_{}/'.format(self.data_id, class_name, force_level)

            for i in range(1,6):
                file = path+'{}_{}_{}_t{}.mat'.format(self.data_id, class_name, force_level, str(i))

                trial = scipy.io.loadmat(file)['t{}'.format(i)]

                Results.append(trial)
                label_list.append(self._class_labels.index(class_name))

        return Results, label_list
