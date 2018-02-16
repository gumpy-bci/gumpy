from abc import ABC, abstractmethod
import numpy as np

class DatasetError(Exception):
    pass


class Dataset(ABC):
    """
    Abstract base class representing a dataset.

    All datasets should subclass from this baseclass and need to implement the
    `load` function. Initializing of the dataset and actually loading the data is
    separated as the latter may require significant time, depending on where the
    data is coming from. It also allows to implement different handlers for the
    remote end where the data originates, e.g. download from server, etc.

    When subclassing form Dataset it is helpful to set fields `data_type`,
    `data_name`, and `data_id`. For more information on this field, see for
    instance the implementation in :func:`gumpy.data.graz.GrazB.__init__`.

    """


    def __init__(self, **kwargs):
        """Initialize a dataset."""
        pass


    @abstractmethod
    def load(self, **kwargs):
        """Load the data and prepare it for usage.

        gumpy expects the EEG/EMG trial data to be in the following format:

            ===========================================> time
                |                                   |
            trial_start                         trial_end
                |<------------trial_len------------>|
                                |<---MotorImager--->|


        Consequentially the class members need to adhere the following structure

            .raw_data       (n_samples, n_channels)  return all channels
            .trials         (,n_trials)
            .labels         (,n_labels)
            .trial_len      scalar
            .sampling_freq  scalar
            .mi_interval    [mi_start, mi_end] within a trial in seconds

        Arrays, such as `.raw_data` have to be accessible using bracket
        notation `[]`. You can provide a custom implementation, however the
        easiest way is to use numpy ndarrays to store the data.

        For an example implementation, have a look at `gumpy.data.nst.NST`.
        """
        return self


    def print_stats(self):
        """Commodity function to print information about the dataset.

        This method uses the fields that need to be implemented when
        subclassing. For more information about the fields that need to be
        implemented see :func:`gumpy.data.dataset.Dataset.load` and
        :func:`gumpy.data.dataset.Dataset.__init__`.
        """

        print("Data identification: {name}-{id}".format(name=self.data_name, id=self.data_id))
        print("{type}-data shape: {shape}".format(type=self.data_type, shape=self.raw_data.shape))
        print("Trials data shape: ", self.trials.shape)
        print("Labels shape: ", self.labels.shape)
        print("Total length of single trial: ", self.trial_total)
        print("Sampling frequency of {type} data: {freq}".format(type=self.data_type, freq=self.sampling_freq))
        print("Interval for motor imagery in trial: ", self.mi_interval)
        print('Classes possible: ', np.unique(self.labels))


