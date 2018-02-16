import mne
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_welch

mne.set_log_level("ERROR")

recording_dir = os.path.join(os.path.dirname(__file__), "..", "REC")

stimulation_duration = 15.0
sampling_frequency   = 256.0
fmin                 = 10.0
fmax                 = 20.0

info = mne.create_info(ch_names=["O1", "OZ", "O2", "reference"],
                       ch_types=["eeg", "eeg", "eeg", "eeg"],
                       sfreq=sampling_frequency)


def extract_features(channel_data):
    mne_raw = mne.io.RawArray(data=channel_data, info=info)

    psds, frequencies = psd_welch(mne_raw, fmin=fmin, fmax=fmax)

    return psds, frequencies


def preprocess_recordings(output_file=os.path.join(recording_dir, "preprocessed_data")):
    """
    Preprocess all the data from the recording.mat files and saves it with the following
    structure:

    {
        'frequencies': [f1, f2, ...], # psd frequencies
        'labels'     : [l1, l2, ...], # labels for feature vectors
        'features'   : [
            [psd_of_f1_c1, psd_of_f2_c1, ..., psd_of_f1_c2, ...], # of channels for feature 1
            [psd_of_f1, psd_of_f2, ...]                           # of channels for feature 2
        ],
        'file_ids' : [0, 0, ..., 1, 1, ...] # unique id for the file the data comes from
    }
    """
    frequencies = []
    features    = []
    labels      = []
    file_ids    = []

    n_splits = 15
    offset = int(sampling_frequency * stimulation_duration / n_splits)
    frequency_to_pds_mean = dict()
    normalisation_factor_of = dict()
    for file_id, recording_mat_file in enumerate(os.listdir(recording_dir)):
        if not recording_mat_file.endswith(".mat"):
            continue

        recording_mat_file = os.path.join(recording_dir, recording_mat_file)

        recording_mat = sio.loadmat(recording_mat_file)

        Y                  = recording_mat["Y"][0, :]
        channel_data       = recording_mat["X"]
        time_stamp_indexes = recording_mat["trial"][0, :]

        for i, stimulation_begin in np.ndenumerate(time_stamp_indexes[::2]):
            file_ids.append(file_id)
            i = i[0]
            stimulation_end = stimulation_begin + int(sampling_frequency * stimulation_duration)

            for j in range(n_splits):
                current_channel_data = channel_data[stimulation_begin : stimulation_begin + offset]
                stimulation_begin += offset

                psds, frequencies = extract_features(current_channel_data.T)

                # only the even indexed labels contain labels with stimulation frequencies
                # uneven indexed labels are breaks and therefor always zero
                labels.append(Y[i * 2])
                features.append(psds.flatten())

            if Y[i * 2] in frequency_to_pds_mean:
                frequency_to_pds_mean[Y[i * 2]] += psds
                normalisation_factor_of[Y[i * 2]] += 1
            else:
                frequency_to_pds_mean[Y[i * 2]] = psds
                normalisation_factor_of[Y[i * 2]] = 1

    for frequency in frequency_to_pds_mean.keys():
        frequency_to_pds_mean[frequency] /= normalisation_factor_of[frequency]

        plt.plot(frequencies, frequency_to_pds_mean[frequency][0], label="Reference")
        plt.plot(frequencies, frequency_to_pds_mean[frequency][1], label="O1")
        plt.plot(frequencies, frequency_to_pds_mean[frequency][2], label="OZ")
        plt.plot(frequencies, frequency_to_pds_mean[frequency][3], label="O2")
        # plt.legend()

        plt.title("Stimulation Frequency " + str(frequency) + " (Hz)", fontsize=20)
        plt.ylabel("Relative Amplitude", fontsize=20)
        plt.xlabel("Frequency (Hz)", fontsize=20)
        plt.tick_params(axis='both', labelsize=15)
        plt.tight_layout()
        plt.savefig("pds_" + str(frequency) + ".png")
        # plt.show()
        plt.clf()

    np.save(output_file, {
        "features"    : features,
        "labels"      : labels,
        "file_ids"    : file_ids,
        "frequencies" : frequencies
    })


def load_preprocessed_data(input_file=os.path.join(recording_dir, "preprocessed_data.npy")):
    file_content = np.load(input_file).item()

    features    = file_content["features"]
    labels      = file_content["labels"]
    frequencies = file_content["frequencies"]
    file_ids    = file_content["file_ids"]

    return features, labels, file_ids, frequencies


if __name__ == '__main__':
    preprocess_recordings()
    # features, labels, file_ids, frequencies = load_preprocessed_data()
