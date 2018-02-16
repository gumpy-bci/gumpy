import threading
import pylsl
import numpy as np
from preprocess import extract_features
import time


class LiveRecorder():
    def __init__(self, sampling_frequency=256.0, classification_time=1.0):
        streams = pylsl.resolve_stream('type', 'EEG')
        inlet = pylsl.stream_inlet(streams[0])

        sample, _ = inlet.pull_sample()
        n_channels = len(sample)

        classification_length = int(sampling_frequency * classification_time)

        self.channel_data = np.zeros([n_channels, classification_length * 4])
        self.inlet = inlet
        self.classification_length = classification_length
        self.i_channel_data = 0

    def start_recording(self):
        def record():
            while True:
                sample, _ = self.inlet.pull_sample()
                self.channel_data[:, self.i_channel_data] = sample

                self.i_channel_data += 1
                if self.i_channel_data == self.channel_data.shape[1]:
                    self.i_channel_data = 0

        record_thread = threading.Thread(target=record)
        record_thread.daemon = True
        record_thread.start()

    def get_features(self):
        i_channel_data = self.i_channel_data
        i_start = i_channel_data - self.classification_length

        if i_start < 0:
            length_from_end = -i_start
            length_from_start = self.classification_length - length_from_end

            channel_data_copy = np.hstack((
                self.channel_data[:, 0:length_from_start],
                self.channel_data[:, self.channel_data.shape[1] - length_from_end:]
            ))
        else:
            channel_data_copy = self.channel_data[:, i_start:i_start + self.classification_length]

        features, _ = extract_features(channel_data_copy)

        return features.flatten()


if __name__ == '__main__':
    recorder = LiveRecorder()
    recorder.start_recording()

    while True:
        recorder.get_features()
        time.sleep(np.random.randint(5))
