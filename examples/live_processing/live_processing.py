"""Utility scripts to provide a filterbank and spectrogram generator.


The filterbank utilizes filter functions defined in gumpy.signal.

To ensure a consistent spectrogram size, either cropping or scaling can be
used. Just change the corresponding lines in the specGen.process function.

"""

import math

import numpy as np
#from scipy.misc import imresize # for spectrogram scaling, requires Pillow
from scipy.signal import spectrogram

import gumpy.signal as gs

class filterbank:

	def __init__(self, lowcut=2, highcut=60, order=3, fs=256):
		self.bandPass = gs.butter_bandpass(lowcut,highcut,order,fs)
		self.notch = gs.butter_bandstop()
		#notch = gs.notch()

	def process(self, data):
		return self.notch.process(self.bandPass.process(data))

class specGen:

    def __init__(self, width = 32, height = 32, numChannels = 3, lowf = 2, periods = 1.5, overlapRatio = 0.95, fs=256):
        self.width = width
        self.height = height
        self.nChannels = numChannels
        self.fs = fs
        self.lowf = lowf # lowcut
        self.SFFTwindowWidth = int(math.ceil(fs/lowf * periods))
        self.SFFToverlap = int(math.floor(self.SFFTwindowWidth * overlapRatio))
        self.smpPerSpec = int(self.SFFTwindowWidth + (self.width - 1) * (self.SFFTwindowWidth - self.SFFToverlap))

    def process(self, data):
        # raw spectrogram generation
        specsRaw = []
        for iChannel in xrange(self.nChannels):
            specsRaw.append(spectrogram(data[:, iChannel], self.fs, nperseg=self.SFFTwindowWidth, noverlap=self.SFFToverlap, detrend=False)[2]) 

        # reshaping
        specs = np.zeros((self.nChannels, self.height, self.width))
        if specsRaw[1].shape[1]>self.width:
            start = spec_1.shape[1] - self.width
        else:
            start = 0

        for iChannel in xrange(self.nChannels):
            # cropped
            specs[iChannel, :, :] = specsRaw[iChannel][self.lowf:self.height+self.lowf, start:].copy()
            # scaled (requires imresize)
            #specs[iChannel, :, :] = imresize(arr=specsRaw[iChannel][self.lowf:, :], size=(self.height, self.width), interp='nearest', mode='F')

        return specs
        