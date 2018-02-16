"""Real time EEG pre-processing, spectrogram generation and transmission demo.


This script uses the lab streaming layer (LSL) to recieve raw EEG data of a
stream with type "EEG" and transmits spectrograms on a stream with type
"EEGPre".

Pre-processing utilizes filter functions defined in gumpy.signal through the
live_processing filterbank.

If verbose is set to True, the processing pipeline steps are visualized. Be
aware that this limits the maximum output rate.

"""

import math
import sys

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pylsl
from scipy.signal import spectrogram

import ringbuffer
import live_processing as lp

# ============================================================================
# PARAMETERS
# ============================================================================
# 0. General
verbose = False
# 1. Output
output_width    = 32
output_height   = 32
output_stacks   = 3  # channels
outlet_sendRate = 2 # [Hz]
# 2. Filterbank
lowcut  = 2  # [Hz]
highcut = 60 # [Hz]
order   = 3
# 3. Spectrogram Generation
periods      = 1.5  # NFFT = 1/lowcut * Fs * periods.
overlapRatio = 0.95

# ============================================================================
# PROCESS
# ============================================================================
print("Opening inlet")
streams = pylsl.resolve_stream('type', 'EEG')
inlet   = pylsl.stream_inlet(streams[0])

inletInfo = inlet.info()

inlet_sampleRate = int(inletInfo.nominal_srate())
inlet_numChannels = int(inletInfo.channel_count())
if verbose:
    print("Reported sample rate: %i , number of channels: %i" %(inlet_sampleRate, inlet_numChannels))

filterbank = lp.filterbank(lowcut,highcut,order,inlet_sampleRate)

specGen = lp.specGen(output_width, output_height, output_stacks, lowcut, periods, overlapRatio, inlet_sampleRate)
if verbose:
    print("window width: %i - overlap: %i - buffer size: %i" %(specGen.SFFTwindowWidth, specGen.SFFToverlap, specGen.smpPerSpec))

rbuffer = ringbuffer.RingBuffer(size_max=specGen.smpPerSpec)
sendEverySmpl = math.ceil(inlet_sampleRate / outlet_sendRate)
if verbose:
    print("Transmitting every %i samples" %sendEverySmpl)

print("Creating outlet")
outlet_numChannels = output_width*output_height*output_stacks
info = pylsl.StreamInfo('ContinuousEEG', 'EEGPre', outlet_numChannels, outlet_sendRate, 'int8', 'transmissionTest')
outlet = pylsl.StreamOutlet(info)

samplesInBuffer = 0
samplesSent = 0

if verbose:
    fig = plt.figure(figsize=(10,8))
    outer = gridspec.GridSpec(3, 1, wspace=0, hspace=0.2)

    class closer:
        def __init__(self):
            self.run = True

        def handle_close(self, evt):
            print('\nClosed figure, shutting down')
            self.run = False

    obs = closer()
    fig.canvas.mpl_connect('close_event', obs.handle_close)
    print("Close the figure to stop the application.")
else:
    class closer:
        def __init__(self):
            self.run = True

    obs = closer()

while obs.run:
    rbuffer.append(inlet.pull_sample()[0])
    samplesInBuffer += 1

    if(rbuffer.full and samplesInBuffer>=sendEverySmpl):
        data_raw = np.array(rbuffer.get())[:,0:3]

        data_filtered = filterbank.process(data_raw)

        data_batch = specGen.process(data_filtered)

        outlet.push_sample(data_batch.flatten())

        samplesSent += 1
        sys.stdout.write('\rsamples sent: %i' %samplesSent) # \r requires stdout to work
        samplesInBuffer = 0

        if verbose:
            if samplesSent < 2:
                inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], wspace=0, hspace=0.05)
                ax11 = plt.Subplot(fig,inner[0])
                h11, = ax11.plot(data_raw[:,0])
                fig.add_subplot(ax11)
                ax12 = plt.Subplot(fig,inner[1])
                h12, = ax12.plot(data_raw[:,1])
                fig.add_subplot(ax12, sharex=ax11)
                ax13 = plt.Subplot(fig,inner[2])
                h13, = ax13.plot(data_raw[:,2])
                fig.add_subplot(ax13, sharex=ax11)
                inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], wspace=0, hspace=0.05)
                ax21 = plt.Subplot(fig,inner[0])
                h21, = ax21.plot(data_filtered[:,0])
                fig.add_subplot(ax21)
                ax22 = plt.Subplot(fig,inner[1])
                h22, = ax22.plot(data_filtered[:,1])
                fig.add_subplot(ax22, sharex=ax21)
                ax23 = plt.Subplot(fig,inner[2])
                h23, = ax23.plot(data_filtered[:,2])
                fig.add_subplot(ax23, sharex=ax21)
                inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.1, hspace=0)
                ax31 = plt.Subplot(fig,inner[0])
                h31 = ax31.pcolormesh(data_batch[0,:,:])
                fig.add_subplot(ax31)
                ax32 = plt.Subplot(fig,inner[1])
                h32 = ax32.pcolormesh(data_batch[1,:,:])
                fig.add_subplot(ax32, sharey=ax31)
                ax33 = plt.Subplot(fig,inner[2])
                h33 = ax33.pcolormesh(data_batch[2,:,:])
                fig.add_subplot(ax33, sharey=ax31)
            else:
                h11.set_ydata(data_raw[:,0])
                h12.set_ydata(data_raw[:,1])
                h13.set_ydata(data_raw[:,2])
                h21.set_ydata(data_filtered[:,0])
                h22.set_ydata(data_filtered[:,1])
                h23.set_ydata(data_filtered[:,2])
                h31.set_array(data_batch[0,:,:].ravel())
                h32.set_array(data_batch[1,:,:].ravel())
                h33.set_array(data_batch[2,:,:].ravel())
                ax11.draw_artist(h11)
                ax12.draw_artist(h12)
                ax13.draw_artist(h13)
                ax21.draw_artist(h21)
                ax22.draw_artist(h22)
                ax23.draw_artist(h23)
                ax31.draw_artist(h31)
                ax32.draw_artist(h32)
                ax33.draw_artist(h33)
            plt.pause(0.0000001)
