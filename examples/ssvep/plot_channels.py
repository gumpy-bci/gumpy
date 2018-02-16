import numpy as np
import pylsl
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import argparse

streams = pylsl.resolve_stream('type', 'EEG')
inlet   = pylsl.stream_inlet(streams[0])


def default_channels():
    sample, _ = inlet.pull_sample()
    return range(len(sample))


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--channels', help="channels to plot, defaults to all",
                    nargs='+', type=int, default=default_channels())

args = vars(parser.parse_args())

channels = args["channels"]


def get_sample():
    sample, time_stamp = inlet.pull_sample()
    time_stamp += inlet.time_correction()
    time_stamp, sample
    return [sample[i] for i in channels]


app = QtGui.QApplication([])
win = pg.GraphicsWindow()

plots  = []
curves = []
ys     = np.zeros([len(channels), 1000])
for i in range(len(channels)):
    plots.append(win.addPlot(row=i, col=0))
    curves.append(plots[i].plot())


def update():
    samples = get_sample()

    for i in range(len(channels)):
        ys[i, :] = np.hstack((ys[i, 1:], samples[i]))
        curves[i].setData(ys[i, :])

    app.processEvents()


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_'):
        QtGui.QApplication.instance().exec_()
