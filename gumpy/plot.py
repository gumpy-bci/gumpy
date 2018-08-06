"""Functions for plotting EEG processing results.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as skm
import seaborn as sns
import pandas as pd
import pywt
import scipy.signal
import sklearn.decomposition
from matplotlib.gridspec import GridSpec
from pylab import rcParams
import itertools


def plot_confusion_matrix(path, cm, target_names, title='Confusion matrix ', cmap=None, normalize=True):
    """Produces a plot for a confusion matrix and saves it to file.

    Args:
        path (str): Filename of produced plot
        cm (ndarray): confusion matrix from sklearn.metrics.confusion_matrix
        target_names ([str]): given classification classes such as [0, 1, 2] the
            class names, for example: ['high', 'medium', 'low']
        title (str): the text to display at the top of the matrix
        cmap: the gradient of the values displayed from matplotlib.pyplot.cm see
            http://matplotlib.org/examples/color/colormaps_reference.html
            plt.get_cmap('jet') or plt.cm.Blues
        normalize (bool): if False, plot the raw numbers. If True, plot the
            proportions

    Example:
        plot_confusion_matrix(cm           = cm,              # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    References:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    fig.savefig(path)


# TODO: check formatting (whitespaces, etc)
# TODO: check all variable names and improve them
def ROC_curve(Y_pred, Y_test, fig=None):
    Y_score = np.array(Y_pred)
    # The following were moved inside the function call (roc_curve) to avoid
    # potential side effects of this functin
    # Y_score -=1
    # Y_test -=1

    # print (roc_auc_score(y_test, y_score))

    fpr, tpr, _ = sklearn.metrics.roc_curve(Y_test - 1, Y_score - 1)

    # plotting
    if fig is None:
        fig = plt.figure()
    plt.plot(fpr, tpr, color= 'red', lw = 2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    plt.legend(loc="lower right")
    plt.show()


def confusion_matrix(true_labels, predicted_labels, cmap=plt.cm.Blues):
    cm = skm.confusion_matrix(true_labels, predicted_labels)
    # TODO:
    # print(cm)
    # Show confusion matrix in a separate window ?
    plt.matshow(cm,cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# TODO: permit the user to specify the figure where this plot shall appear
def accuracy_results_plot(data_path):
    data = pd.read_csv(data_path,index_col=0)
    sns.boxplot(data=data)
    sns.set(rc={"figure.figsize": (9, 6)})
    ax = sns.boxplot( data=data)
    ax.set_xlabel(x_label,fontsize=15)
    ax.set_ylabel(y_label,fontsize=15)
    plt.show()


def reconstruct_without_approx(xs, labels, level, fig=None):
    # reconstruct
    rs = [pywt.upcoef('d', x, 'db4', level=level) for x in xs]

    # generate plot
    if fig is None:
        fig = plt.figure()
    for i, x in enumerate(xs):
        plt.plot((np.abs(x))**2, label="Power of reconstructed signal ({})".format(labels[i]))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return rs, fig


def reconstruct_with_approx(cDs, labels, wavelet, fig=None):
    rs = [pywt.idwt(cA=None, cD=cD, wavelet=wavelet) for cD in cDs]

    if fig is None:
        fig = plt.figure()

    for i, r in enumerate(rs):
        plt.plot((np.abs(r))**2, label="Power of reconstructed signal ({})".format(labels[i]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return rs, fig


def fft(x, fs, fig_fft=None, fig_psd=None):
    t = np.arange(fs)
    signal_fft = np.fft.fft(x)
    signal_psd = np.abs(signal_fft)**2
    freq = np.linspace(0, fs, len(signal_fft))
    freq1 = np.linspace(0, fs, len(signal_psd))

    if fig_fft is None:
        fig_fft = plt.figure()
    plt.plot(freq, signal_fft, label="fft")

    if fig_psd is None:
        fig_psd = plt.figure()
    plt.plot(freq, signal_psd, label="PSD")

    return signal_fft, signal_psd, fig_fft, fig_psd


def dwt(approx, details, labels, level, sampling_freq, class_str=None):
    """
    Plot the results of a DWT transform.
    """

    fig, axis = plt.subplots(level+1, 1, figsize=(8, 8))
    fig.tight_layout()

    # plot the approximation
    for i, l in enumerate(labels):
        axis[0].plot(approx[i], label=l)
    axis[0].legend()
    if class_str is None:
        axis[0].set_title('DWT approximations (level={}, sampling-freq={}Hz)'.format(level, sampling_freq))
    else:
        axis[0].set_title('DWT approximations, {} (level={}, sampling-freq={}Hz)'.format(class_str, level, sampling_freq))
    axis[0].set_ylabel('(A={})'.format(level))

    # build the rows of detail coefficients
    for j in range (1,level+1):
        for i, l in enumerate(labels):
            axis[j].plot(details[i][j-1], label=l)
        if class_str is None:
            axis[j].set_title('DWT Coeffs (level{}, sampling-freq={}Hz)'.format(level, sampling_freq))
        else:
            axis[j].set_title('DWT Coeffs, {} (level={}, sampling-freq={}Hz)'.format(class_str, level, sampling_freq))
        axis[j].legend()
        axis[j].set_ylabel('(D={})'.format(j))

    return axis


def welch_psd(xs, labels, sampling_freq, fig=None):
    """Compute and plot the power spectrum density (PSD) using Welch's method.
    """

    fs = []
    ps = []
    for i, x in enumerate(xs):
        f, p = scipy.signal.welch(x, sampling_freq, 'flattop', scaling='spectrum')
        fs.append(f)
        ps.append(p)

    if fig is None:
        fig = plt.figure()

    plt.subplots_adjust(hspace=0.4)
    for i, p in enumerate(ps):
        plt.semilogy(f/8, p.T, label=labels[i])

    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()
    plt.show()

    return ps, fig



def artifact_removal(X, S, S_reconst, fig=None):
    """Plot the results of an artifact removal.

    This function displays the results after artifact removal, for instance
    performed via :func:`gumpy.signal.artifact_removal`.

    Parameters
    ----------
    X:
        Observations
    S:
        True sources
    S_reconst:
        The reconstructed signal
    """

    if fig is None:
        fig = plt.figure()

    models = [X, S, S_reconst]
    names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(3, 1, ii)
        plt.title(name)
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def PCA_2D(X, X_train, Y_train, colors=None):
    # computation
    pca_2comp = PCA(n_components=2)
    X_2comp = pca_2comp.fit_transform(X)

    # color and figure initialization
    if colors is None:
        colors = ['red','cyan']
    if fig is None:
        fig = plt.figure()

    # plotting
    fig.suptitle('2D - Data')
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X_train.T[0], X_train.T[1], alpha=0.5,
               c=Y_train, cmap=mpl.colors.ListedColormap(colors))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


def PCA_3D(X, X_train, Y_train, fig=None, colors=None):
    # computation
    pca_3comp = sklearn.decomposition.PCA(n_components=3)
    X_3comp = pca_3comp.fit_transform(X)

    # color and figure initialization
    if colors is None:
        colors = ['red','cyan']
    if fig is None:
        fig = plt.figure()

    # plotting
    fig.suptitle('3D - Data')
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(X_train.T[0], X_train.T[1], X_train.T[2], alpha=0.5,
               c=Y_train, cmap=mpl.colors.ListedColormap(colors))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')


# TODO: allow user to pass formatting control, e.g. colors, cmap, etc
def PCA(ttype, X, X_train, Y_train, fig=None, colors=None):
    plot_fns = {'2D': PCA_2D, '3D': PCA_3D}
    if not ttype in plot_fns:
        raise Exception("Transformation type '{ttype}' unknown".format(ttype=ttype))
    plot_fns[ttype](X, X_train, Y_train, fig, colors)



def EEG_bandwave_visualizer(data, band_wave, n_trial, lo, hi, fig=None):
    if not fig:
        fig = plt.figure()

    plt.clf()
    plt.plot(band_wave[data.trials[n_trial]-data.mi_interval[0]*data.sampling_freq : data.trials[n_trial]+data.mi_interval[0]*data.sampling_freq, 0],
            alpha=0.7, label='C3')
    plt.plot(band_wave[data.trials[n_trial]-data.mi_interval[0]*data.sampling_freq : data.trials[n_trial]+data.mi_interval[0]*data.sampling_freq, 1],
            alpha=0.7, label='C4')
    plt.plot(band_wave[data.trials[n_trial]-data.mi_interval[0]*data.sampling_freq : data.trials[n_trial]+data.mi_interval[0]*data.sampling_freq, 2],
            alpha=0.7, label='Cz')

    plt.legend()
    plt.title("Filtered data  (Band wave {}-{})".format(lo, hi))


# TODO: check if this is too specific
# TODO: documentation
# TODO: units missing
def average_power(data_class1, lowcut, highcut, interval, sampling_freq, logarithmic_power):
    fs = sampling_freq
    if logarithmic_power:
        power_c3_c1_a  = np.log(np.power(data_class1[0], 2).mean(axis=0))
        power_c4_c1_a  = np.log(np.power(data_class1[1], 2).mean(axis=0))
        power_cz_c1_a  = np.log(np.power(data_class1[2], 2).mean(axis=0))
        power_c3_c2_a  = np.log(np.power(data_class1[3], 2).mean(axis=0))
        power_c4_c2_a  = np.log(np.power(data_class1[4], 2).mean(axis=0))
        power_cz_c2_a  = np.log(np.power(data_class1[5], 2).mean(axis=0))
    else:
        power_c3_c1_a  = np.power(data_class1[0], 2).mean(axis=0)
        power_c4_c1_a  = np.power(data_class1[1], 2).mean(axis=0)
        power_cz_c1_a  = np.power(data_class1[2], 2).mean(axis=0)
        power_c3_c2_a  = np.power(data_class1[3], 2).mean(axis=0)
        power_c4_c2_a  = np.power(data_class1[4], 2).mean(axis=0)
        power_cz_c2_a  = np.power(data_class1[5], 2).mean(axis=0)

    # time indices
    t = np.linspace(interval[0],interval[1],len(power_c3_c1_a[fs*interval[0]:fs*interval[1]]))

    # first figure, left motor imagery
    plt.figure()
    plt.plot(t, power_c3_c1_a[fs*interval[0]:fs*interval[1]], c='blue',
                    label='C3', alpha=0.7)
    plt.plot(t,power_c4_c1_a [fs*interval[0]:fs*interval[1]],c='red',
                    label='C4', alpha=0.7)
    plt.legend()
    plt.xlabel('Time')
    if logarithmic_power:
        plt.ylabel('Logarithmic Power')
    else:
        plt.ylabel('Power')
    plt.title("Left motor imagery movements ".format(lowcut, highcut))
    plt.show()

    # second figure, right motor imagery
    plt.figure()
    plt.clf()
    plt.plot(t, power_c3_c2_a[fs*interval[0] : fs*interval[1]], c='blue', label='C3', alpha=0.7)
    plt.plot(t, power_c4_c2_a[fs*interval[0] : fs*interval[1]], c='red', label='C4', alpha=0.7)
    plt.legend()
    plt.xlabel('Time')
    if logarithmic_power:
        plt.ylabel('Logarithmic Power')
    else:
        plt.ylabel('Power')
    plt.title("Right motor imagery movements".format(lowcut, highcut))
