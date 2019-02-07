

from .classification import available_classifiers
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np
import scipy.linalg as la
import pywt


def sequential_feature_selector(features, labels, classifier, k_features, kfold, selection_type, plot=True, **kwargs):
    """Sequential feature selection to reduce the number of features.

    The function reduces a d-dimensional feature space to a k-dimensional
    feature space by sequential feature selection. The features are selected
    using ``mlxtend.feature_selection.SequentialFeatureSelection`` which
    essentially selects or removes a feature from the d-dimensional input space
    until the preferred size is reached.

    The function will pass ``ftype='feature'`` and forward ``features`` on to a
    classifier's ``static_opts`` method.

    Args:
        features: The original d-dimensional feature space
        labels: corresponding labels
        classifier (str or object): The classifier which should be used for
            feature selection. This can be either a string (name of a classifier
            known to gumpy) or an instance of a classifier which adheres
            to the sklearn classifier interface.
        k_features (int): Number of features to select
        kfold (int): k-fold cross validation
        selection_type (str): One of ``SFS`` (Sequential Forward Selection),
            ``SBS`` (Sequential Backward Selection), ``SFFS`` (Sequential Forward
            Floating Selection), ``SBFS`` (Sequential Backward Floating Selection)
        plot (bool): Plot the results of the dimensinality reduction
        **kwargs: Additional keyword arguments that will be passed to the
            Classifier instantiation

    Returns:
        A 3-element tuple containing

        - **feature index**: Index of features in the remaining set
        - **cv_scores**: cross validation scores during classification
        - **algorithm**: Algorithm that was used for search

    """

    # retrieve the appropriate classifier
    if isinstance(classifier, str):
        if not (classifier in available_classifiers):
            raise ClassifierError("Unknown classifier {c}".format(c=classifier.__repr__()))

        kwopts = kwargs.pop('opts', dict())
        # opts = dict()

        # retrieve the options that we need to forward to the classifier
        # TODO: should we forward all arguments to sequential_feature_selector ?
        opts = available_classifiers[classifier].static_opts('sequential_feature_selector', features=features)
        opts.update(kwopts)

        # XXX: now merged into the static_opts invocation. TODO: test
        # if classifier == 'SVM':
        #     opts['cross_validation'] = kwopts.pop('cross_validation', False)
        # elif classifier == 'RandomForest':
        #     opts['cross_validation'] = kwopts.pop('cross_validation', False)
        # elif classifier == 'MLP':
        #     # TODO: check if the dimensions are correct here
        #     opts['hidden_layer_sizes'] = (features.shape[1], features.shape[2])
        # get all additional entries for the options
        # opts.update(kwopts)

        # retrieve a classifier object
        classifier_obj = available_classifiers[classifier](**opts)

        # extract the backend classifier
        clf = classifier_obj.clf
    else:
        # if we received a classifier object we'll just use this one
        clf = classifier.clf


    if selection_type == 'SFS':
        algorithm = "Sequential Forward Selection (SFS)"
        sfs = SFS(clf, k_features, forward=True, floating=False,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    elif selection_type == 'SBS':
        algorithm = "Sequential Backward Selection (SBS)"
        sfs = SFS(clf, k_features, forward=False, floating=False,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    elif selection_type == 'SFFS':
        algorithm = "Sequential Forward Floating Selection (SFFS)"
        sfs = SFS(clf, k_features, forward=True, floating=True,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    elif selection_type == 'SBFS':
        algorithm = "Sequential Backward Floating Selection (SFFS)"
        sfs = SFS(clf, k_features, forward=True, floating=True,
                verbose=2, scoring='accuracy', cv=kfold, n_jobs=-1)

    else:
        raise Exception("Unknown selection type '{}'".format(selection_type))


    pipe = make_pipeline(StandardScaler(), sfs)
    pipe.fit(features, labels)
    subsets = sfs.subsets_
    feature_idx = sfs.k_feature_idx_
    cv_scores = sfs.k_score_

    if plot:
        fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
        plt.ylim([0.5, 1])
        plt.title(algorithm)
        plt.grid()
        plt.show()

    return feature_idx, cv_scores, algorithm, sfs, clf


# TODO: improve description of argument. I have no clue what exactly I should
# pass to the function!
def CSP(tasks):
    """This function extracts Common Spatial Pattern (CSP) features.

    Args:
        For N tasks, N arrays are passed to CSP each with dimensionality (# of
        trials of task N) x (feature vector)

    Returns:
        A 2D CSP features matrix.

    """
    if len(tasks) < 2:
        print("Must have at least 2 tasks for filtering.")
        return (None,) * len(tasks)
    else:
        filters = ()
        # CSP algorithm
        # For each task x, find the mean variance matrices Rx and not_Rx, which will be used to compute spatial filter SFx
        iterator = range(0,len(tasks))
        for x in iterator:
            # Find Rx
            Rx = covarianceMatrix(tasks[x][0])
            for t in range(1,len(tasks[x])):
                Rx += covarianceMatrix(tasks[x][t])
            Rx = Rx / len(tasks[x])

            # Find not_Rx
            count = 0
            not_Rx = Rx * 0
            for not_x in [element for element in iterator if element != x]:
                for t in range(0,len(tasks[not_x])):
                    not_Rx += covarianceMatrix(tasks[not_x][t])
                    count += 1
            not_Rx = not_Rx / count

            # Find the spatial filter SFx
            SFx = spatialFilter(Rx,not_Rx)
            filters += (SFx,)

            # Special case: only two tasks, no need to compute any more mean variances
            if len(tasks) == 2:
                filters += (spatialFilter(not_Rx,Rx),)
                break
        return filters


# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    """This function computes the covariance Matrix

    Args:
        A: 2D matrix

    Returns:
        A 2D covariance matrix scaled by the variance
    """
    #Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
    Ca = np.cov(A)
    return Ca


def spatialFilter(Ra,Rb):
    """This function extracts spatial filters

    Args:
        Ra, Rb: Covariance matrices Ra and Rb

    Returns:
        A 2D spatial filter matrix
    """

    R = Ra + Rb
    E,U = la.eig(R)

    # CSP requires the eigenvalues E and eigenvector U be sorted in descending order
    ord = np.argsort(E)
    ord = ord[::-1] # argsort gives ascending order, flip to get descending
    E = E[ord]
    U = U[:,ord]

    # Find the whitening transformation matrix
    P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

    # The mean covariance matrices may now be transformed
    Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
    Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

    # Find and sort the generalized eigenvalues and eigenvector
    E1,U1 = la.eig(Sa,Sb)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:,ord1]

    # The projection matrix (the spatial filter) may now be obtained
    SFa = np.dot(np.transpose(U1),P)
    #return SFa.astype(np.float32)
    return SFa


def PCA_dim_red(features, var_desired):
    """Dimensionality reduction of features using PCA.

    Args:
        features (matrix (2d np.array)): The feature matrix
        var_desired (float): desired preserved variance

    Returns:
        features with reduced dimensions

    """
    # PCA
    pca = sklearn.decomposition.PCA(n_components=features.shape[1]-1)
    pca.fit(features)
    # print('pca.explained_variance_ratio_:\n',pca.explained_variance_ratio_)
    var_sum = pca.explained_variance_ratio_.sum()
    var = 0
    for n, v in enumerate(pca.explained_variance_ratio_):
        var += v
        if var / var_sum >= var_desired:
            features_reduced = sklearn.decomposition.PCA(n_components=n+1).fit_transform(features)
            return features_reduced


def RMS_features_extraction(data, trial_list, window_size, window_shift):
    """Extract RMS features from data

    Args:
        data: 2D (time points, Channels)
        trial_list: list of the trials
        window_size: Size of the window for extracting features
        window_shift: size of the overalp

    Returns:
        The features matrix (trials, features)
    """
    if window_shift > window_size:
        raise ValueError("window_shift > window_size")

    fs = data.sampling_freq

    n_features = int(data.duration/(window_size-window_shift))

    X = np.zeros((len(trial_list), n_features*4))

    t = 0
    for trial in trial_list:
        # x3 is the worst of all with 43.3% average performance
        x1=gumpy.signal.rms(trial[0], fs, window_size, window_shift)
        x2=gumpy.signal.rms(trial[1], fs, window_size, window_shift)
        x3=gumpy.signal.rms(trial[2], fs, window_size, window_shift)
        x4=gumpy.signal.rms(trial[3], fs, window_size, window_shift)
        x=np.concatenate((x1, x2, x3, x4))
        X[t, :] = np.array([x])
        t += 1
    return X


def dwt_features(data, trials, level, sampling_freq, w, n, wavelet):
    """Extract discrete wavelet features

    Args:
        data: 2D (time points, Channels)
        trials: Trials vector
        lLevel: level of DWT decomposition
        sampling_freq: Sampling frequency

    Returns:
        The features matrix (Nbre trials, Nbre features)
    """

    # number of features per trial
    n_features = 9
    # allocate memory to store the features
    X = np.zeros((len(trials), n_features))

    # Extract Features
    for t, trial in enumerate(trials):
        signals = data[trial + fs*4 + (w[0]) : trial + fs*4 + (w[1])]
        coeffs_c3 = pywt.wavedec(data = signals[:,0], wavelet=wavelet, level=level)
        coeffs_c4 = pywt.wavedec(data = signals[:,1], wavelet=wavelet, level=level)
        coeffs_cz = pywt.wavedec(data = signals[:,2], wavelet=wavelet, level=level)

        X[t, :] = np.array([
            np.std(coeffs_c3[n]), np.mean(coeffs_c3[n]**2),
            np.std(coeffs_c4[n]), np.mean(coeffs_c4[n]**2),
            np.std(coeffs_cz[n]), np.mean(coeffs_cz[n]**2),
            np.mean(coeffs_c3[n]),
            np.mean(coeffs_c4[n]),
            np.mean(coeffs_cz[n])])

    return X


def alpha_subBP_features(data):
    """Extract alpha bands

    Args:
        data: 2D (time points, Channels)

    Returns:
        The alpha sub-bands
    """
    # filter data in sub-bands by specification of low- and high-cut frequencies
    alpha1 = gumpy.signal.butter_bandpass(data, 8.5, 11.5, order=6)
    alpha2 = gumpy.signal.butter_bandpass(data, 9.0, 12.5, order=6)
    alpha3 = gumpy.signal.butter_bandpass(data, 9.5, 11.5, order=6)
    alpha4 = gumpy.signal.butter_bandpass(data, 8.0, 10.5, order=6)

    # return a list of sub-bands
    return [alpha1, alpha2, alpha3, alpha4]


def beta_subBP_features(data):
    """Extract beta bands

    Args:
        data: 2D (time points, Channels)

    Returns:
        The beta sub-bands
    """
    beta1 = gumpy.signal.butter_bandpass(data, 14.0, 30.0, order=6)
    beta2 = gumpy.signal.butter_bandpass(data, 16.0, 17.0, order=6)
    beta3 = gumpy.signal.butter_bandpass(data, 17.0, 18.0, order=6)
    beta4 = gumpy.signal.butter_bandpass(data, 18.0, 19.0, order=6)
    return [beta1, beta2, beta3, beta4]


def powermean(data, trial, fs, w):
    """Compute the mean power of the data

    Args:
        data: 2D (time points, Channels)
        trial: trial vector
        fs: sampling frequency
        w: window

    Returns:
        The mean power
    """
    return np.power(data[trial+fs*4+w[0]: trial+fs*4+w[1],0],2).mean(), \
           np.power(data[trial+fs*4+w[0]: trial+fs*4+w[1],1],2).mean(), \
           np.power(data[trial+fs*4+w[0]: trial+fs*4+w[1],2],2).mean()


def log_subBP_feature_extraction(alpha, beta, trials, fs, w):
    """Extract the log power of alpha and beta bands

    Args:
        alpha: filtered data in the alpha band
        beta: filtered data in the beta band
        trials: trial vector
        fs: sampling frequency
        w: window

    Returns:
        The features matrix
    """
    # number of features combined for all trials
    n_features = 15
    # initialize the feature matrix
    X = np.zeros((len(trials), n_features))

    # Extract features
    for t, trial in enumerate(trials):
        power_c31, power_c41, power_cz1 = powermean(alpha[0], trial, fs, w)
        power_c32, power_c42, power_cz2 = powermean(alpha[1], trial, fs, w)
        power_c33, power_c43, power_cz3 = powermean(alpha[2], trial, fs, w)
        power_c34, power_c44, power_cz4 = powermean(alpha[3], trial, fs, w)
        power_c31_b, power_c41_b, power_cz1_b = powermean(beta[0], trial, fs, w)

        X[t, :] = np.array(
            [np.log(power_c31), np.log(power_c41), np.log(power_cz1),
             np.log(power_c32), np.log(power_c42), np.log(power_cz2),
             np.log(power_c33), np.log(power_c43), np.log(power_cz3),
             np.log(power_c34), np.log(power_c44), np.log(power_cz4),
             np.log(power_c31_b), np.log(power_c41_b), np.log(power_cz1_b)])

    return X


