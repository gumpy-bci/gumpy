

from .classification import available_classifiers
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


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
        sfs = SFS(clf, k_features, forward=False, floating=True,
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

    return feature_idx, cv_scores, algorithm




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


