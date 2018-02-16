from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np

class ClassifierError(Exception):
    pass


class Classifier(ABC):
    """
    Abstract base class representing a classifier.

    All classifiers should subclass from this baseclass. All subclasses need to
    implement `run()`, which will be called for the classification. Additional
    arguments to the initialization should be captured via `**kwargs`. For an
    example, see the SVM classifier.

    In case a classifier auto-tunes its hyperparameters (for instance with the
    help of a grid search) but should avoid this behavior during voting
    classification or feature selection, a set of static options should be
    obtainable in form of a key-value dictionary using the ``static_opts``
    member function which will subsequently be passed to ``__init__``. Note that
    this function has to be defined with the staticmethod decorator. The
    Classifier provides an empty static_opts implementation. For an example of a
    customization, see the SVM classifier which should not perform grid search
    during voting classification or feature selection.

    """

    def __init__(self):
        pass


    @staticmethod
    def static_opts(ftype, **kwargs):
        """Return a kwargs dict for voting classification or feature computation.

        For more information see the documentation of the Classifier class. For additional
        information about the passed keyword arguments see the corresponding documentation
        in
            - ``gumpy.classification.classifier.vote``
            - ``gumpy.features.sequential_feature_selector``

        Args:
            ftype (str): function type for which the options are requested.
                One of the following: 'vote', 'sequential_feature_selector'
            **kwargs (dict): Additional arguments, depends on the function type

        Returns:
            A kwargs dictionary that can be passed to ``__init__``
        """
        return {}


    @abstractmethod
    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        """
        Run a classification.

        Args:
            self: reference to object
            X_train: training data (values)
            Y_train: training data (labels)
            X_test: evaluation data (values)
            Y_test: evaluation data (labels)
            **kwargs: Any additional arguments that may be passed to a classifier

        Returns:
            2-element tuple containing

            - **ClassificationResult**: Object with all the classification results
            - **Classifier**: Reference to the classifier

        """
        return None, self


    def __call__(self, X_train, Y_train, X_test, Y_test, **kwargs):
        return self.run(X_train, Y_train, X_test, Y_test)


class ClassificationResult:
    """
    The result of a classification run.

    The result includes the accuracy of the classification, a reference to the y
    data, as well as the prediction.

    """

    def __init__(self, test, pred):
        self.test = test
        self.pred = pred
        self.n_correct = len(np.where(test - pred == 0)[0])
        self.accuracy = (self.n_correct / len(pred)) * 100.0
        self.report = classification_report(self.test, self.pred)

    def __str__(self):
        return self.report



# list of known classifiers.
available_classifiers = {}


def register_classifier(cls):
    """Automatically register a class in the classifiers dictionary.

    This function should be used as decorator.

    Args:
        cls: subclass of `gumpy.classification.Classifier` that should be
            registered to gumpy.

    Returns:
        The class that was passed as argument

    Raises:
        ClassifierError: This error will be raised when a classifier is
            registered with a name that is already used.

    """
    if cls.__name__ in available_classifiers:
        raise ClassifierError("Classifier {name} already exists in available_classifiers".format(name=cls.__name__))

    available_classifiers[cls.__name__] = cls
    return cls



def classify(c, *args, **kwargs):
    """Classify EEG data given a certain classifier.

    The classifier can be specified by a string or be passed as an object. The
    latter option is useful if a classifier has to be called repeatedly, but the
    instantiation is computationally expensive.

    Additional arguments for the classifier instantiation can be passed in
    kwargs as a dictionary with name `opts`. They will be forwarded to the
    classifier on construction. If the classifier was passed as object, this
    will be ignored.

    Args:
        c (str or object): The classifier. Either specified by the classifier
            name, or passed as object
        X_train: training data (values)
        Y_train: training data (labels)
        X_test: evaluation data (values)
        Y_test: evaluation data (labels)
        **kwargs: additional arguments that may be passed on to the classifier. If the
            classifier is selected via string/name, you can pass options to the
            classifier by a dict with the name `opts`, i.e. `classify('SVM',
            opts={'a': 1})`.

    Returns:
        2-element tuple containing

        - **ClassificationResult**: The result of the classification.
        - **Classifier**:  The classifier that was used during the classification.

    Raises:
        ClassifierError: If the classifier is unknown or classification fails.

    Examples:
        >>> import gumpy
        >>> result, clf = gumpy.classify("SVM", X_train, Y_train, X_test, Y_test)

    """

    if isinstance(c, str):
        if not (c in available_classifiers):
            raise ClassifierError("Unknown classifier {c}".format(c=c.__repr__()))

        # instantiate the classifier
        opts = kwargs.pop('opts', None)
        if opts is not None:
            clf = available_classifiers[c](**opts)
        else:
            clf = available_classifiers[c]()
        return clf.run(*args, **kwargs)

    elif isinstance(c, Classifier):
        return c.run(*args, **kwargs)

    # invalid argument passed to the function
    raise ClassifierError("Unknown classifier {c}".format(c=c.__repr__()))



def vote(X_train, Y_train, X_test, Y_test, voting_type, feature_selection, k_features):
    """Invokation of a soft voting/majority rule classification.

    This is a wrapper around `sklearn.ensemble.VotingClassifier` which
    automatically uses all classifiers that are known to `gumpy` in
    `gumpy.classification.available_classifiers`.

    Args:
        X_train: training data (values)
        Y_train: training data (labels)
        X_test: evaluation data (values)
        Y_test: evaluation data (labels)
        voting_type (str): either of 'soft' or 'hard'. See the
            sklearn.ensemble.VotingClassifier documentation for more details

    Returns:
        2-element tuple containing

        - **ClassificationResult**: The result of the classification.
        - **Classifier**:  The instance of `sklearn.ensemble.VotingClassifier`
          that was used during the classification.

    """

    k_cross_val = 10
    N_JOBS=-1

    clfs = []
    for classifier in available_classifiers:
        # determine kwargs such that the classifiers get initialized with
        # proper default settings. This avoids cross-validation, for instance
        opts = available_classifiers[classifier].static_opts('vote', X_train=X_train)

        # retrieve instance
        cobj = available_classifiers[classifier](**opts)
        clfs.append((classifier, cobj.clf))

    # instantiate the VotingClassifier
    soft_vote_clf = VotingClassifier(estimators=clfs, voting=voting_type)

    if feature_selection:
        sfs = SFS(soft_vote_clf,
                  k_features,
                  forward=True,
                  floating=True,
                  verbose=2,
                  scoring='accuracy',
                  cv=k_cross_val,
                  n_jobs=N_JOBS)
        sfs = sfs.fit(X_train, Y_train)
        X_train = sfs.transform(X_train)
        X_test = sfs.transform(X_test)

    soft_vote_clf.fit(X_train, Y_train)
    Y_pred = soft_vote_clf.predict(X_test)
    return ClassificationResult(Y_test, Y_pred), soft_vote_clf



# TODO: what to do with this old code? adopt it similar to `vote` above?
# def cross_validation_classification (classifier,X, y, k):
#
#
#     k_cross_val = 10
#     N_JOBS=4
#     if classifier == "SVM":
#         parameters_svm = [{'kernel': ['rbf', 'sigmoid', 'poly'],
#                        'C': [1e1, 1e2, 1e3, 1e4],
#                        'gamma': [1e4, 1e3, 1e2, 1, 1e-1, 1e-2],
#                        'degree': [2,3,4]}]
#         clf = GridSearchCV(svm.SVC(max_iter=1e6),
#                                parameters_svm, cv=k_cross_val)
#
#     elif classifier == "LDA":
#
#         from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#         clf  = LinearDiscriminantAnalysis()
#
#     elif classifier == "Random Forest":
#
#         parameters_rf = [{'n_estimators': [10, 100, 1000],
#                       'criterion': ['gini', 'entropy']}]
#         clf = GridSearchCV(RandomForestClassifier(n_jobs=N_JOBS),
#                                parameters_rf, cv=k_cross_val)
#
#     elif classifier == "Naive Bayes": # Without Feed Back
#         clf = GaussianNB()
#
#     elif classifier == "KNN": # Without Feed Back
#         from sklearn import neighbors
#         clf = neighbors.KNeighborsClassifier(n_neighbors=5)
#
#     elif classifier == "Logistic regression": # Without Feed Back
#         from sklearn.linear_model import LogisticRegression
#         clf =  LogisticRegression(C=100)
#
#     elif classifier == "MLP":
#         from sklearn.neural_network import MLPClassifier
#
#         clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(X.shape[1],X.shape[1]), random_state=1)
#
#     elif classifier == "LDA_with_shrinkage":
#         from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
#
#
#
#     kfold = KFold(n_splits = k, random_state = 777)
#
#     results = cross_val_score(clf, X, y, cv = kfold)
#
#     # VISULALISATION
#
#
#     print('Accuracy Score')
#     print('Avearge %: ', results.mean()*100)
#     print('Standard deviation: ', results.std())
