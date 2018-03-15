"""Implementations of common classifiers.

The implementations rely mostly on scikit-learn. They use default parameters
that were found to work on most datasets.
"""

# TODO: check consistency in variable naming
# TODO: implement unit tests

from .classifier import Classifier, ClassificationResult, register_classifier

# selectively import relevant sklearn classes. Prepend them with _ to avoid
# confusion with classes specified in this module
from sklearn.svm import SVC as _SVC
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier as _MLPClassifier
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis as _QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB as _GaussianNB
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier


# some 'good' default values across different classifiers


@register_classifier
class SVM(Classifier):
    """Support Vector Machine classifier for EEG data.

    """

    def __init__(self, **kwargs):
        """Initialize the SVM classifier.

        All keyword arguments that are not listed will be forwarded to the
        underlying classifier. In this case, it is sklearn.SVC. For instance,
        if you pass an argument ``probability=True``, this will be forwarded
        to the initialization of SVC.

        Keyword arguments
        -----------------
        max_iter: int, default = 1e6
            number of iterations during hyper-parameter tuning
        k_cross_val: int, default = 5
            number cross-validations (k-fold)
        cross_validation: Boolean, default = True
            Enable k-fold cross validation for hyper-parameter tuning. If False,
            the the SVM will use `probability=True` if not specified otherwise
            in kwargs.
        """
        super(SVM, self).__init__()

        # initialize some default values for the SVM backend
        self.max_iter = kwargs.pop('max_iter', 1e6)

        # parameters for k cross validation / hyper-parameter tuning
        self.params = [{
            'kernel': ['rbf', 'sigmoid', 'poly'],
            'C': [1e1, 1e2, 1e3, 1e4],
            'gamma': [1e4, 1e3, 1e2, 1, 1e-1, 1e-2],
            'degree': [2,3,4]}]
        self.k_cross_val = kwargs.pop('k_cross_val', 5)

        # initialize the classifier using grid search to find optimal parameters
        # via cross validation
        if kwargs.pop('cross_validation', True):
            self.clf = GridSearchCV(_SVC(max_iter=self.max_iter, **kwargs),
                                    self.params,
                                    cv=self.k_cross_val)
        else:
            probability = kwargs.pop('probability', True)
            self.clf = _SVC(max_iter=self.max_iter, probability=probability, **kwargs)


    @staticmethod
    def static_opts(ftype, **kwargs):
        """Returns default options for voting classification.

        This will avoid grid search during initialization.
        """
        return {'cross_validation': False}


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        result = ClassificationResult(Y_test, Y_pred)
        return result, self



@register_classifier
class KNN(Classifier):
    """
    """

    def __init__(self, **kwargs):
        """Initialize a K Nearest Neighbors (KNN) classifier.

        All additional keyword arguments will be forwarded to the underlying
        classifier, which is here ``sklearn.neighbors.KNeighborsClassifier``.

        Keyword Arguments
        -----------------
        n_neighbors: int, default 5
            Number of neighbors
        """

        super(KNN, self).__init__()
        self.nneighbors = kwargs.pop('n_neighbors', 5)
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=self.nneighbors, **kwargs)


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self



@register_classifier
class LDA(Classifier):
    """Linear Discriminant Analysis classifier.

    """

    def __init__(self, **kwargs):
        super(LDA, self).__init__()
        self.clf = _LinearDiscriminantAnalysis(**kwargs)


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self


@register_classifier
class Tree(Classifier):
    """Decision Tree 

    """

    def __init__(self, **kwargs):
        super(Tree, self).__init__()
        self.clf = _DecisionTreeClassifier(**kwargs)


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self
    

@register_classifier
class LogisticRegression(Classifier):
    """
    """

    def __init__(self, **kwargs):
        """Initialize a Logistic Regression Classifier.

        Additional keyword arguments will be passed to the classifier
        initialization which is ``sklearn.linear_model.LogisticRegression``
        here.

        Keyword Arguments
        -----------------
        C: int, default = 100
        """
        super(LogisticRegression, self).__init__()
        self.C = kwargs.pop("C", 100)
        self.clf = _LogisticRegression(C=self.C, **kwargs)

    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self



@register_classifier
class MLP(Classifier):
    """
    """

    def __init__(self, **kwargs):
        """This 'initializes' an MLP Classifier.

        If no further keyword arguments are passed, the initializer is not fully
        created and the MLP will only be constructed during `run`. If, however,
        the hidden layer size is specified, the MLP will be constructed fully.

        Keyword Arguments
        -----------------
        solver: default = ``lbfgs``
            The internal solver for weight optimization.
        alpha: default = ``1e-5``
            Regularization parameter.
        random_state: int or None
            Seed used to initialize the random number generator. default = 1,
            can be None.
        hidden_layer_sizes: tuple
            The sizes of the hidden layers.
        """

        super(MLP, self).__init__()

        # TODO: why lbfgs and not adam?
        self.solver = kwargs.pop('solver', 'lbfgs')
        self.alpha = kwargs.pop('alpha', 1e-5)
        self.random_state = kwargs.pop('random_state', 1)

        # determine if the MLP can be initialized or not
        self.clf = None
        self.hidden_layer_sizes = kwargs.pop('hidden_layer_sizes', -1)
        if not (self.hidden_layer_sizes == -1):
            self.initMLPClassifier(**kwargs)


    @staticmethod
    def static_opts(ftype, **kwargs):
        """Sets options that are required during voting and feature selection runs.

        """

        opts = dict()

        if ftype == 'sequential_feature_selector':
            # check if we got the features
            features = kwargs.pop('features', None)
            if features is not None:
                opts['hidden_layer_sizes'] = (features.shape[0], features.shape[1])

        if ftype == 'vote':
            # check if we got the training data
            X_train = kwargs.pop('X_train', None)
            if X_train is not None:
                # TODO: check dimensions!
                opts['hidden_layer_sizes'] = (X_train.shape[1], X_train.shape[1])

        return opts


    def initMLPClassifier(self, **kwargs):
        self.hidden_layer_sizes = kwargs.pop('hidden_layer_sizes', self.hidden_layer_sizes)
        self.clf = _MLPClassifier(solver=self.solver,
                                  alpha=self.alpha,
                                  hidden_layer_sizes=self.hidden_layer_sizes,
                                  random_state=self.random_state,
                                  **kwargs)


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        """Run the MLP classifier.

        In case the user did not specify layer sizes during
        initialization, the run method will automatically deduce
        the size from the input arguments.
        """
        if self.clf is None:
            self.hidden_layer_sizes = (X_train.shape[1], X_train.shape[1])
            self.initMLPClassifier(**kwargs)

        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self



@register_classifier
class NaiveBayes(Classifier):
    """
    """

    def __init__(self, **kwargs):
        super(NaiveBayes, self).__init__()
        self.clf = _GaussianNB(**kwargs)


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self



@register_classifier
class RandomForest(Classifier):
    """
    """

    def __init__(self, **kwargs):
        """Initialize a RandomForest classifier.

        All keyword arguments that are not listed will be forwarded to the
        underlying classifier. In this case, it is ``sklearn.esemble.RandomForestClassifier``.

        Keyword Arguments
        -----------------
        n_jobs: int, default = 4
            Number of jobs for the RandomForestClassifier
        k_cross_val: int, default = 5
            Number of cross-validations in hyper-parameter tuning.
        cross_validation: Boolean, default True
            Enable k-fold cross validation for hyper-parameter tuning. If set to
            false, the criterion will be `gini` and 10 estimators will be used
            if not specified otherwise in kwargs.
        """
        # TODO: document that all additional kwargs will be passed to the
        # RandomForestClassifier!

        super(RandomForest, self).__init__()

        self.n_jobs = kwargs.pop("n_jobs", 4)
        self.params = [{
            'n_estimators': [10, 100, 1000],
            'criterion': ['gini', 'entropy']}]
        self.k_cross_val = kwargs.pop('k_cross_val', 5)

        # initialize the classifier, which will be optimized using k cross
        # validation during fitting
        if kwargs.pop('cross_validation', True):
            self.clf = GridSearchCV(_RandomForestClassifier(n_jobs=self.n_jobs, **kwargs),
                                    self.params,
                                    cv=self.k_cross_val)
        else:
            # default arguments to use if not specified otherwise
            # TODO: move to static_opts?
            criterion = kwargs.pop('criterion', 'gini')
            n_estimators = kwargs.pop('n_estimators', 10)
            self.clf = _RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, n_jobs=self.n_jobs, **kwargs)


    @staticmethod
    def static_opts(ftype, **kwargs):
        """Returns default options for voting classification.

        This will avoid grid search during initialization.
        """
        return {'cross_validation': False}


    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        result = ClassificationResult(Y_test, Y_pred)
        return result, self



@register_classifier
class QuadraticLDA(Classifier):
    def __init__(self, **kwargs):
        super(QuadraticLDA, self).__init__()
        self.clf = _QuadraticDiscriminantAnalysis(**kwargs)

    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self



@register_classifier
class ShrinkingLDA(Classifier):
    def __init__(self, **kwargs):
        """Initializes a ShrinkingLDA classifier.

        Additional arguments will be forwarded to the underlying classifier
        instantiation, which is
        ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis`` here.

        Keyword Arguments
        -----------------
        solver: string, default = lsqr
            Solver used in LDA
        shrinkage: string, default = 'auto'

        """
        super(ShrinkingLDA, self).__init__()
        self.solver = kwargs.pop('solver', 'lsqr')
        self.shrinkage = kwargs.pop('shrinkage', 'auto')
        self.clf = _LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage, **kwargs)

    def run(self, X_train, Y_train, X_test, Y_test, **kwargs):
        self.clf.fit(X_train, Y_train.astype(int))
        Y_pred = self.clf.predict(X_test)
        return ClassificationResult(Y_test, Y_pred), self

