# -*- coding: utf-8 -*-

import datetime
import logging
from time import time

from sklearn.model_selection import GridSearchCV

from mltoolbox.base.logs import SetLogger


# TODO: Include other techniques like RandomizedSearchCV or BayesianOptimization

class MultiLearnerCV(SetLogger):
    """Compute the grid search cv for a set of estimators

    Parameters
    ----------
    models : dict
        Contains the label : estimator to be used

    params : dict
        Contains the parameters of the estimator to be optimized

    verbose : int
        Level of the logger.
            0 - No messages
            1 - Info
            2 - Debug

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.svm import SVC
    >>> from mltoolbox.model_selection.search import MultiLearnerCV
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data[:, 1:3], iris.target
    >>> model_params_test = {
    ...    'RFC': {'n_estimators': [8]},
    ...    'SVC': {}
    ... }
    >>> models_test = {
    ...    'RFC': RandomForestClassifier(random_state=1),
    ...    'SVC': SVC(random_state=1)
    ... }
    >>> mp = MultiLearnerCV(models=models_test, params=model_params_test)
    >>> mp.fit(X,y)
    >>> y_pred = mp.predict(X)
    >>> print("RFC: {0:.2f}".format(accuracy_score(y, y_pred['RFC'])))
    RFC: 0.99
    >>> print("SVC: {0:.2f}".format(accuracy_score(y, y_pred['SVC'])))
    SVC: 0.96

    """

    def __init__(self, models, params, verbose=0):
        SetLogger.__init__(self, verbose)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, method='gridsearch', cv_params=None):
        """ Optimize the hyper parameters.

         Look for the best estimator and its hyper parameters. It can be used to train multiple estimators on the
         same data, for classification or regression.

         Parameters
         ----------
         X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

         y : array-like, shape = [n_samples, n_features]
            Target values.

        method : string, [gridsearch (default)]
            Name of the method to use for the optimization.

        cv_params : dict
            Parameters to perform the  cv.

        Returns
        -------
            self : object
         """

        if self.models is None or len(self.models) == 0:
            raise AttributeError('Invalid `models` attribute, `models`'
                                 ' should be a dict of {key: models}')

        if self.params is None or len(self.params) == 0:
            raise AttributeError('Invalid `params` attribute, `params`'
                                 ' should be a dict of {key: parameters}')

        if len(self.models) != len(self.params):
            raise AttributeError('Number of models and params must be equal'
                                 '; got %d models, %d params'
                                 % (len(self.models), len(self.params)))

        if not set(self.models.keys()).issubset(set(self.params.keys())):
            missing_params = list(set(self.models.keys()) - set(self.params.keys()))
            raise ValueError("Some models are missing parameters: %s" % missing_params)

        methods = ('gridsearch')
        if method not in methods:
            raise ValueError('Method has to be one of ' +
                             str(methods))

        if cv_params is None:
            cv_params = {'cv': 3}

        # Start counting time
        t0 = time()
        logging.info('==================== BEGIN ====================')
        logging.info('Start running time: {} s'.format(datetime.datetime.now().strftime('%H:%M:%S')))

        logging.debug("X: {}, y: {}".format(X.shape, y.shape))

        for key in self.keys:
            logging.info("Running GridSearchCV for: {}".format(key))
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, **cv_params)
            gs.fit(X, y)
            self.grid_searches[key] = gs

        logging.info('Total running time: {} s'.format(round(time() - t0, 3)))
        logging.info('+++++++++++++++++++++ END +++++++++++++++++++++')
        logging.debug('Done.')

    def predict(self, X):
        """ Performs the prediction based on the best trained estimator.


         Parameters
         ----------
         X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_pred : dict {'estimator': array-like}
            The dictionary contains as key the name of the predictor and as value the prediction.
         """
        y_pred = {}
        for key in self.keys:
            y_pred[key] = self.grid_searches[key].predict(X)
        logging.debug('Done.')
        return y_pred

    def predict_proba(self, X):
        """ Performs the probability prediction based on the best trained estimator.


         Parameters
         ----------
         X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
                n_features is the number of features.

        Returns
        -------
        y_pred_proba : dict {'estimator': array-like}
            The dictionary contains as key the name of the predictor and and an array of the probabilities by class.
         """
        y_pred_proba = {}
        for key in self.keys:
            y_pred_proba[key] = self.grid_searches[key].predict_proba(X)
        logging.debug('Done.')
        return y_pred_proba

# if __name__ == '__main__':
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn import datasets
#     from sklearn.metrics import accuracy_score
#     from sklearn.svm import SVC
#
#     iris = datasets.load_iris()
#     X, y = iris.data[:, 1:3], iris.target
#
#     model_params_test = {
#         'RFC': {'n_estimators': [8]},
#         'SVC': {}
#     }
#     models_test = {
#         'RFC': RandomForestClassifier(random_state=1),
#         'SVC': SVC(random_state=1)
#     }
#     mp = MultiLearnerCV(models=models_test, params=model_params_test)
#     mp.fit(X,y)
#     y_pred = mp.predict(X)
#     print("RFC: {0:.2f}".format(accuracy_score(y, y_pred['RFC'])))
#     print("SVC: {0:.2f}".format(accuracy_score(y, y_pred['SVC'])))
