# -*- coding: utf-8 -*-

import datetime
import logging
import numpy as np
from time import time

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels

from mltoolbox.base.logs import SetLogger
from mltoolbox.metrics.classification_metrics import compute_classification_scores
from mltoolbox.model_selection.search import MultiLearnerCV
from mltoolbox.utils.format import format_results_table


# TODO: Tests with multiclass problem

class MultiClassifier(SetLogger):
    """Train multiple estimators with optimized parameters on a Out-of-Fold

    Multiple estimators can be trained on a specified configuration of folds. For each fold, the best estimator is
    found using cross validation. Once that the best estimator is computed, it is tested on the fold that was left out.
    The process is repeated n times depending on the configuration of folds. At the end, a report of the performance
    of each estimator is generated.

    Parameters
    ----------
    n_splits : int
        Number of folds

    shuffle : bool
        True if the data should be shuffle before the generation of the folds, false otherwise.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.

    verbose : int
        Level of the logger.
            0 - No messages
            1 - Info
            2 - Debug


    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from mltoolbox.model_selection.classification import MultiClassifier
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data[:, 1:4], iris.target
    >>> models = {
    ...    'SVC': SVC(random_state=1),
    ...     'RandomForestClassifier': RandomForestClassifier(random_state=1)
    ... }
    >>> model_params = {
    ...    'SVC': {},
    ...    'RandomForestClassifier': {'n_estimators': [8]}
    ... }
    >>> y_binary_idx = np.where(y != 2)
    >>> mc = MultiClassifier()
    >>> mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)
    >>> print(mc.report_score_summary_by_classifier('RandomForestClassifier'))
                   Accuracy Specificity  Precision     Recall   F1-score        AUC
    <BLANKLINE>
              1      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              2      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              3      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              4      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              5      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
    <BLANKLINE>
        Average      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
    <BLANKLINE>
    >>> print(mc.report_score_summary_by_classifier('SVC'))
                   Accuracy Specificity  Precision     Recall   F1-score        AUC
    <BLANKLINE>
              1      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              2      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              3      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              4      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
              5      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
    <BLANKLINE>
        Average      1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
    <BLANKLINE>
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None, verbose=0):
        SetLogger.__init__(self, verbose)
        self.verbose = verbose
        # Divide the data in n-folds
        self.__kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.__fold_models = {}
        self.__fold_pred = {}
        self.__fold_pred_proba = {}
        self.__fold_test_index = {}
        self.__fold_train_index = {}
        self._X = []
        self._y = []
        self.__n_labels = 0
        self.__proba = True

    def train(self, X, y, models, model_params, method='gridsearch', cv_params=None):
        """Train each model with the optimization of its hyper parameters using k-folds

        Parameters
        ----------
         X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

         y : array-like, shape = [n_samples, n_features]
            Target values.

        models : dict
            Contains the estimators

        model_params : dict
            Contains the parameters to be optimized for each estimator in `models`

        method : string
            The method to be used to perform the cross validation to find the best hyper parameters

        cv_params : dict
            Contains the parameters to perform the cross validation, those parameters depend on the `method`.

        """

        # Start counting time
        t0 = time()
        logging.info('==================== BEGIN ====================')
        logging.info('Start running time: {} s'.format(datetime.datetime.now().strftime('%H:%M:%S')))

        self._X = X
        self._y = y
        self.__n_labels = unique_labels(y, y).size

        for index, (train_index, _) in enumerate(self.__kf.split(self._X, self._y)):
            logging.info("{}-fold".format(index + 1))
            X_train = self._X[train_index]
            y_train = self._y[train_index]

            mp = MultiLearnerCV(models, model_params, verbose=1)
            mp.fit(X_train, y_train, method, cv_params)
            self.__fold_models[index] = mp

        logging.info('Total running time: {} s'.format(round(time() - t0, 3)))
        logging.info('+++++++++++++++++++++ END +++++++++++++++++++++')

    def predict(self):
        """Performs the prediction of each estimator on the k-folds

        Returns
        -------
        y_pred : dict
            Contains the prediction of each estimator on the k-folds
        """
        self.__proba = False
        self.predict_proba()
        logging.debug('Done.')
        return self.__fold_pred

    def predict_proba(self):
        """Performs the prediction based on probabilities of each estimator on the k-folds

        Returns
        -------
        y_pred_proba : dict
            Contains the prediction of each estimator on the k-folds
        """

        for index, (train_index, test_index) in enumerate(self.__kf.split(self._X, self._y)):
            X_test = self._X[test_index]

            self.__fold_pred[index] = self.__fold_models[index].predict(X_test)
            if self.__proba:
                self.__fold_pred_proba[index] = self.__fold_models[index].predict_proba(X_test)
            self.__fold_test_index[index] = test_index
            self.__fold_train_index[index] = train_index

        logging.debug('Done.')
        return self.__fold_pred_proba

    def score_summary_by_classifier(self, classifier_name):
        """Compute the scores the classifier n the k-folds.

        Parameters
        ----------
        classifier_name : string
            Key used in the dict `models` to point to an estimator.

        Returns
        -------
        results : {array-like}, shape = [len(n_folds), n_scores]
            If the classification is binary, an array with the scores `Accuracy`,
            `Precision`, `Recall`, `F1-score`, `AUC` is returned.
            If the classification is multi, an array with the scores `Accuracy`,
            `Precision`, `Recall`, `F1-score` is returned.
            An extra row is included, it contains the average of every scores.
        """

        # If the prediction hasn't been computed
        if {} == self.__fold_pred:
            self.predict()

        if self.__n_labels == 2:
            results = np.zeros([len(self.__fold_pred), 6])
        else:
            results = np.zeros([len(self.__fold_pred), 4])

        for fold_key in range(len(self.__fold_pred)):
            y_true = self._y[self.__fold_test_index[fold_key]]
            y_pred = self.__fold_pred[fold_key][classifier_name]
            results[fold_key, :] = np.asarray(compute_classification_scores(y_true, y_pred, self.verbose))
        logging.debug('Done.')
        return results

    def report_score_summary_by_classifier(self, classifier_name):
        """Generate a formatted version of the summary

        Parameters
        ----------
        classifier_name : string
            Key used in the dict `models` to point to an estimator.

        Returns
        -------
        results_formatted : string
            Formatted version of the results computed in score_summary_by_classifier().

        """
        results = self.score_summary_by_classifier(classifier_name)
        logging.debug(results)

        if self.__n_labels == 2:
            header_names = ['Accuracy', 'Specificity', 'Precision', 'Recall', 'F1-score', 'AUC']
        else:
            header_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

        results_formatted = format_results_table(results, header_names, row_names=None, operation='average', digits=4)
        logging.debug('Done.')
        return results_formatted

    def best_estimator(self, classifier_name, fold_key=None):
        """Returns the data used to train and test the estimator and its predictions on a particular fold.

        Parameters
        ----------
        classifier_nam : string
            Key used in the dict `models` to point to an estimator.

        fold_key : int
            Key of one of the folds.

        Returns
        -------
        fold_key : int
            Key of one of the folds.

        bm_model : object
            Estimator

        bm_y_pred : {array-like}, shape = [n_samples, ]
            Predicted labels.

        bm_train_indices : {array-like}, shape = [n_samples, ]
            Contains the indices of the samples used to train the model.

        bm_test_indices : {array-like}, shape = [n_samples, ]
            Contains the indices of the samples used to test the model.
        """
        results = self.score_summary_by_classifier(classifier_name)

        best_model = {}

        if (fold_key is None) or (fold_key - 1 not in range(0, results[:, 0].shape[0])):
            # Get the highest accuracy
            fold_key = results[:, 0].argmax()
        else:
            fold_key = fold_key - 1

        # Get the test and training indices of the data
        bm_test_indices = self.__fold_test_index[fold_key]
        bm_train_indices = self.__fold_train_index[fold_key]

        # Get the best __models
        bm_model = self.__fold_models[fold_key].grid_searches_[classifier_name].best_estimator_

        # Get the prediction
        bm_y_pred = self.__fold_pred[fold_key][classifier_name]

        best_model[classifier_name] = [fold_key + 1, bm_model, bm_y_pred, bm_train_indices, bm_test_indices]

        logging.debug('Done.')
        return best_model

    def feature_importances(self, classifier_name, fold_key=None):
        """

        Parameters
        ----------
        classifier_name : string
            Key used in the dict `models` to point to an estimator.

        fold_key : int
            Key of one of the folds.

        Returns
        -------
        feature_importances : {array-like}
            Scores of the features.

        """
        best_model = self.best_estimator(classifier_name, fold_key)
        return best_model[classifier_name][1].feature_importances_

    # if __name__ == '__main__':
    #     from sklearn import datasets
    #     from sklearn.ensemble import RandomForestClassifier
    #     from sklearn.svm import SVC
    #
    #     iris = datasets.load_iris()
    #     X, y = iris.data, iris.target
    #
    #     from sklearn import preprocessing
    #
    #     std_scale = preprocessing.StandardScaler().fit(X)
    #     X_std = std_scale.transform(X)
    #     print('After standardization:{:.4f}, {:.4f}'.format(X_std.mean(), X_std.std()))
    #
    #
    #     random_state = 2017  # seed used by the random number generator
    #
    #     models = {
    #         # NOTE: SVC and RFC are the names that will be used to make reference to the models after the training step.
    #         'SVC': SVC(probability=True,
    #                    random_state=random_state),
    #         'RFC': RandomForestClassifier(random_state=random_state)
    #     }
    #
    #     model_params = {
    #         'SVC': {'kernel': ['linear']},
    #         'RFC': {'n_estimators': [25]}
    #     }
    #
    #     cv_params = {
    #         'cv': StratifiedKFold(n_splits=3, shuffle=False, random_state=random_state)
    #     }
    #
    #     # Training
    #     mc = MultiClassifier(n_splits=5, shuffle=True, random_state=random_state, verbose=0)
    #     mc.train(X, y, models, model_params, cv_params=cv_params)
    #     print('RFC\n{}\n'.format(mc.report_score_summary_by_classifier('RFC')))
    #
    #     # Get the results of the parition that has the high accuracy
    #
    #     fold, bm_model, bm_y_pred, bm_train_indices, bm_test_indices = mc.best_estimator('RFC')['RFC']
    #
    #     print(">>Best model in fold: {}".format(fold))
    #     print(">>>Trained model \n{}".format(bm_model))
    #     print(">>>Predicted labels: \n{}".format(bm_y_pred))
    #     print(">>>Indices of the samples used for training: \n{}".format(bm_train_indices))
    #     print(">>>Indices of samples used for predicting: \n{}".format(bm_test_indices))


    # # Compute the best_estimator without previous predict
    # bs = mc.best_estimator('RandomForestClassifier', 4)['RandomForestClassifier']
    #
    # print(mc.feature_importances('RandomForestClassifier'))
    #
    # #assert_equal(bs[0], 4)
    #
    #
    # print(mc.report_score_summary_by_classifier('RandomForestClassifier'))
    #
    # text_file = open("tests/test_classification_report_files/report_binary.txt", "w")
    # text_file.write(report)
    # text_file.close()
