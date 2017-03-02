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


# TODO: Documentation

class MultiClassifier(SetLogger):
    def __init__(self, n_splits=5, shuffle=False, random_state=2016, verbose=0):
        SetLogger.__init__(self, verbose)
        # Divide the data in 10-folds
        # self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.fold_models = {}
        self.fold_pred = {}
        self.fold_pred_proba = {}
        self.fold_test_index = {}
        self.fold_train_index = {}
        self.__score_results_ = {}
        self.X = []
        self.y = []
        self.n_labels = 0
        self.__proba = True

    def train(self, X, y, models, model_params, method='gridsearch', gs_params=None):

        # Start counting time
        t0 = time()
        logging.info('==================== BEGIN ====================')
        logging.info('Start running time: {} s'.format(datetime.datetime.now().strftime('%H:%M:%S')))

        self.X = X
        self.y = y
        self.n_labels = unique_labels(y, y).size

        for index, (train_index, _) in enumerate(self.kf.split(self.X, self.y)):
            logging.info("{}-fold".format(index + 1))
            X_train = self.X[train_index]
            y_train = self.y[train_index]

            mp = MultiLearnerCV(models, model_params, verbose=1)
            mp.fit(X_train, y_train, method, gs_params)
            self.fold_models[index] = mp

        logging.info('Total running time: {} s'.format(round(time() - t0, 3)))
        logging.info('+++++++++++++++++++++ END +++++++++++++++++++++')

    def predict(self):
        self.__proba = False
        self.predict_proba()
        logging.debug('Done.')
        return self.fold_pred

    def predict_proba(self):

        for index, (train_index, test_index) in enumerate(self.kf.split(self.X, self.y)):
            X_test = self.X[test_index]

            self.fold_pred[index] = self.fold_models[index].predict(X_test)
            if self.__proba:
                self.fold_pred_proba[index] = self.fold_models[index].predict_proba(X_test)
            self.fold_test_index[index] = test_index
            self.fold_train_index[index] = train_index

        logging.debug('Done.')
        return self.fold_pred_proba

    def score_summary_by_classifier(self, classifier_name):

        # If the prediction hasn't been computed
        if {} == self.fold_pred:
            self.predict()

        if self.n_labels == 2:
            results = np.zeros([len(self.fold_pred), 5])
        else:
            results = np.zeros([len(self.fold_pred), 4])

        for fold_key in range(len(self.fold_pred)):
            y_true = self.y[self.fold_test_index[fold_key]]
            y_pred = self.fold_pred[fold_key][classifier_name]
            results[fold_key, :] = np.asarray(compute_classification_scores(y_true, y_pred))
        logging.debug('Done.')
        return results

    def report_score_summary_by_classifier(self, classifier_name):
        results = self.score_summary_by_classifier(classifier_name)

        if self.n_labels == 2:
            header_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
        else:
            header_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

        results_formatted = format_results_table(results, header_names, row_names=None, operation='average', digits=4)
        logging.debug('Done.')
        return results_formatted

    def best_estimator(self, classifier_name, fold_key=None):

        results = self.score_summary_by_classifier(classifier_name)

        best_model = {}

        if (fold_key is None) or (fold_key not in range(0, results[:, 0].shape[0])):
            # Get the highest accuracy
            fold_key = results[:, 0].argmax()

        # Get the test and training indices of the data
        bm_test_indices = self.fold_test_index[fold_key]
        bm_train_indices = self.fold_train_index[fold_key]

        # Get the best models
        bm_model = self.fold_models[fold_key].grid_searches[classifier_name].best_estimator_

        # Get the prediction
        bm_y_pred = self.fold_pred[fold_key][classifier_name]

        best_model[classifier_name] = [fold_key, bm_model, bm_y_pred, bm_train_indices, bm_test_indices]

        logging.debug('Done.')
        return best_model

    def feature_importances(self, classifier_name, fold_key=None):

        best_model = self.best_estimator(classifier_name, fold_key)
        return best_model[classifier_name][1].feature_importances_

# if __name__ == '__main__':
#     from sklearn import datasets
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.svm import SVC
#
#     iris = datasets.load_iris()
#     X, y = iris.data[:, 1:4], iris.target
#
#     models = {
#         'SVC': SVC(random_state=1)
#     }
#
#     model_params = {
#         'SVC': {}
#     }
# #
#     mc = MultiClassifier()
#     mc.train(X, y, models, model_params)
#     print(mc.predict_proba())
