# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message

from ..classification import MultiClassifier

iris = datasets.load_iris()
X, y = iris.data[:, 1:4], iris.target

# The test of the correct parameters is managed by MultiLearnerCV
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=1)
}

model_params = {
    'RandomForestClassifier': {'n_estimators': [8]}
}


def test_multiclassifier_init():
    mc = MultiClassifier()
    mc.train(X, y, models, model_params)


def test_multiclassifier_prediction_multiclass():
    y_true = {
        0: {'RandomForestClassifier': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                                2, 2, 2, 1, 2, 2, 2])},
        1: {'RandomForestClassifier': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                                2, 2, 2, 2, 2, 2, 1])},
        2: {'RandomForestClassifier': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2,
                                                2, 2, 2, 2, 2, 2, 1])},
        3: {'RandomForestClassifier': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                                2, 2, 2, 2, 2, 2, 2])},
        4: {'RandomForestClassifier': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                                2, 2, 2, 2, 2, 2, 2])}}

    mc = MultiClassifier()
    mc.train(X, y, models, model_params)
    y_pred = mc.predict()

    for p, t in zip(y_pred, y_true):
        pred = y_pred[p]['RandomForestClassifier']
        true = y_true[t]['RandomForestClassifier']

        assert_almost_equal(true, pred, decimal=2)


def test_multiclassifier_summary_binary():
    y_binary_idx = np.where(y != 2)
    mc = MultiClassifier()
    mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)

    summary = mc.score_summary_by_classifier('RandomForestClassifier')

    true_summary = np.ones([5, 5])

    assert_almost_equal(np.mean(true_summary), np.mean(summary), decimal=2)


def test_multiclassifier_report_binary():
    y_binary_idx = np.where(y != 2)
    mc = MultiClassifier()
    mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)
    report = mc.report_score_summary_by_classifier('RandomForestClassifier')

    with open("mltoolbox/model_selection/tests/test_classification_report_files/report_binary.txt",
              "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_multiclassifier_best_estimator_binary():
    y_binary_idx = np.where(y != 2)
    mc = MultiClassifier()
    mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)
    # Compute the best_estimator without previous predict
    bs = mc.best_estimator('RandomForestClassifier')['RandomForestClassifier']
    assert_equal(bs[0], 1)


def test_multiclassifier_best_estimator_false_fold_index():
    y_binary_idx = np.where(y != 2)
    mc = MultiClassifier()
    mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)

    # Compute the best_estimator without previous predict
    bs = mc.best_estimator('RandomForestClassifier', 4)['RandomForestClassifier']
    assert_equal(bs[0], 4)

    bs = mc.best_estimator('RandomForestClassifier', 6)['RandomForestClassifier']
    assert_equal(bs[0], 1)

    bs = mc.best_estimator('RandomForestClassifier', -1)['RandomForestClassifier']
    assert_equal(bs[0], 1)


def test_multiclassifier_feature_importances():
    y_binary_idx = np.where(y != 2)
    mc = MultiClassifier()
    mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)

    true_fi = [ 0.25347217, 0.56107946, 0.18544837]
    fi = mc.feature_importances('RandomForestClassifier')
    assert_array_almost_equal(true_fi, fi, decimal=2)


def test_multiclassifier_compare_best_estimator():
    models = {
        'SVC': SVC(random_state=1)
    }

    model_params = {
        'SVC': {}
    }

    y_binary_idx = np.where(y != 2)
    mc = MultiClassifier()
    mc.train(X[y_binary_idx], y[y_binary_idx], models, model_params)

    _, bm_model, _, bm_train_indices, bm_test_indices = mc.best_estimator('SVC')[
        'SVC']

    X_train = X[bm_train_indices]
    y_train = y[bm_train_indices]

    X_test = X[bm_test_indices]
    y_test = y[bm_test_indices]

    rfc = SVC(**bm_model.get_params())
    rfc.fit(X_train, y_train)

    assert_equal(bm_model.score(X_test, y_test), rfc.score(X_test, y_test))


def test_multiclassifier_estimator_with_prob():
    models = {
        'SVC': SVC(random_state=1)
    }

    model_params = {
        'SVC': {}
    }

    mc = MultiClassifier()
    mc.train(X, y, models, model_params)
    msg = ('predict_proba is not available when  probability=False')
    assert_raise_message(AttributeError, msg, mc.predict_proba)

# def test_multiclassifier_best_estimator_predict():
#     mc = MultiClassifier()
#     mc.train(X, y, models, model_params)
#     # Compute the prediction before obtaining the best_estimator
#     mc.predict()
#     bs = mc.best_estimator('RandomForestClassifier')['RandomForestClassifier']
#     assert_equal(bs[0], 4)
#
#
# def test_multiclassifier_best_estimator_predict_proba():
#     mc = MultiClassifier()
#     mc.train(X, y, models, model_params)
#     # Compute the prediction before obtaining the best_estimator
#     mc.predict_proba()
#     bs = mc.best_estimator('RandomForestClassifier')['RandomForestClassifier']
#     assert_equal(bs[0], 4)

# def test_multiclassifier_summary_multiclass():
#     mc = MultiClassifier()
#     mc.train(X, y, models, model_params)
#
#     summary = mc.score_summary_by_classifier('RandomForestClassifier')
#
#     true_summary = [[0.96666667, 0.96969697, 0.96666667, 0.96658312],
#                     [0.96666667, 0.96969697, 0.96666667, 0.96658312],
#                     [0.93333333, 0.93333333, 0.93333333, 0.93333333],
#                     [0.96666667, 0.96969697, 0.96666667, 0.96658312],
#                     [1, 1, 1, 1]]
#
#     assert_almost_equal(np.mean(true_summary), np.mean(summary), decimal=2)


# def test_multiclassifier_report_multiclass():
#     mc = MultiClassifier()
#     mc.train(X, y, models, model_params)
#     report = mc.report_score_summary_by_classifier('RandomForestClassifier')
#
#     with open("mltoolbox/model_selection/tests/test_classification_report_files/report_multiclass.txt",
#               "r") as expected_report_file:
#         expected_report = expected_report_file.read()
#
#     assert_equal(report, expected_report)
