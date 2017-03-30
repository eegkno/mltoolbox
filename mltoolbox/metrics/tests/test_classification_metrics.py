# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal

from ..classification_metrics import compute_classification_scores


def test_compute_classification_scores_binary():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1])

    accuracy, specificity, precision, recall, f1_score, auc = compute_classification_scores(y_true, y_pred)
    assert_almost_equal(round(accuracy, 2), 0.90, decimal=2)
    assert_almost_equal(round(specificity, 2), 0.80, decimal=2)
    assert_almost_equal(round(precision, 2), 0.83, decimal=2)
    assert_almost_equal(round(recall, 2), 1.0, decimal=2)
    assert_almost_equal(round(f1_score, 2), 0.91, decimal=2)
    assert_almost_equal(round(auc, 2), 0.90, decimal=2)


def test_compute_classification_scores_multiclass():
    y_true = np.array([1, 1, 1, 1, 2, 0, 2, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 2, 2, 0, 0, 0, 1])

    accuracy, precision, recall, f1_score = compute_classification_scores(y_true, y_pred)
    assert_almost_equal(round(accuracy, 2), 0.70, decimal=2)

    assert_array_almost_equal([0.66, 0.8, 0.5], precision, decimal=2)
    assert_array_almost_equal([0.5, 1., 0.5], recall, decimal=2)
    assert_array_almost_equal([0.57, 0.88, 0.5], f1_score, decimal=2)
