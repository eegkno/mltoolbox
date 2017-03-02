# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal

from ..classification_metrics import compute_classification_scores


def test_compute_classification_scores_binary():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1])

    accuracy, precision, recall, f1_score, auc = compute_classification_scores(y_true, y_pred)
    assert_almost_equal(round(accuracy, 2), 0.90, decimal=2)
    assert_almost_equal(round(precision, 2), 0.92, decimal=2)
    assert_almost_equal(round(recall, 2), 0.90, decimal=2)
    assert_almost_equal(round(f1_score, 2), 0.90, decimal=2)
    assert_almost_equal(round(auc, 2), 0.90, decimal=2)

def test_compute_classification_scores_multiclass():
    y_true = np.array([1, 1, 1, 1, 2, 0, 2, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 2, 2, 0, 0, 0, 1])

    accuracy, precision, recall, f1_score = compute_classification_scores(y_true, y_pred)
    assert_almost_equal(round(accuracy, 2), 0.70, decimal=2)
    assert_almost_equal(round(precision, 2), 0.68, decimal=2)
    assert_almost_equal(round(recall, 2), 0.69, decimal=2)
    assert_almost_equal(round(f1_score, 2), 0.68, decimal=2)
