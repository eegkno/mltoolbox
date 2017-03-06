# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message

from ..search import MultiLearnerCV

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_multiple_estimator_init():
    mp = MultiLearnerCV(models={}, params={})

    # Validate that models dict is not empty
    msg = ('Invalid `models` attribute, `models`'
           ' should be a dict of {key: models}')
    assert_raise_message(AttributeError, msg, mp.fit, X, y)

    # Validate that params is not empty
    models = {
        'RandomForestClassifier': RandomForestClassifier()
    }
    mp = MultiLearnerCV(models=models, params={})
    msg = ('Invalid `params` attribute, `params`'
           ' should be a dict of {key: parameters}')
    assert_raise_message(AttributeError, msg, mp.fit, X, y)

    # Validate that the size of models and params is the same
    # Case 1:
    model_params_test = {
        'RandomForestClassifier': {'n_estimators': [8]},
        'RandomForestClassifier2': {'n_estimators': [8]}
    }
    mp = MultiLearnerCV(models=models, params=model_params_test)
    msg = ('Number of models and params must be equal'
           '; got 1 models, 2 params')
    assert_raise_message(AttributeError, msg, mp.fit, X, y)

    # Case 2:
    models_test = {
        'RandomForestClassifier': RandomForestClassifier(),
        'RandomForestClassifier2': RandomForestClassifier()
    }
    model_params = {
        'ExtraTreesRegressor': {'n_estimators': [8]}
    }
    mp = MultiLearnerCV(models=models_test, params=model_params)
    msg = ('Number of models and params must be equal'
           '; got 2 models, 1 params')
    assert_raise_message(AttributeError, msg, mp.fit, X, y)

    # Validate that both dicts have the same __keys
    model_params_test = {
        'RandomForestClassifier': {'n_estimators': [8]},
        'RandomForestClassifier2': {'n_estimators': [8]}
    }
    models_test = {
        'RandomForestClassifier': RandomForestClassifier(),
        'RandomForestClassifier3': RandomForestClassifier()
    }
    mp = MultiLearnerCV(models=models_test, params=model_params_test)
    msg = "Some models are missing parameters: ['RandomForestClassifier3']"
    assert_raise_message(ValueError, msg, mp.fit, X, y)

    model_params_test = {
        'RandomForestClassifier': {'n_estimators': [8]}
    }
    models_test = {
        'RandomForestClassifier': RandomForestClassifier()
    }
    mp = MultiLearnerCV(models=models_test, params=model_params_test)
    msg = "Method has to be one of gridsearch"
    assert_raise_message(ValueError, msg, mp.fit, X, y, method='other')


def test_fit():
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=1)
    }

    model_params = {
        'RandomForestClassifier': {'n_estimators': [8]}
    }

    # Default splits+
    mp = MultiLearnerCV(models=models, params=model_params)
    mp.fit(X, y)
    assert_equal(mp.grid_searches_['RandomForestClassifier'].n_splits_, 3)

    # Custum splits
    cv_params = {'cv': 5}
    mp = MultiLearnerCV(models=models, params=model_params)
    mp.fit(X, y, cv_params=cv_params)
    assert_equal(mp.grid_searches_['RandomForestClassifier'].n_splits_, 5)


def test_one_predictor():
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=1)
    }

    model_params = {
        'RandomForestClassifier': {'n_estimators': [8]}
    }
    mp = MultiLearnerCV(models=models, params=model_params)
    mp.fit(X, y)
    y_pred = mp.predict(X)
    assert_almost_equal(accuracy_score(y, y_pred['RandomForestClassifier']), 0.98, decimal=2)


def test_multiple_predictors():
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=1),
        'RandomForestClassifier2': RandomForestClassifier(random_state=1)
    }

    model_params = {
        'RandomForestClassifier': {'n_estimators': [8]},
        'RandomForestClassifier2': {'n_estimators': [8]}
    }

    mp = MultiLearnerCV(models=models, params=model_params)
    mp.fit(X, y)
    y_pred = mp.predict(X)
    assert_almost_equal(accuracy_score(y, y_pred['RandomForestClassifier']), 0.98, decimal=2)
    assert_almost_equal(accuracy_score(y, y_pred['RandomForestClassifier2']), 0.98, decimal=2)
    assert_array_equal(y_pred['RandomForestClassifier'], y_pred['RandomForestClassifier2'])


def test_prob_pred():
    iris = datasets.load_iris()
    X, y = iris.data[0:6, 1:3], iris.target[0:6]

    original_prob = np.array([[1.], [1.], [1.], [1.], [1.], [1.]])

    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=1)
    }

    model_params = {
        'RandomForestClassifier': {'n_estimators': [8]}
    }
    mp = MultiLearnerCV(models=models, params=model_params)
    mp.fit(X, y)
    y_pred = mp.predict_proba(X)
    assert_array_equal(original_prob, y_pred['RandomForestClassifier'])
