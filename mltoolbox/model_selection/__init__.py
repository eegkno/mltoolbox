# -*- coding: utf-8 -*-
"""
The :mod:`mltoolbox.model_selection` module includes wrapper classes to train and test models.
"""

from .search import MultiLearnerCV
from .classification import MultiClassifier

__all__ = [
    'MultiLearnerCV',
    'MultiClassifier'
]
