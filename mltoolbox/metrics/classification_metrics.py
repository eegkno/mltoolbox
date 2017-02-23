# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.utils.multiclass import unique_labels


def compute_classification_scores(y_true, y_pred):
    """Compute the accuracy, precision, recall, f1-score and auc if the classification is binary

        Read more about the metrics in:
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    Parameters
    ----------
    y_true : numpy 1d-array-like, shape = [n_samples]
        Ground truth labels

    y_pred : numpy 1d-array-like, shape = [n_samples]
        Predicted labels

    Returns
    -------
    accuracy : float
        Return the correctly classified samples.

    precision : float
        The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        true positives and ``fp`` the number of false positives. The precision is
        intuitively the ability of the classifier not to label as positive a sample
        that is negative.

    recall : float
        The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
        true positives and ``fn`` the number of false negatives. The recall is
        intuitively the ability of the classifier to find all the positive samples.

    f1_score : float
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.

    auc : float
        Computes the area under the ROC curve.

    """
    n_labels = unique_labels(y_true, y_pred).size

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred)

    auc_score = None
    if n_labels == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)

    return accuracy, np.average(precision), np.average(recall), np.average(f1_score), auc_score


# if __name__ == '__main__':
#     y_true = np.array([1, 1, 1, 1, 2, 0, 2, 0, 0, 0])
#     y_pred = np.array([1, 1, 1, 1, 2, 2, 0, 0, 0, 1])
#
#     print(compute_classification_scores(y_true, y_pred))
