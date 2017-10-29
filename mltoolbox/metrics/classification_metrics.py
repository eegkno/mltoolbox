# -*- coding: utf-8 -*-
import logging
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from mltoolbox.base.logs import set_logger


def compute_classification_scores(y_true, y_pred, verbose=0):
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

    Example
    -------
    >>> from mltoolbox.metrics.classification_metrics import compute_classification_scores
    >>> y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    >>> y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
    >>> print(np.round(compute_classification_scores(y_true, y_pred), 2))
    [ 0.9   0.8   0.83  1.    0.91  0.9 ]
    """
    set_logger(verbose)

    n_labels = unique_labels(y_true, y_pred).size

    if n_labels == 2:

        CM = confusion_matrix(y_true, y_pred)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)  # sensitivity
        f1_score = (2 * TP) / (2 * TP + FP + FN)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        return accuracy, specificity, precision, recall, f1_score, auc_score
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, s = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        logging.debug("\n{}".format(classification_report(y_true, y_pred)))
        return accuracy, np.average(precision, weights=s), np.average(recall, weights=s), np.average(f1_score,
                                                                                                     weights=s)
# if __name__ == '__main__':
#     y_true = np.array([1, 1, 1, 1, 2, 0, 2, 0, 0, 0])
#     y_pred = np.array([1, 1, 1, 1, 2, 2, 0, 0, 0, 1])
#
#     print(compute_classification_scores(y_true, y_pred))
