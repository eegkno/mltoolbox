# -*- coding: utf-8 -*-

import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message

from ..format import format_results_table


def test_format_results_table_parameters():
    # Case 1:
    data = np.arange(8.0).reshape(4, 2)
    headers = ["A", "B", "C", "C"]
    msg = 'The header_names size is different to number of columns in results_table ; got 4 and 2 respectively'
    assert_raise_message(ValueError, msg, format_results_table, data, headers)

    # Case 2:
    data = np.arange(8.0).reshape(2, 4)
    headers = ["A", "B", "C", "D"]
    msg = 'operation has to be one of (None, \'average\', \'sum\')'
    assert_raise_message(ValueError, msg, format_results_table, data, headers, operation='x')

    # Case 3:
    data = np.arange(8.0).reshape(2, 4)
    headers = ["A", "B", "C", "D"]
    row_names = [1, 2, 3]
    msg = 'The row_names size is different to number of rows in results_table ; got 3 and 2 respectively'
    assert_raise_message(ValueError, msg, format_results_table, data, headers, row_names)


def test_format_results_table_average():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    report = format_results_table(data, headers, operation='average')

    with open("mltoolbox/utils/tests/test_format_files/average.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_format_results_table_sum():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    report = format_results_table(data, headers, operation='sum')

    with open("mltoolbox/utils/tests/test_format_files/sum.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_format_results_table_none():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    report = format_results_table(data, headers)

    with open("mltoolbox/utils/tests/test_format_files/none.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_format_results_table_none_2_digits():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    report = format_results_table(data, headers)

    with open("mltoolbox/utils/tests/test_format_files/none_2_digits.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_format_results_table_average_word_labels():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    labels = ['one', 'two']
    report = format_results_table(data, headers, row_names=labels, operation='average')

    with open("mltoolbox/utils/tests/test_format_files/average_word_labels.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_format_results_table_sum_long_word_label():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    labels = ['one', 'super_long_label']
    report = format_results_table(data, headers, row_names=labels, operation='sum')

    with open("mltoolbox/utils/tests/test_format_files/sum_long_word_label.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)


def test_format_results_table_sum_long_word_header():
    data = np.arange(10.0).reshape(2, 5)
    headers = ['Accuracy', 'Long label Precision', 'Recall', 'F1-score', 'AUC']
    labels = ['one', 'two']
    report = format_results_table(data, headers, row_names=labels, col_span=21, operation='sum')

    with open("mltoolbox/utils/tests/test_format_files/sum_long_word_header.txt", "r") as expected_report_file:
        expected_report = expected_report_file.read()

    assert_equal(report, expected_report)
