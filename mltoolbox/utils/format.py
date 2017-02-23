# -*- coding: utf-8 -*-

import numpy as np


def format_results_table(results_table, header_names, row_names=None, operation=None, col_span=10, digits=4):
    """Build a customized text formatted table

    Parameters
    ----------
    results_table : 2d array-like, shape = [n_rows, n_cols]
        Array of data to be formatted.

    header_names : list of strings, shape = [n_headers_name]
        List of names of each column.

    row_names : list of strings, shape = [n_headers_name]
        Optional list of row names.

    operation : string, [None (default), 'average', 'sum']
        Optional parameter, it is required if the average or the sum of each column needs to be calculated.

    col_span : int
        Optional value to indicate the separation between two headers.

    digits : int
        Optional number of digits for formatting output floating point values.

    Returns
    -------
    report: string
        Text formatted according to the headers names, row names and operation indicated.

    Examples
    --------
    >>> from mltoolbox.utils.format import format_results_table
    >>> data = np.arange(10.0).reshape(2, 5)
    >>> headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    >>> print(format_results_table(data, headers, operation='average'))
                 Accuracy  Precision     Recall   F1-score        AUC
    <BLANKLINE>
            1      0.0000     1.0000     2.0000     3.0000     4.0000
            2      5.0000     6.0000     7.0000     8.0000     9.0000
    <BLANKLINE>
      Average      2.5000     3.5000     4.5000     5.5000     6.5000
    <BLANKLINE>
    """
    if len(header_names) != results_table.shape[1]:
        raise ValueError('The header_names size is different to number of columns in results_table ' +
                         '; got %d and %d respectively'
                         % (len(header_names), results_table.shape[1]))

    operation_values = (None, 'average', 'sum')
    if operation not in operation_values:
        raise ValueError('operation has to be one of ' +
                         str(operation_values))

    if row_names is None:
        row_names = [str(i + 1) for i in range(results_table.shape[0])]
    elif len(row_names) != results_table.shape[0]:
        raise ValueError('The row_names size is different to number of rows in results_table ' +
                         '; got %d and %d respectively'
                         % (len(row_names), results_table.shape[0]))

    # Space used when an operation is specified
    last_row = 'average'
    header_width = max(len(cn) for cn in header_names)
    row_width = max(len(cn) for cn in row_names)
    width = max(header_width, len(last_row), digits)
    width = max(width, row_width, digits)

    # Generate the format for the header
    cols = len(header_names)
    headers_format = u'{:>{width}s} ' + (u' {:>' + str(col_span) + '}') * cols
    report = headers_format.format(u'', *header_names, width=width)
    report += u'\n\n'

    # Generate the format for the rows

    row_format = u'{:>{width}s} ' + (u' {:>' + str(col_span) + '.{digits}f}') * cols + u'\n'
    rows = zip(row_names, results_table)
    for label, row in rows:
        report += row_format.format(label, *row, width=width, digits=digits)

    report += u'\n'

    mean_results = []
    if operation == 'average':
        last_row = 'Average'
        mean_results = np.mean(results_table, axis=0)
    elif operation == 'sum':
        last_row = 'Sum'
        mean_results = np.sum(results_table, axis=0)

    if operation is not None:
        # Generate the operation row
        report += row_format.format(last_row, *mean_results, width=width, digits=digits)

    return report

# if __name__ == '__main__':
#    data = np.arange(10.0).reshape(2, 5)
#    headers = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
#    report = format_results_table(data, headers, operation='average')

#    text_file = open("tests/test_format_files/sum_long_word_header.txt", "w")
#    text_file.write(report)
#    text_file.close()
