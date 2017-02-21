#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mltoolbox
----------------------------------

Tests for `mltoolbox` module.
"""

import pytest

from mltoolbox import mltoolbox


def hello_world():
    """Return the string `'Hello world!'`.

    Examples
    --------
    >>> hello_world()
    'Hello world!'
    """
    return 'Hello world!'


def test_hello_world():
    assert hello_world() == 'Hello world!'
