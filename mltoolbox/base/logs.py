# -*- coding: utf-8 -*-
import logging


class SetLogger(object):
    """Configure logger to use it as verbose mode in classes

    Parameters
    ----------
    verbose : int
        Level of the logger.

    """

    def __init__(self, verbose=0):
        if verbose > 0:
            level = 0
            if verbose == 1:
                level = logging.INFO
            elif verbose == 2:
                level = logging.DEBUG
            elif verbose == 3:
                level = logging.WARNING
            elif verbose == 4:
                level = logging.ERROR
            elif verbose == 5:
                level = logging.CRITICAL

            format = '%(module)s::%(funcName)s() - %(levelname)s: %(message)s'
            datefmt = '%m-%d %H:%M'
            logging.StreamHandler()
            logging.basicConfig(level=level, format=format, datefmt=datefmt)


def set_logger(verbose):
    """Configure logger to use it as verbose mode in functions

    Parameters
    ----------
    verbose : int
        Level of the logger.

    """

    level = 0
    if verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    elif verbose == 3:
        level = logging.WARNING
    elif verbose == 4:
        level = logging.ERROR
    elif verbose == 5:
        level = logging.CRITICAL

    format = '%(module)s::%(funcName)s() - %(levelname)s: %(message)s'
    datefmt = '%m-%d %H:%M'
    logging.basicConfig(level=level, format=format, datefmt=datefmt)


