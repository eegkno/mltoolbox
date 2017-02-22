#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='mltoolbox',
    version='0.0.0',
    description="Set of tools for ML in Python.",
    long_description=readme + '\n\n' + history,
    author="Edgar Garcia Cano",
    author_email='eegkno@gmail.com',
    url='https://github.com/eegkno/mltoolbox',
    packages=[
        'mltoolbox',
    ],
    package_dir={'mltoolbox':
                 'mltoolbox'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='mltoolbox',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
