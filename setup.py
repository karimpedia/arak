#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:18:39 2017

@author: karim
"""

from setuptools import setup
from setuptools import find_packages


install_requires = []
tests_require = install_requires + ["pytest"]
docs_require = install_requires
extras_require = {"tests": tests_require,
                  "docs": docs_require}


setup_options = \
    dict(name='arak',
         version='0.0.1',
         packages=['arak'],
         license='GNU',
         author='Karim Said',
         author_email='karim.pedia@gmail.com',
         description='neural networks',
         url='https://github.com/karimpedia/arak',
         download_url='https://github.com/karimpedia/arak.git',
         install_requires=install_requires,
         tests_require=tests_require,
         extras_require=extras_require,
         packages=find_packages())


if __name__ == '__main__':
    setup(**setup_options)

