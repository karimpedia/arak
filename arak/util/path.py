#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:21:22 2017

@author: karim
"""


import errno
import os


def splitroot(path):
    '''
    A method that splits a given path into components (root-dir, sub-dir-path).
    For instance: if path='/root/some/path/to/a/file.py', the output value will
    be ('/root', 'some/path/to/a/file.py')
    '''
    parent_dir, sub_dir = os.path.split(path)
    if not parent_dir or parent_dir == '/':
        return path, ''
    else:
        rec_parent, ext_parent = splitroot(parent_dir)
        return rec_parent, os.path.join(ext_parent, sub_dir)


def makedirpath(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as ose:
            if ose.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
    return path
