#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:26:09 2017

@author: karim
"""


from arak.util.path import splitroot


def test_splitroot_example():
    a_in = '/root/some/path/to/a/file.py'
    a_out = splitroot(a_in)
    a_tgt = ('/root', 'some/path/to/a/file.py')
    assert all([x == y for x, y in zip(a_out, a_tgt)])


def test_splitroot_example_without_root():
    a_in = 'root/some/path/to/a/file.py'
    a_out = splitroot(a_in)
    a_tgt = ('root', 'some/path/to/a/file.py')
    assert all([x == y for x, y in zip(a_out, a_tgt)])


def test_splitroot_example_without_extension():
    a_in = '/root/some/path/to/a/file'
    a_out = splitroot(a_in)
    a_tgt = ('/root', 'some/path/to/a/file')
    assert all([x == y for x, y in zip(a_out, a_tgt)])


def test_splitroot_example_trailing_directory():
    a_in = '/root/some/path/to/a/directory/'
    a_out = splitroot(a_in)
    a_tgt = ('/root', 'some/path/to/a/directory/')
    assert all([x == y for x, y in zip(a_out, a_tgt)])
