#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 01:47:01 2017

@author: karim
"""

from datetime import datetime

def TS(msg=None):
    if msg is None:
        return datetime.now().strftime('%Y.%m.%d %H:%M:%S.%f') + ') '
    else:
        return datetime.now().strftime('%Y.%m.%d %H:%M:%S.%f') + ') ' + msg
