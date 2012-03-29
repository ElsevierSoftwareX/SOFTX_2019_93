#!/usr/bin/env python
# encoding: utf-8 
"""
system.py

Created by Jörg Frauendiener on 2010-12-26.
Modified by Ben Whale on 2011-03-04.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

import sys
import os
import unittest

#############################################################################
class System(object):

    def timestep(self, u):
        return NotImplementedError
    
    def evaluate(self, t, u0):
        return -u0/t
        
    def initialValues(self, t0, grid = None):
        return 1.0
        
    def constraint_violation(self,u):
        return NotImplementedError
        
    def left(self, t):
        pass

    def right(self, t):
        pass
    
    def __str__(self):
        return self.name

class systemTests(unittest.TestCase):
    def setUp(self):
        pass
