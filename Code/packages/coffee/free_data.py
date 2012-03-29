#!/usr/bin/env python
# encoding: utf-8
"""
freedata.py

Created by Jörg Frauendiener on 2012-03-09.
Copyright (c) 2012 University of Otago. All rights reserved.
"""

class FreeData(object):
    """
    This is a simple abstract base class for the implementation of exact 
    solutions.
    All the 'virtual' functions must be defined. 
    """
    
    def left_boundary(self, tslice):
        raise NotImplementedError(
            'You must define the function left_boundary().'
            )

    def right_boundary(self, tslice):
        raise NotImplementedError(
            'You must define the function right_boundary().'
            )

    def initial_data(self, t, r):
        raise NotImplementedError(
            'You must define the function initial_data().'
            )
        
    def exact(self, t, r):
        raise NotImplementedError('You must define the function exact().')
        
    def dirichlet(self, u, intStep = None):
        raise NotImplementedError('You must define the function dirichlet().')
