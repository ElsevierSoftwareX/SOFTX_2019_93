#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import unittest
import math
import numpy as np
import logger

from tslices import timeslice
from solvers import Solver
from system import System


class Grid(object):
    """docstring for Grid"""
    def __init__(self, shape):
        self.dim = len(shape)
        self.shape = shape
        self.name = "Unabstracted grid"
    
    def grid(self,time):
        raise NotImplementedError()
        
    def domain(self,time):
        raise NotImplementedError()
        
    def validate(self,u,time):
        raise NotImplementedError()
    
class Interval(Grid):
    """docstring for Interval"""
    def __init__(self, shape, bounds):
        super(Interval, self).__init__(shape)
        self._x = np.linspace(bounds[0],bounds[1],shape[0]+1)
        self.name = "1D Interval: %s" % str(self.shape)
    
    def domain(self,time):
        return self._x
        
    def validate(self,u,time):
        return u

class IntervalDrop(Grid):

    def __init__(self, shape, rMax, tMax, mu):
        super(IntervalDrop, self).__init__(shape)
        self._x = np.linspace(0,rMax,shape[0]+1)
        self.name = "1D FriedrichConformal: %s" % str(shape)
        self.mu = mu
        self.tMax = tMax
        self.log = logging.getLogger(debug_parent.join('.IntervalDrop'))

    def domain(self,time):
        return self._x[self.rMinIndex(self._x,time):]
            
    def validate(self,u,time):
        rMinIndex = self.rMinIndex(u.x,time)
        fields = u.fields[:,rMinIndex:]
        domain = u.x[rMinIndex:]
        return timeslice(fields,domain,u.time)

    def rMinIndex(self,domain,time):
        if time<=1:
            return 0
        else:
            mu = self.mu(domain)
            rMinIndex = np.where(mu<=(self.tMax/time))[0][0]#max(0,np.where(mu<=(self.tMax/time))[0][0]-1)
            return rMinIndex

    def __repr__(self):
        return self.name

class IntervalZero(Grid):

    def __init__(self, shape, rMax, tMax, mu, debug_parent="main"):
        super(IntervalZero, self).__init__(shape)
        self._x = np.linspace(0,rMax,shape[0]+1)
        self.name = "1D FriedrichConformal: %s" % str(shape)
        self.mu = mu
        self.tMax = tMax
        

    def domain(self,time):
        return self._x
            
    def validate(self,u,time):
        rMinIndex = self.rMinIndex(u.x,time)
        fields = u.fields
        fields[:,:rMinIndex+1] = 0
        return timeslice(fields,u.x,time)

    def rMinIndex(self,domain,time):
        if time<=1:
            return 0
        else:
            mu = self.mu(domain)
            rMinIndex = max(0,np.where(mu<=(self.tMax/time))[0][0]-1)
            if __debug__:
                print "rMinIndex for domain truncation = %i"%rMinIndex
                print "mu is = %s"%repr(mu)
                print "scri is = %s"%repr(mu-time)
            return rMinIndex

    def __repr__(self):
        return self.name

class Periodic(Grid):
    """docstring for Loop"""
    def __init__(self, shape, bounds):
        super(Periodic, self).__init__(shape)
        x = np.linspace(bounds[0],bounds[1],shape[0]+1)
        self.domain= x[:-1]
        self.name = "1D Periodic: %s" % (str(self.grid.shape))
        
    def domain(self,values,time):
        return self.domain
