#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import sys
import os
import unittest
import math
import numpy as np
import logging

from tslices import timeslice
from solvers import Solver
from system import System


class Grid(object):
    """docstring for Grid"""
    def __init__(self, shape, log):
        self.dim = len(shape)
        self.shape = shape
        self.name = "Unabstracted grid"
        self.log = log
    
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
            rMinIndex = np.where(mu<=(self.tMax/time))[0][0]
            return rMinIndex

    def __repr__(self):
        return self.name

class IntervalZero(Grid):

    def __init__(self, shape, rMax,mu, mup, debug_parent="main",\
        percent = 0):
        super(IntervalZero, self).__init__(shape,\
            logging.getLogger(debug_parent+".IntervalZeroGrid"))
        self._x = np.linspace(0,rMax,shape[0]+1)
        self.name = "1DIntervalZero:%s" % str(shape)
        self.mu = mu
        self.mup = mup
        self.percent = percent
        def kappa(r):
            return r*self.mu(r)
        def kappap(r):
            return self.mu(r)+r*self.mup(r)
        self.kappa = kappa
        self.kappap = kappap

    def domain(self,time):
        return self._x
            
    def validate(self,u,time):
        rMinIndex = self.rMinIndex(u.x,time)
        fields = u.fields
        fields[:,:rMinIndex+1] = 0
        return timeslice(fields,u.x,time)

    def rMinIndex(self,domain,time):
        """This method selects the index, i, so that
        domain[i] is greater than the value of
        (1-t*kappap) (the line which describes the degeneracy of
        the characteristics) and less than to value of
        (1/mu)-time (the line which describes where scri+ is) and
        so that i is the best estimate for the coordinate point
        self.percent from (1-t*kappap) to (1/mu)-time.
        That is if self.percent = 0.5 then i will be the index which
        is the best approximation to half way between
        (1-t*kappa) and (1/mu)-time.        
        """
        if time<=1:
            return 0
        else:
            mu = self.mu(domain)
            bound = 1-time*self.kappap(domain)
            scri = (1./self.mu(domain))-time
            boundMinIndex = np.nonzero(\
                np.absolute(bound)==np.min(np.absolute(bound))\
                )[0][0]
            scriMinIndex = np.nonzero(\
                np.absolute(scri) == np.min(np.absolute(scri))\
                )[0][0]
            if bound[boundMinIndex]<0:
                boundMinIndex +=1
            if scri[scriMinIndex]>0:
                scriMinIndex -=1
            self.log.debug("bound is = %s"%repr(bound))
            self.log.debug("boundMinIndex = %i"%boundMinIndex)
            self.log.debug("scriMinIndex = %i"%scriMinIndex)
            self.log.debug("scri is = %s"%repr(scri))
            #return scriMinIndex
            if boundMinIndex == scriMinIndex:
                rIndex = boundMinIndex
            elif boundMinIndex<scriMinIndex:
                indexArrayPercent = (np.array(\
                    range(boundMinIndex,scriMinIndex+1)\
                    )\
                    -boundMinIndex)/(scriMinIndex-boundMinIndex)
                rIndex = np.absolute(indexArrayPercent-self.percent).argmin()\
                   +boundMinIndex
            else:
                rIndex = 0
            self.log.debug("Cut off index is %i"%rIndex)
            return rIndex

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
