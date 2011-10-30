#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import sys
import os
import math
import numpy as np
import logging

import mpiinterfaces
from tslices import timeslice
from solvers import Solver
from system import System

LEFT = -1
CENTRE = 0
RIGHT = 1

################################################################################
# Base Grid class
################################################################################

class Grid(np.ndarray):
    """The base class for Grid objects. Not that the log.getChild is not called
    on the passed log."""
    def __new__(cls, grid, axes_step_sizes, \
        name = "Grid", log = None,\
        comparison = None):
        obj = np.asarray(grid).view(cls)
        obj.dim = len(grid.shape)
        obj.name = name
        obj.comparison = comparison
        if axes_step_sizes is None:
            axes_step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        obj.step_sizes = axes_step_sizes
        obj.log = log
        #log.debug("obj = %s"%repr(obj))
        return obj
        
    def __array_finalize__(self,obj):
        if obj is None: return
        self.dim = getattr(obj, 'dim', None)
        self.name = getattr(obj, 'name', None)
        self.log = getattr(obj, 'log', None)
        self.step_sizes = getattr(obj, 'step_sizes', None)
        self.comparison = getattr(obj, 'comparison', None)
        
    def __array_wrap__(self,out_arr,context = None):
        return np.ndarray.__array_wrap__(self, out_arr,context)
        
    def validate(self,u,time):
        return self
        
    # These methods allow for integration with mpi enabled code for
    # grid classes that are not mpi enabled.
    def send(self,data): pass
        
    def recv(self,data): pass
        
    def collect_data(self,data):
        return data, self

    def __repr__(self): 
        return "<%s, grid shape = %s>"%(self.name,self.shape)

################################################################################
# Constructors for specific cases
################################################################################

class Interval_2D(Grid):
    """A Grid object to represent a 2D interval of coordinates"""
    
    def __new__(cls, shape, bounds, **kwds):
        assert len(bounds) == len(shape)
        axes = [np.linspace(bounds[i][0],bounds[i][1],shape[i]+1)\
            for i in range(len(bounds))]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        name = "Interval_2D"
        if log is None:
            log = logging.getLogger(name)
        else:
            log = log.getChild(name)
        mesh = np.meshgrid(axes[1],axes[0])
        grid = np.dstack((mesh[1],mesh[0]))
        obj = Grid.__new__(cls,grid,step_sizes,name = name, comparison = comparison,\
            log = log)
        return obj

    @property
    def axes(self):
        return [np.asarray(self[:,0,0]),np.asarray(self[0,:,1])]
        
    @property
    def r(self):
        return 0
        
    @property
    def phi(self):
        return 1 
 
class Interval_1D(Grid):
    """A Grid object to represent a 1D interval of coordinates"""

    def __new__(cls, shape, bounds, comparison = None, log = None):
        assert len(bounds) == 1
        axes = [np.linspace(bounds[0][0],bounds[0][1],shape[0]+1)]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        name = "Interval_1D"
        if log is None:
            log = logging.getLogger(name)
        else:
            log = log.getChild(name)
        obj = Grid.__new__(cls,axes[0],step_sizes,name = name, comparison = comparison,\
            log = log)
        return obj

    @property
    def axes(self):
        return np.asarray(self)
        
class Interval_2D_polar_mpi(Grid):
    """A Grid object to represent a 2D interval of polar coordinates with
    mpi splitting of domains on the first axis."""
    
    def __new__(cls,shape,bounds,ghost_points,comparison,log=None):
        # Get step sizes
        axes = [np.linspace(bounds[0][0],bounds[0][1],shape[0]+1),\
            np.linspace(bounds[1][0],bounds[1][1],shape[1]+1)]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        mpiinter = mpiinterfaces.OneD_even(shape[0],ghost_points)
        # Get axes
        axes = [axes[0][mpiinter.domain],axes[1]]
        # Set name and log
        name = "Interval_2D_polar_mpi"
        if log is None:
            log = logging.getLogger(name)
        else:
            log = log.getChild(name)
        # Calculate grid
        mesh = np.meshgrid(axes[1],axes[0])
        grid = np.dstack((mesh[1],mesh[0]))
        obj = Grid.__new__(cls,grid,step_sizes,name = name, comparison = comparison,\
            log = log)
        obj.mpiinter = mpiinter
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: return
        self.mpiinter = getattr(obj, 'mpiinter', None)
    
    def send(self,data):
        self.mpiinter.send(data)
        
    def recv(self,data):
        self.mpiinter.recv(data)
        
    def collect_data(self,data):
        return self.mpiinter.collect_data(self,data)

    @property
    def axes(self):
        return [np.asarray(self[:,0,0]),np.asarray(self[0,:,1])]
        
    @property
    def r(self):
        return 0
        
    @property
    def phi(self):
        return 1

class Interval_2D_polar_mpi_PT(Grid):
    """A Grid object to represent a 2D interval of polar coordinates with
    mpi splitting of domains on the first axis using a penalty boundary
    method to implement the parallelization."""
    
    def __new__(cls,gridshape,bounds,comparison,log=None,mpi=True):
        # Get step sizes
        axes = [np.linspace(bounds[0][0],bounds[0][1],gridshape[0]+1),\
            np.linspace(bounds[1][0],bounds[1][1],gridshape[1]+1)]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        # If using mpi divide up axes
        mpiinter = None
        if mpi:
            mpiinter = mpiinterfaces.PT_2D_1D(axes[0].shape[0])
            axes = [axes[0][mpiinter.domain],axes[1]]
        # Set name and log
        name = "Interval_2D_polar_mpi_%s"%repr(comparison)
        if log is None:
            log = logging.getLogger(name)
        else:
            log = log.getChild(name)
        # Calculate grid
        mesh = np.meshgrid(axes[1],axes[0])
        grid = np.dstack((mesh[1],mesh[0]))
        obj = Grid.__new__(cls,grid,step_sizes,name = name, comparison = comparison,\
            log = log)
        obj.mpiinter = mpiinter
        obj.gridshape = gridshape
        obj.bounds = bounds
        obj.comparison = comparison
        obj.log = log        
        return obj
    
    def __array_finalize__(self,obj):
        if obj is None: return
        self.mpiinter = getattr(obj, 'mpiinter', None)
        self.gridshape = getattr(obj, 'gridshape',None)
        self.bounds = getattr(obj, 'bounds',None)
        self.comparison = getattr(obj, 'comparison',None)
        self.log = getattr(obj, 'log',None)
    
    def get_edge(self,data):
        return self.mpiinter.get_edge(data)
   
    def collect_data(self,data):
        rdata = self.mpiinter.collect_data(data)
        rgrid = Interval_2D_polar_mpi_PT(self.gridshape,self.bounds,self.comparison,
            log = self.log,mpi=False)
        return rdata,rgrid
   
    @property
    def axes(self):
        return [np.asarray(self[:,0,0]),np.asarray(self[0,:,1])]
        
    @property
    def r(self):
        return 0
        
    @property
    def phi(self):
        return 1

           
################################################################################
# Old Grid objects - The code below will not work
################################################################################        
    
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
            if __debug__:
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
            if __debug__:
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
