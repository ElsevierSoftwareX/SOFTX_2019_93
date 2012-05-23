#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import sys
import os
import math
import numpy as np
import logging

from ..mpi import mpiinterfaces
#from tslices import timeslice
#from solvers import Solver
#from system import System

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
        name = "Grid", comparison = None):
        obj = np.asarray(grid).view(cls)
        obj.dim = len(grid.shape)
        obj.name = name
        if axes_step_sizes is None:
            axes_step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        obj.step_sizes = axes_step_sizes
        obj.log = logging.getLogger(name)
        obj.comparison = comparison
        return obj
        
    def __array_finalize__(self,obj):
        if obj is None: return
        self.dim = getattr(obj, 'dim', None)
        self.name = getattr(obj, 'name', None)
        self.log = getattr(obj, 'log', None)
        self.step_sizes = getattr(obj, 'step_sizes', None)
        self.comparison = getattr(obj, 'comparison', None)
        
    def __array_wrap__(self,out_arr,context = None):
        return np.asarray(out_arr)
        
    def validate(self,u,time):
        return self
        
    # These methods allow for integration with mpi enabled code for
    # grid classes that are not mpi enabled.
    def send(self,data): pass
        
    def recv(self,data): pass
        
    def collect_data(self,data):
        return data, self

#    def __repr__(self): 
#        return "<%s, grid shape = %s>"%(self.name,self.shape)

################################################################################
# Constructors for specific cases
################################################################################

class Interval_2D(Grid):
    """A Grid object to represent a 2D interval of coordinates"""
    
    def __new__(cls, shape, bounds, comparison = None):
        assert len(bounds) == len(shape)
        axes = [np.linspace(bounds[i][0],bounds[i][1],shape[i]+1)\
            for i in range(len(bounds))]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        name = "Interval_2D_%s_%s"%(repr(shape),repr(bounds))
        mesh = np.meshgrid(axes[1],axes[0])
        grid = np.dstack((mesh[1],mesh[0]))
        obj = Grid.__new__(cls, grid, step_sizes, name = name, 
            comparison = comparison)
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
        axes = [np.linspace(bounds[0][0],bounds[0][1],shape+1)]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        name = "Interval_1D_%s_%s"%(repr(shape),repr(bounds))
        if log is None:
            log = logging.getLogger(name)
        else:
            log = log.getChild(name)
        obj = Grid.__new__(cls, axes[0], step_sizes, name = name, 
            comparison = comparison, log = log)
        return obj

    def axes(self):
        return np.asarray(self)

    def right_boundary(self):
        return self[-1]
        
    def left_boundary(self):
        return self[0]

    @property
    def axes(self):
        return np.asarray(self)
        
class Interval_2D_polar_mpi(Grid):
    """A Grid object to represent a 2D interval of polar coordinates with
    mpi splitting of domains on the first axis."""
    
    def __new__(cls, shape, bounds, ghost_points, comparison, log=None):
        # Get step sizes
        axes = [np.linspace(bounds[0][0],bounds[0][1],shape[0]+1),\
            np.linspace(bounds[1][0],bounds[1][1],shape[1]+1)]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        mpiinter = mpiinterfaces.OneD_even(shape[0],ghost_points)
        # Get axes
        axes = [axes[0][mpiinter.domain],axes[1]]
        # Set name and log
        name = "Interval_2D_polar_mpi"
        log = log.getChild(name)
        # Calculate grid
        mesh = np.meshgrid(axes[1],axes[0])
        grid = np.dstack((mesh[1],mesh[0]))
        obj = Grid.__new__(cls, grid, step_sizes, name=name, 
            comparison=comparison,\
            log=log)
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
