#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import sys
import os
import math
import numpy as np
import abc
import logging

#from ..mpi import mpiinterfaces
#from tslices import timeslice
#from solvers import Solver
#from system import System

LEFT = -1
CENTRE = 0
RIGHT = 1

################################################################################
# Base Grid class
################################################################################

class Grid(object):
    """The abstract base class for Grid objects."""
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, shape, bounds, name = "Grid", comparison = None):
        self.dim = len(shape)
        self.name = name
        self.log = logging.getLogger(name)
        self.comparison = comparison
        self.shape = shape
        self.bounds = bounds
        
    def validate(self,u,time):
        return self
        
    @abc.abstractproperty
    def axes(self): pass
    
    @abc.abstractproperty
    def step_sizes(self): pass
    
    @abc.abstractproperty
    def step_sizes(self): pass
    
    @property
    def meshes(self):
        axes = self.axes
        grid_shape = tuple([axis.size for axis in axes])
        mesh = []
        for i, axis in enumerate(axes):
            strides = np.zeros((len(self.shape), ))
            strides[i] = axis.itemsize
            mesh += [np.lib.stride_tricks.as_strided(
                axis,
                grid_shape,
                strides)
                ]
        return mesh

class GridMpi(Grid):
    """The abstract base class for Grid objects with mpi implementation."""
    
    def __init__(self, shape, bounds, name = "GridMpi", comparison = None):
        super(GridMpi, self).__init__(shape, bounds, name, comparison)
        
    @abc.abstractmethod
    def send(self,data): pass
        
    @abc.abstractmethod
    def recv(self,data): pass
        
    @abc.abstractmethod    
    def collect_data(self, data):
        return data, self

################################################################################
# Constructors for specific cases
################################################################################

class UniformCart(Grid):
    """A Grid object to represent an ND interval of coordinates"""
    
    def __init__(self, shape, bounds, comparison = None):
        assert len(bounds) == len(shape)
        name = "UniformCart%s%s%s"%(shape,bounds,comparison)
        super(UniformCart, self).__init__(shape, bounds, name=name, 
            comparison = comparison)
            
    @property
    def axes(self):
        axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            )
            for i in range(len(self.bounds))
            ]
        return axes
        
    @property
    def step_sizes(self):
        step_sizes = [axis[1]-axis[0] for axis in self.axes]
        return step_sizes
         
class S2(Grid):
    """A Grid object representing the sphere. Note that
    theta is in [0, pi) and phi is in [0, 2*pi)."""
    
    def __init__(self, shape, comparison = None):
        assert len(shape) == 2
        name = "S2%s%s"%(shape,comparison)
        super(S2, self).__init__(shape, 
            [[0, math.pi], [0, 2*math.pi]], 
            name=name, 
            comparison = comparison
            )
            
    @property
    def axes(self):
        axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i],
                endpoint=False
            )
            for i in range(len(self.bounds))
            ]
        return axes
        
    @property
    def step_sizes(self):
        step_sizes = [axis[1]-axis[0] for axis in self.axes]
        return step_sizes   
            
################################################################################
# Obsoleat constructors
################################################################################
            
class Interval_2D(Grid):
    """A Grid object to represent a 2D interval of coordinates"""
    
    def __new__(cls, shape, bounds, comparison = None):
        assert len(bounds) == len(shape)
        axes = [np.linspace(bounds[i][0],bounds[i][1],shape[i]+1)\
            for i in range(len(bounds))]
        step_sizes = np.asarray([axis[1]-axis[0] for axis in axes])
        name = "Interval_2D_%s_%s"%(repr(shape),repr(bounds))
        meshes = np.meshgrid(axes[1],axes[0])
        grid = np.dstack((mesh[1],mesh[0]))
        obj = Grid.__new__(cls, grid, meshes, step_sizes, name = name, 
            comparison = comparison)
        return obj

    @property
    def axes(self):
        return [np.asarray(self[:,0,0]),np.asarray(self[0,:,1])]
        
    @property
    def mesh(self):
        return self.mesh
        
 
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
        obj = Grid.__new__(cls, axes[0], None, step_sizes, name = name, 
            comparison = comparison)
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
