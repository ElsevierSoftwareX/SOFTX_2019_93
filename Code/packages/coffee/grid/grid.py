#!/usr/bin/env python
# encoding: utf-8

# imports from external libraries
from __future__ import division
import math
import numpy as np
import abc
import logging

# imports from coffee
from coffee.mpi import mpiinterfaces

################################################################################
# Base Grid class
################################################################################
class ABCGrid(object):
    """The abstract base class for Grid objects."""
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, 
        shape, bounds, name = "Grid", comparison = None,
        mpi = None, *args, **kwds
        ):
        self.mpi = mpi
        self.dim = len(shape)
        self.name = name
        self.log = logging.getLogger(name)
        self.comparison = comparison
        self.shape = shape
        self.bounds = bounds
    
    def __strs__(self):
        return self.name

    def __repr__(self):
        return "<%s shape=%s, bounds=%s, comparison=%s, mpi=%s>"%(
            self.name, self.shape, self.bounds, self.comparison, self.mpi
            )

    def communicate(self, data):
        if self.mpi is None:
            return []
        return self.mpi.communicate(data)

    def boundary_slices(self, shape):
        if self.mpi is None:
            if __debug__:
                self.log.debug(
                    "No mpi object. Calculating boundaries in grid object"
                    )
            extra_dims = len(shape) - len(self.shape)
            edims_shape = shape[:extra_dims]
            edims_slice = tuple([
                slice(None,None,None)
                for i in range(extra_dims)
                ])
            r_slices = []
            for i in range(len(self.shape)):
                r_slice = [
                    slice(None, None, None)
                    for d in shape
                    ]
                r_slice[i] = slice(None, 1, None)
                r_slices += [tuple(r_slice)]
                r_slice[i] = slice(-1, None, None)
                r_slices += [tuple(r_slice)]
            return r_slices
        return self.mpi.boundary_slices(shape)

    def collect_data(self, data):
        if self.mpi is None:
            return 
        return self.mpi.collect_data(data)
    
    @abc.abstractproperty
    def axes(self): pass
    
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

################################################################################
# Constructors for specific cases
################################################################################
class UniformCart(ABCGrid):
    
    def __init__(self, 
            shape, bounds, 
            mpi_comm=None, comparison=None, name=None, ghost_points=1,
            *args, **kwds):
        _shape = tuple([s+1 for s in shape])
        mpi = mpiinterfaces.EvenCart(
            _shape, 
            mpi_comm=mpi_comm, 
            ghost_points=ghost_points
            )
        if name is None:
            name = "UniformCart%s%s%s"%(shape, bounds, comparison)
        super(UniformCart, self).__init__(
            shape, bounds, 
            name=name, comparison=comparison,
            mpi=mpi, *args, **kwds
            ) 
        _axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            )
            for i in range(len(self.bounds))
            ]
        self._step_sizes = [axis[1]-axis[0] for axis in _axes]
        self._axes = [axis[self.mpi.subdomain] for axis in _axes]

    @property
    def axes(self):
        return self._axes

    @property
    def full_grid(self):
        if self.mpi is None:
            return self
        return UniformCart(
            self.shape, self.bounds, 
            comparison=self.comparison, name=self.name
            )
        
    @property
    def step_sizes(self):
        return self._step_sizes

class GeneralGrid(ABCGrid):

    def __init__(self, 
            shape, 
            bounds, 
            periods,
            #mpi_comm=None, 
            comparison=None, 
            name=None, 
            #ghost_points=1,
            *args, **kwds):
        _shape = []
        for i,p in enumerate(periods):
            if p:
                _shape.append(shape[i])
            else:
                shape.append(shape[i]+1)
        _shape = tuple(_shape)
        mpi=None
        #mpi = mpiinterfaces.EvenCart(
            #_shape, 
            #mpi_comm=mpi_comm, 
            #ghost_points=ghost_points
            #)
        if name is None:
            name = "<GeneralGrid shape=%s, bounds=%s,periods=%s, comparison=%s>"%(
                shape, 
                bounds, 
                periods,
                comparison)
        super(UniformCart, self).__init__(
            shape, bounds, 
            name=name, 
            comparison=comparison,
            mpi=mpi, 
            *args, 
            **kwds
            ) 
        _axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            )
            for i in range(len(self.bounds))
            ]
        self._step_sizes = [axis[1]-axis[0] for axis in _axes]
        #self._axes = [axis[self.mpi.subdomain] for axis in _axes]

    @property
    def axes(self):
        return self._axes

    @property
    def full_grid(self):
        if self.mpi is None:
            return self
        return UniformCart(
            self.shape, self.bounds, 
            comparison=self.comparison, name=self.name
            )
        
    @property
    def step_sizes(self):
        return self._step_sizes
