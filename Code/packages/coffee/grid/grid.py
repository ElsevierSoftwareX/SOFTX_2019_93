#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import math
import numpy as np
import abc
import logging

################################################################################
# Base Grid class
################################################################################

class Grid(object):
    """The abstract base class for Grid objects."""
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, 
        shape, bounds, name = "Grid", comparison = None
        mpi = None):
        self.mpi = mpi
        self.dim = len(shape)
        self.name = name
        self.log = logging.getLogger(name)
        self.comparison = comparison
        self.shape = shape
        self.bounds = bounds
        
    def communicate(self, data):
        if self.mpi is None:
            return
        return self.mpi.communicate(self, data)

    #def sendrecv(self, data):
        #if self.mpi is None:
            #return
        #self.mpi.sendrecv(data)

    #def send(self, data):
        #if self.mpi is None:
            #return 
        #self.mpi.send(data)
        
    #def recv(self,data):
        #if self.mpi is None:
            #return 
        #self.mpi.recv(data)
        #return data
        
    def collect_data(self, data):
        if self.mpi is None:
            return 
        return self.mpi.collect_data(self, data)
    
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

class UniformCart(Grid):
    """A Grid object to represent an ND interval of coordinates"""
    
    def __init__(self, shape, *args, mpi_comm=None, **kwds):
        mpi = mpiinterfaces.EvenCart(shape, mpi_comm)
        name = "UniformCart%s%s%s"%(shape,bounds,comparison)
        super(UniformCart, self).__init__(
            shape, *args,
            name=name, mpi=mpi, **kwds
            ) 
        axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            )
            for i in range(len(self.bounds))
            ]
        self.step_sizes = [axis[1]-axis[0] for axis in axes]
        self.axes = [axis[self.mpi.subdomain] for axis in axes]

    #@property
    #def axes(self):
        #axes = [
            #np.linspace(
                #self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            #)[self.mpi.subdomian]
            #for i in range(len(self.bounds))
            #]
        #return axes
        
    #@property
    #def step_sizes(self):
        #axes = [
            #np.linspace(
                #self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            #)
            #for i in range(len(self.bounds))
            #]
        #step_sizes = [axis[1]-axis[0] for axis in axes]
        #return step_sizes
         
class S2(Grid):
    """A Grid object representing the sphere. Note that
    theta is in [0, pi) and phi is in [0, 2*pi)."""
    
    def __init__(self, shape, *args, **kwds):
        if len(shape) != 2:
            return Exception("Shape length is not 2")
        name = "S2%s%s"%(shape, comparison)
        super(S2, self).__init__(shape, 
            [[0, math.pi], [0, 2*math.pi]],
            *args,
            name=name, 
            **kwds
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
