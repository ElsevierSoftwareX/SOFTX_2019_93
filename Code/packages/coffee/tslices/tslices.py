#!/usr/bin/env python
# encoding: utf-8 
"""
tslices.py

Created by Jörg Frauendiener on 2010-11-17.
Additional development by Ben Whale.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

import sys
import os
import unittest
import numpy as np
import logging
from mpi4py import MPI


LEFT = -1
CENTRE = 0
RIGHT = 1
log = None
num_ghost_points = None
boundary = None

class timeslice(np.ndarray):
    
    def __new__(cls, data, domain, time):
        obj = np.asarray(data).view(cls)
        obj.log = logging.getLogger('timeslice')
        obj.domain = domain
        obj.time = time        
        obj.mpicomm = MPI.COMM_WORLD
        obj.mpirank = obj.mpicomm.rank
        obj.mpisize = obj.mpicomm.size
        return obj
        
    def __array_finalize__(self,obj):
        if obj is None: return obj
        self.domain = getattr(obj, 'domain', None)
        self.time = getattr(obj, 'time', None)
        self.mpicomm = getattr(obj, 'mpicomm', None)
        self.mpirank = getattr(obj, 'mpirank', None)
        self.mpisize = getattr(obj, 'mpisize', None)
        self.log = getattr(obj, 'log', logging.getLogger('timeslice'))
    
    @property
    def numFields(self):
        return self.shape[0]
    
    @property
    def dx(self):
        return self.step_sizes
      
    @property
    def step_sizes(self):
        return self.domain.step_sizes
        
    @property
    def fields(self):
        return np.asarray(self)

    @property        
    def x(self):
        return self.domain

    def communicate(self):
        if self.mpisize == 1:
#            if __debug__:
#                self.log.debug("no data swap")
            return
        self.domain.send(self)
        self.domain.recv(self)
#        if __debug__:
#            self.log.debug("time slice after data swapping = %s"%repr(self))
        return
        
    def collect_data(self):
        if self.mpisize == 1:
            return self
        data,domain = self.domain.collect_data(self)
        if data is None: return None
        return timeslice(data,domain,self.time)
