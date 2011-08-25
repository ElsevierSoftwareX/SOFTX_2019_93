#!/usr/bin/env python
# encoding: utf-8 
"""
ibvp.py

Created by Jörg Frauendiener on 2010-12-24.
Copyright (c) 2010 University of Otago. All rights reserved.

Editted by Ben Whale and George Doulis.
"""
import sys
import os
import unittest
import math
import numpy as np
import logging
from mpi4py import MPI

from solvers import Solver
from system import System
import tslices

class IBVP:
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    
    def __init__(self, sol, eqn, grid = None, action = None,\
        maxIteration = 10000,log = None):
        sol.useSystem(eqn)
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
        self.log = log.getChild(".IBVP")
        self.mpicomm = MPI.COMM_WORLD
        self.mpirank = self.mpicomm.rank
        self.mpisize = self.mpicomm.size
        np.set_printoptions(precision = 17,suppress = True)
        
    def _ic(self,t0):
        """
        We check how many processors are being used and divide the computational
        domain amongst them. We include ghoast points, depending on the size of
        the differential operator stencil (there is an implicit assumption that
        we are using FD). We check that the computational domain is large enough
        to include the whole boundary region of the operator. This removes the
        need to include the case when this is not true, which would complicate
        things significantly. We also set the variable self.boundary
        which will be used to indicate if the boundary FD should be applied.
        """
        self.log.debug("Setting up initial data...")
        #If no mpi
        if self.mpisize == 1:
            ic = self.theSystem.initialValues(t0, self.theGrid.domain(t0))
            return ic 
        else:
                
            #Get the number of ghoast points required.
            #We assume here that the stencil length is odd
            tslices.num_ghost_points = self.theSystem.D.A.shape[0]//2
            self.log.debug("Number of ghoast points is %i"%tslices.num_ghost_points)
            
            ic = self.theSystem.initialValues(t0, self.theGrid.domain(t0))
        
            #If mpi then divide up domain and return the correct bit
            s,e = self._domain_indices(ic.fields.shape[1])
            self.log.debug("Subdomain to be processed: start index = %i, end index = %i"%\
                (s,e))
            
            #Reconstruct the timeslice and set the boundary ID indicator
            ic = self._subdomain(ic,s,e)   
        self.log.debug("IC is %s"%(repr(ic)))
        self.log.debug("Intial data constructed.")
        return ic
    
    def _subdomain(self,tslice,s,e):
        self.log.debug("Creating sub tslice")
        self.log.debug("Oringal tslice is %s"%str(self))
        num_ghost_points = self.theSystem.D.A.shape[0]//2
        if self.mpisize == 1:
            return
        if self.mpirank == 0:
            tslices.boundary = tslices.LEFT
            tslice = tslices.timeslice(\
                tslice.fields[:,s:e+tslice.num_ghost_points],\
                tslice.domain[s:e+tslice.num_ghost_points],
                tslice.time\
                )
        elif self.mpirank == self.mpisize-1:
            tslices.boundary = tslices.RIGHT
            tslice = tslices.timeslice(\
                tslice.fields[:,s-tslice.num_ghost_points:e],\
                tslice.domain[s-tslice.num_ghost_points:e],\
                tslice.time\
                )
        else:
            tslices.boundary = tslices.CENTRE
            tslice = tslices.timeslice(\
                tslice.fields[:,s-tslice.num_ghost_points:e+tslice.num_ghost_points],\
                tslice.domain[s-tslice.num_ghost_points:e+tslice.num_ghost_points],\
                tslice.time\
                )
        self.log.debug("tslice.boundary is set to %i"%tslice.boundary)
        self.log.debug("Sub tslice is %s"%repr(tslice))
        return tslice
    
    def _domain_indices(self,array_length):
        self.log.debug("Calculated start and end indices for sub timeslice")
        self.log.debug("Array_length is %i"%array_length)
        q,r = divmod(array_length, self.mpisize)
        self.log.debug("q = %i, r = %i"%(q,r))
        s = self.mpirank*q + min(self.mpirank,r)
        e = s + q
        if self.mpirank < r:
            self.log.debug("Self.mpirank > r so we add one to end point")
            e = e + 1
        self.log.debug("Start index = %i, End index = %i"%(s,e))
        return s,e
    
    def run(self, tstart, tstop = float('inf')):
        """Go for it"""
        t = tstart
        self.log.info("starting =============================")
        #Get initial data and configure timeslices for multiple processors
        u = self._ic(t)
        dt = self.theSystem.timestep(u)
        self.log.info("Running system %s"%str(self.theSystem))
        self.log.info("Grid = %s"%str(self.theGrid))
        self.log.info("Using timestep dt=%s"%repr(dt))
        self.log.debug("dt = %.53f"%dt)
        self.log.info("Using spacestep dx=%s"%repr(u.dx))
        self.log.debug("dx = %.53f"%u.dx)
        advance = self.theSolver.advance
        validate = self.theGrid.validate
        while(True):
            if (self.iteration > self.maxIteration):
                self.log.info("Maximum number of iterations exceeded")
                break
            
            if (math.fabs(t-tstop) < dt/2):
                self.log.info("Maximum time reached at %f for iterations: %d"%\
                    (t, self.iteration))
                break
            
            if self.theActions is not None:
                tslice = u.collect_data()
                for action in self.theActions:
                    if tslice is not None:
                        self.log.info("Running action %s at iteration %i"%(str(action),\
                            self.iteration))
                        action(self.iteration, tslice)
                    
            try:
                #t, u = advance(t, validate(u,t+dt), dt,self.boundary)
                self.log.debug("About to advance for iteration = %i"%self.iteration)
                t, u = advance(t, u, dt)
                self.log.info("time slice after advance = %s"%repr(u))
                self.iteration+=1
                 
            except ValueError as error:
                if error.__str__() == 'matrices are not aligned' or\
                    error.__str__() == \
                        'domain too small for application of diffop':
                    self.log.info("Limit of computational domain reached at"\
                        "time: %f and iteration: %i"%(t,self.iteration))
                self.log.exception(repr(error))
                raise error
            except IndexError as error:
                if error.__str__() == 'index out of bounds':
                    self.log.info("Limit of computational domain reached at"\
                        "time: %f and iteration: %i"%(t,self.iteration))
                self.log.exception(repr(error))
                raise error
        self.log.info("Finished computation")
        self.log.info("stopped =============================")
        return u


###############################################################################
class IBVPTests(unittest.TestCase):
    def setUp(self):
        eqn = System()
        euler = Solver()
        self.ode = IBVP(euler,eqn)
    
    def test_run(self):
        self.ode.run(1.0)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
