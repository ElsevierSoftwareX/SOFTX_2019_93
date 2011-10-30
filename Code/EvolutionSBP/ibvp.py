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
import logging

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
        if log is not None: 
            self.log = log.getChild(".IBVP")
        else:
            self.log = logging.getLogger("IBVP")
        self.mpicomm = MPI.COMM_WORLD
        self.mpirank = self.mpicomm.rank
        self.mpisize = self.mpicomm.size
        np.set_printoptions(precision = 17,suppress = True)
             
    def run(self, tstart, tstop = float('inf')):
        """Go for it"""
        t = tstart
        self.log.info("starting =============================")
        #Get initial data and configure timeslices for multiple processors
        u = self.theSystem.initialValues(t, self.theGrid)
        dt = self.theSystem.timestep(u)
        self.log.info("Running system %s"%str(self.theSystem))
        self.log.info("Using timestep dt=%s"%repr(dt))
        self.log.info("Using spacestep dx=%s"%repr(u.dx))
        if __debug__:
            self.log.debug("Initial data is = %s"%repr(u))
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
                actions_do_actions = [action.will_run(self.iteration,u) \
                    for action in self.theActions]
                if any(actions_do_actions):
                    tslice = u.collect_data()
                    for action in self.theActions:
                        if action.will_run(self.iteration,u) and tslice is not None:
                            if __debug__:
                                self.log.debug("Running action %s at iteration %i"%(str(action),\
                                    self.iteration))
                            action(self.iteration, tslice)
                        
            try:
                #t, u = advance(t, validate(u,t+dt), dt,self.boundary)
                if __debug__:
                    self.log.debug("About to advance for iteration = %i"%self.iteration)
                t, u = advance(t, u, dt)
                if __debug__:
                    self.log.debug("time slice after advance = %s"%repr(u))
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
