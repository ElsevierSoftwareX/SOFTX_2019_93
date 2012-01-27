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

from solvers import Solver
from system import System

class IBVP:
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    def __init__(self, sol, eqn, grid = None, action = None,\
        maxIteration = 10000,debug_parent = "main"):
        sol.useSystem(eqn)
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
        self.log = logging.getLogger(debug_parent+".IBVP")
        
    def _ic(self,t0):
        self.log.debug("Setting up initial data...")
        return self.theSystem.initialValues(t0, self.theGrid.domain(t0))
        self.log.debug("Intial data constructed.")
    
    def run(self, tstart, tstop = float('inf')):
        """Go for it"""
        t = tstart
        self.log.info("starting =============================")
        u = self._ic(tstart)
        dt = self.theSystem.timestep(u)
        self.log.info("Running system %s"%str(self.theSystem))
        self.log.info("Grid = %s"%str(self.theGrid))
        self.log.info("Using timestep dt=%f"%(dt,))
        self.log.info("Using spacestep dx=%f"%(u.dx,))
        advance = self.theSolver.advance
        validate = self.theGrid.validate
        computation_valid = True
        while(computation_valid):
            if (self.iteration > self.maxIteration):
                self.log.info("Maximum number of iterations exceeded")
                computation_valid = False
            
            if (math.fabs(t-tstop) < dt/2):
                self.log.info("Maximum time reached at %f for iterations: %d"%\
                    (t, self.iteration))
                computation_valid = False
            
            if self.theActions is not None:
                for action in self.theActions:
                    try:
                        action(self.iteration, u)
                    except Exception as error:
                        if error.__str__() == "Function values are above the cutoff":
                            self.log.info("Function values are above the cutoff")
                        self.log.exception(repr(error))
                        computation_valid = False
            try:
                t, u = advance(t, validate(u,t+dt), dt)
            except ValueError as error:
                if error.__str__() == 'matrices are not aligned' or\
                    error.__str__() == \
                        'domain too small for application of diffop':
                    self.log.info("Limit of computational domain reached at"\
                        "time: %f and iteration: %i"%(t,self.iteration))
                self.log.exception(repr(error))
                computation_valid = False
            except IndexError as error:
                if error.__str__() == 'index out of bounds':
                    self.log.info("Limit of computational domain reached at"\
                        "time: %f and iteration: %i"%(t,self.iteration))
                self.log.exception(repr(error))
                computation_valid = False
            self.iteration+=1
        self.log.info("Finished computation at time %f for iteration %i"%(t,self.iteration))
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
