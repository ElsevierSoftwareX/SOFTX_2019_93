#!/usr/bin/env python
# encoding: utf-8
"""
ibvp.py

Created by Jörg Frauendiener on 2010-12-24.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

import sys
import os
import unittest
import math
import numpy as np

from solvers import Solver
from system import System


class Grid(object):
    """docstring for Grid"""
    def __init__(self, shape):
        self.dim = len(shape)
        self.shape = shape
        self.name = None
        self.grid = None
        
    def getX1(self):
        return self.x1
        
    def __str__(self):
        return self.name
    
class Interval(Grid):
    """docstring for Interval"""
    def __init__(self, shape, bounds):
        super(Interval, self).__init__(shape)
        self.x1 = np.linspace(bounds[0],bounds[1],shape[0]+1)
        self.grid = self.x1
        self.name = "Interval grid, %s" % str(self.grid.shape)

class Periodic(Grid):
    """docstring for Loop"""
    def __init__(self, shape, bounds):
        super(Periodic, self).__init__(shape)
        x = np.linspace(bounds[0],bounds[1],shape[0]+1)
        self.x1 = x[:-1]
        self.grid = self.x1
        self.name = "Periodic grid, %s" % (str(self.grid.shape))
        

#############################################################################
class IBVP:
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    def __init__(self, sol, eqn, grid = None, action = None,\
        maxIteration = 10000):
        sol.useSystem(eqn)
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
        
    def _ic(self,t0):
        print ("Setting up initial data")
        return self.theSystem.initialValues(t0, grid = self.theGrid)
        print "...Done.-\n"
    
    def run(self, tstart, tstop = float('inf')):
        """Go for it"""
        t = tstart
        u = self._ic(tstart)
        dt = self.theSystem.timestep(u)
        print "Running system %s"%str(self.theSystem)
        print "Grid = %s"%str(self.theGrid)
        print("Using timestep dt=%f"%(dt,))
        print("Using spacestep dx=%f"%(u.dx,))
        advance = self.theSolver.advance
        while(True):
            if (self.iteration > self.maxIteration):
                print("Maximum number of iterations exceeded\n")
                break
            
            if (math.fabs(t-tstop) < dt/2):
                print("Time: %f,  Iterations: %d" % (t, self.iteration))
                break
            
            if self.theActions is not None:
                for action in self.theActions:
                    action(self.iteration, u)
            
            t, u = advance(t, u, dt)
            self.iteration+=1
        print("Finished.-\n\n")
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
