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


from solvers import Solver
from system import System


class Grid(object):
    """docstring for Grid"""
    def __init__(self, shape):
        self.dim = len(shape)
        self.shape = shape
    
    
    


#############################################################################
class IBVP:
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    def __init__(self, sol, eqn, grid = None, action = None, maxIteration = 1000):
        sol.useSystem(eqn)
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
        
    
    
    def _ic(self,t0):
        return self.theSystem.initialValues(t0, grid = self.theGrid)
    
    
    def run(self, tstart, tstop = float('inf')):
        """Go for it"""
        t = tstart
        u = self._ic(tstart)
        dt = self.theSystem.timestep(u)
        advance = self.theSolver.advance
        while(True):
            if (self.iteration > self.maxIteration):
                print("Maximum number of iterations exceeded\n")
                break
            
            if (t >= tstop):
                break
            
            if self.theActions is not None:
                for action in self.theActions:
                    action(self.iteration, u)
            
            t, u = advance(t, u, dt)
            self.iteration+=1
        print("Finished.-")






#############################################################################
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
