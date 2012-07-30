#!/usr/bin/env python
# encoding: utf-8
"""
ibvp.py

Created by Jörg Frauendiener on 2010-12-24.
Copyright (c) 2010 University of Otago. All rights reserved.

Edited by Ben Whale and George Doulis.
"""
import math
import logging

class IBVP:
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    
    def __init__(self, sol, eqn, grid, action = [], 
        maxIteration = 10000, minTimestep = 1e-8): #, CFL = 1.0):
        sol.use_system(eqn)
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
#        self.cfl = CFL
        self.log = logging.getLogger("IBVP")
        self.minTimestep = minTimestep
             
    def run(self, tstart, tstop = float('inf')):
        """Go for it"""
        t = tstart
        #Get initial data and configure timeslices for multiple processors
        u = self.theSystem.initial_data(t, self.theGrid)
        self.log.info("Running system %s"%str(self.theSystem))
        self.log.info("Grid = %s"%str(self.theGrid))
        self.log.info("Using spacestep dx=%s"%repr(u.dx))
        if __debug__:    
            self.log.debug("Initial data is = %s"%repr(u))
        advance = self.theSolver.advance
        validate = self.theGrid.validate
        computation_valid = True
        
        while(computation_valid):
            
            if (self.iteration > self.maxIteration):
                self.log.warning("Maximum number of iterations exceeded")
                break
            
            dt = self.theSystem.timestep(u)
            timeleft = tstop - t
            if timeleft < dt:
                dt = timeleft
                self.log.warning(
                    "Final time step: adjusting to dt = %.15f" % dt
                    )
                computation_valid = False
            
            if __debug__: 
                self.log.debug("Using timestep dt = %f"%dt)
            
            if dt < self.minTimestep:
                self.log.error(
                    'Exiting computation: timestep too small dt = %.15f'%dt
                )
                break
             
            actions_do_actions = [action.will_run(self.iteration,u) 
                for action in self.theActions]
            if any(actions_do_actions):
                tslice = u.collect_data()
                for i, action in enumerate(self.theActions):
                    if actions_do_actions[i]:
                        if __debug__:
                            self.log.debug(
                                "Running action %s at iteration %i"%\
                                 (str(action), self.iteration)
                                 )
                        action(self.iteration, tslice)

            if __debug__:
                self.log.debug("About to advance for iteration = %i"%
                    self.iteration)
            t, u = advance(t, u, dt)
            self.iteration+=1
            if __debug__:
                self.log.debug("time slice after advance = %s"%repr(u))
        # end (while)
        
        # do the last round of actions
        actions_do_actions = [action.will_run(self.iteration,u) 
            for action in self.theActions]
        if any(actions_do_actions):
            tslice = u.collect_data()
            for i, action in enumerate(self.theActions):
                if actions_do_actions[i]:
                    if __debug__:
                        self.log.debug(
                            "Running action %s at iteration %i"%\
                             (str(action), self.iteration)
                             )
                    action(self.iteration, tslice)

            
        self.log.info("Finished computation at time %f for iteration %i"%(t,self.iteration))
        return u
