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

#import warnings

#warnings.simplefilter('error', RuntimeWarning)

class IBVP:
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    
    def __init__(self, sol, eqn, grid, action = [], 
        maxIteration = 10000, minTimestep = 1e-8
        ): 
        sol.use_system(eqn)
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
        self.log = logging.getLogger("IBVP")
        self.minTimestep = minTimestep
             
    def run(self, tstart, tstop = float('inf'), thits = None):
        """Go for it"""
        if thits is None:
            thits = []
        if tstop not in thits:
            thits += [tstop]
        thits = sorted(thits)
        thits.reverse()
        tstop = thits.pop()
        t = tstart
        #Get initial data and configure timeslices for multiple processors
        u = self.theSystem.initial_data(t, self.theGrid)
        self.log.info("Running system %s"%str(self.theSystem))
        self.log.info("Grid = %s"%str(self.theGrid))
        self.log.info("Stepsizes = %s"%repr(u.domain.step_sizes))
        if __debug__:    
            self.log.debug("self.actions is %s"%repr(self.theActions))
            self.log.debug("Initial data is = %s"%repr(u))
        advance = self.theSolver.advance
        computation_valid = True
        while(computation_valid and t < tstop):
            if __debug__:
                self.log.debug("Beginning new iteration")

            if self.iteration > self.maxIteration:
                self.log.warning("Maximum number of iterations exceeded")
                break
           
            dt = self.theSystem.timestep(u)
            #import warnings

            #with warnings.catch_warnings(record=True) as w:
                #dt = self.theSystem.timestep(u)
                #if len(w)==1 and issubclass(w[-1].category, RuntimeWarning):
                    #print t
                    #print u.time
                    #print w

            if dt < self.minTimestep:
                self.log.error(
                    'Exiting computation: timestep too small dt = %.15f'%dt
                )
                break
            
            timeleft = tstop - t
            if timeleft < dt:
                dt = timeleft
                if not thits:
                    self.log.warning(
                            "Final time step: adjusting to dt = %.15f" % dt
                        )
                    computation_valid = False
                else:
                    self.log.warning(
                        "Forcing evaluation at time %f"%tstop
                    )
                    tstop = thits.pop()
            
            if __debug__: 
                self.log.debug("Using timestep dt = %f"%dt)
           
            self._run_actions(t, u)

            if __debug__:
                self.log.debug("About to advance for iteration = %i"%
                    self.iteration)
            try:
                t, u = advance(t, u, dt)
            except OverflowError as e:
                print "Overflow error({0}): {1}".format(e.errno, e.strerror)
                computation_valid = False
            u.domain.barrier()
            self.iteration+=1
            if __debug__:
                self.log.debug("time slice after advance = %s"%repr(u))
        # end (while)
        self._run_actions(t, u)
        u.domain.barrier()
        self.log.info(
            "Finished computation at time %f for iteration %i"%
            (t,self.iteration)
            )
        return u

    def _run_actions(self, t, u):
        #Ideally u.collect_data() should only be executed if there
        #actions that will run. because of single process access to
        #actions this causes an issue.
        #Some thought is required to fix this problem.
        tslice = u.collect_data()
        if tslice is not None:
            if __debug__:
                self.log.debug(
                    "tslice is not None. Computing if actions will run"
                    )
            actions_do_actions = [
                action.will_run(self.iteration, tslice) 
                for action in self.theActions
                ]
            if any(actions_do_actions):
                if __debug__:
                    self.log.debug("Some actions will run")
                for i, action in enumerate(self.theActions):
                    if actions_do_actions[i]:
                        if __debug__:
                            self.log.debug(
                                "Running action %s at iteration %i"%\
                                 (str(action), self.iteration)
                                 )
                        action(self.iteration, tslice)
                if __debug__:
                    self.log.debug("All actions done")
