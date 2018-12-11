#!/usr/bin/env python
# encoding: utf-8
"""
ibvp.py

Created by Jörg Frauendiener on 2010-12-24, further edits by Ben Whale, George
Doulis and Jörg Frauendiener since then.

This is the class that handles the iterative step and actions of the
simulation. The name ibvp comes from the original design of this class as the
iterative step in an initial boundary value problem solver. However, this class
is general enough to handle numerical techniques that have, at their highest
level, some iterative method.

Classes:

IBVP - a class that handles each iterative step in the numerical method being
       used and interleaves this with calls to a list of actions.
"""
import logging

class IBVP:
    """This class manages the interative process of simulations and interleaves
    interative steps with calls to a list of actions. Information about events
    during the simulation are writen to logging.getLogger("IBVP").

    """
    theActions  = None
    theGrid = None
    theSolver  = None
    iteration = 0
    maxIteration = None
    
    
    def __init__(self, sol, eqn, grid, action = [], 
        maxIteration = 10000, minTimestep = 1e-8
        ): 
        """IBVP constructor.

        Positional Arguments:

        sol - the solver. An instance of solver.Solver
        eqn - the system. An instance of system.System
        grid - the grid. An instance of grid.Grid
        action - a list of instances of actions.Prototype

        Keyword Arguments:

        maxIteraction - an int giving the maximum allowed number of iterations.
                        The default is 10,000
        minTimestep - a number giving the smallest allowed timestep. The
                      default is 1e-8

        Returns:
        no returned object

        """
        self.theSolver = sol
        self.theSystem = eqn
        self.maxIteration = maxIteration
        self.theGrid = grid
        self.theActions = action
        self.log = logging.getLogger("IBVP")
        self.minTimestep = minTimestep
             
    def run(self, tstart, tstop = float('inf'), thits = None):
        """Go for it! Starts the simulation.
        
        Positional Arguments:
        
        tstart - a number giving the time at the initial interation
        
        Keyword Arguments:

        tstop - a number giving the time to finish the simulation at. The
                default is currently positive infinty
        thits - a list of numbers that the simulation is forced to hit exactly.
                The default is None.

        """
        # Set up thits
        if thits is None:
            thits = []
        if tstop not in thits:
            thits += [tstop]
        # Order the list of times, and ensure that they are popped from smallest
        # to largest.
        thits = sorted(thits)
        thits.reverse()
        tstop = thits.pop()
        # Set start time.
        t = tstart
        # Get initial data and configure timeslices for multiple processors.
        u = self.theSystem.initial_data(t, self.theGrid)
        self.log.info("Running system %s"%str(self.theSystem))
        self.log.info("Grid = %s"%str(self.theGrid))
        self.log.info("Stepsizes = %s"%repr(u.domain.step_sizes))
        if __debug__:    
            self.log.debug("self.actions is %s"%repr(self.theActions))
            self.log.debug("Initial data is = %s"%repr(u))
        # Run the actions.
        self._run_actions(t, u)
        advance = self.theSolver.advance
        computation_valid = True
        while(computation_valid and t < tstop):
            if __debug__:
                self.log.debug("Beginning new iteration")

            # Check against maxIteration
            if self.iteration > self.maxIteration:
                self.log.warning("Maximum number of iterations exceeded")
                break
           
            dt = self.theSystem.timestep(u)

            # Check dt for size
            if dt < self.minTimestep:
                self.log.error(
                    'Exiting computation: timestep too small dt = %.15f'%dt
                )
                break
            
            # Check if dt needs to change in order to hit the next thits value.
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
           

            if __debug__:
                self.log.debug(
                    "About to advance for iteration = %i"%self.iteration
                )
            try:
                # Advance Australia Fair.
                #
                # Australians all let us rejoice,
                # For we are young and free;
                # We've golden soil and wealth for toil;
                # Our home is girt by sea;
                # Our land abounds in nature's gifts
                # Of beauty rich and rare;
                # In history's page, let every stage
                # Advance Australia Fair.
                # In joyful strains then let us sing,
                # Advance Australia Fair.
                #
                # Beneath our radiant Southern Cross
                # We'll toil with hearts and hands;
                # To make this Commonwealth of ours
                # Renowned of all the lands;
                # For those who've come across the seas
                # We've boundless plains to share;
                # With courage let us all combine
                # To Advance Australia Fair.
                # In joyful strains then let us sing,
                # Advance Australia Fair.

                t, u = advance(t, u, dt)
            except OverflowError as e:
                # OverflowErrors arn't always appropirately handled. So we
                # do it ourselves here.
                print "Overflow error({0}): {1}".format(e.errno, e.strerror)
                computation_valid = False
            # If we're using an mpi enable grid, this ensures that all
            # processes have gotten to the same point before continuing the
            # simulation.
            u.barrier()
            # On to the next iteration.
            self.iteration+=1
            if __debug__:
                self.log.debug("time slice after advance = %s"%repr(u))
        # end (while)
        # Run the actions once more before exiting.
        self._run_actions(t, u)

        # This statement might be unnecessary. In principle it ensures that all
        # processes are about to complete the current simulation before exit
        # occurs.
        u.barrier()
        self.log.info(
            "Finished computation at time %f for iteration %i"%
            (t,self.iteration)
            )
        return u

    def _run_actions(self, t, u):
        # Ideally u.collect_data() should only be executed if there
        # actions that will run. because of single process access to
        # actions this causes an issue.
        # Some thought is required to fix this problem.
        if __debug__:
            self.log.debug("Running actions")
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
                

