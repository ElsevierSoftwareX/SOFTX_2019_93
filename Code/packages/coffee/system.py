#!/usr/bin/env python
# encoding: utf-8 
"""
system.py

Created by Jörg Frauendiener on 2010-12-26.
Modified by Ben Whale since then.

The abstract base class (abc) for System objects. This class serves to specify
the interface that System classes are assumed to have. These assumptions are
made by virtually all classes that call the system. For example, ibvp.IBVP,
subclasses of solver.Solver, almost all actions.hdf_actions.SimOutType
subclasses.

While you don't need to actually implement all the methods below, it really is
recomended. You may get errors about the methods missing as a consequence.

system.System has abc.ABCMeta as a meta class. This serves as a warning to
really think about what you are doing if you don't implement all methods below.

Classes:
System - an abc class that specifies the interface for subclasses of System
         that are assumed by all other classes in the coffee package.
"""

import abc
from abc import ABCMeta

#############################################################################
class System(object):
    """The System class that specifies the interface for all other system
    classes. You don't need to go home, but you can't stay here. *ahem* Sorry.
    I mean you don't need to implement everything below, but you can expect
    errors if you don't.

    """
    __metaclass__ = ABCMeta

    def timestep(self, tslice):
        """Returns a number giving the time step for the next iteration.

        Positional Arguments:
        tslice - a time slice with the current data

        Returns - a number giving the dt for which the next data will be
                  calculated in the next iteration.

        """
        return NotImplementedError("You need to implement this function")
    
    def evaluate(self, t, tslice):
        """Returns the data needed by the solver.Solver subclass needed to
        calcualte the values of the functions at the next time step.

        For example, in a IBVP problem using an RK sover this method will
        return the values of the derivatives of the functions being evolved.

        Yes I know it all sounds a bit vague. But really, this method is the
        heart and soul of coffee. You need to know what you're doing, how the
        solver is implemented and what data the system is meant to pass to the
        solver.

        http://bit.ly/ps1RIe

        Positional Arguments:
        t - the time at which the 'derivatives' need to be calculated. NOTE
            THAT THIS TIME MAY BE DIFFERENT FROM THE TIME STORED IN u. The time
            stored in u is the time for the values of the functions stored in
            u, not the time at which the derivatives of the functions stored in
            u are needed. In the case of RK methods t and u.time can be
            different. YOU HAVE BEEN WARNED.

            http://www.youtube.com/watch?v=OYj_krIHoLs
        tslice - a tslice.TimeSlice object containing the data from which the
                 'evaluate' method is meant to calculate data from.

        Returns - data needed for the solver.Solver subclass to calculate the
                  values of the functions being evolved at the next time slice.

        """
        return NotImplementedError("You need to implement this function")
        
    def initialValues(self, t, grid):
        """Returns initial data for the simulation.

        Arguments:
        t - the time at which the initial data is to be calcualted.
        grid - the grid over which the data is to be calculated.

        Returns - the initial values for the functions to be evolved.

        """
        return NotImplementedError("You need to implement this function")
        
    def left(self, t):
        """Returns the boundary data on the 'left' boundary.

        Yes this also makes the assumption that left is well defined. See the
        documentation for free_data.FreeData. To be honest I'm not sure that
        this method needs to be defined here. After all: does any class other
        than the system class need to access this method? If not there there is
        no need for it.

        http://bit.ly/yb10oP

        Yes there is no grid object passed as an argument - deal with it. In
        other news, this will very likely change at some point.

        Arguments:
        t - the time for which the left data needs to be calculated.

        Returns - boundary data.

        """
        return NotImplementedError("You need to implement this function")

    def right(self, t):
        """Returns the boundary data on the 'right' boundary.

        See the documentation for the function system.System.left

        Arguements:
        t - the time at which the right boundary data must be calculated

        Returns - boundary data

        """
        return NotImplementedError("You need to implement this function")
