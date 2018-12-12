#!/usr/bin/env python
# encoding: utf-8 
"""
solvers.py

Created by Jörg Frauendiener on 2010-12-26.
Editted by Ben Whale since 2011.

This module contains the abstract base class for solvers as well as several
default implementations.

This default implementations all assume that tslice.TimeSlice objects can be
operated on by numerically (e.g. addition, multiplication, subtraction).

Classes:
ABCSolver - the abstract base class (abc) for solver object. This object
            specifies the assumed interface for solver object. 
            They principally interact with the ibvp class.

Euler - an implementation of the first order Euler method

RungeKutta4 - an implementation of the 4th order Runge Kutta method.

RungeKutta4Dirichlet - an implementation of the 4th order Runge Kutta method
                       that makes calls to system.dirichlet_boundary for 
                       intermediate steps.

rk45 - an adaptive RungeKutta 4th order routine. This class made require
       debugging.

"""

import math
import logging
import abc


################################################################################
# Solver Abstract Base Class
################################################################################
class ABCSolver(object):
    """The ABSolver abstract base class.

    This class provides the interface that is assumed of all other solvers.
    These methods are called in the ibvp.IBVP class.

    Methods:
    name - returns a string representation of the object
    advance - called by ibvp.IBVP to advance to the next time step


    """
    def __init__(self, system, *args, **kwds):
        """Returns an instance of ABCSolver.

        This method should be called by any subclasses. It defines the system
        atribute and sets up a logging object.

        Arguments:
        system - the system being used.

        """
        self.system = system
        super(ABCSolver, self).__init__(**kwds)
        self.log = logging.getLogger(self.name)
 
    @abc.abstractproperty
    def name(self):
        """This should return a name associated to the subclass. 
        
        The name will need to be defined in the subclasses constructor.

        Returns - a string name for the class. This defaults to ABCSolver.

        """
        return "ABCSolver"

 
    @abc.abstractmethod
    def advance(self, t, u, dt): 
        """Returns a tslice.TimeSlice containing data at time t + dt.

        Arguments:
        t - the current time. This should be equal to u.time. So... maybe we
            should do something about this? Seems unnecessary.
        u - the current tslice.TimeSlice
        dt - the step in time required

        Returns - a tslice.TimeSlice object containing the data at time t+dt.

        """
        raise NotImplementedError("This method needs to be implemented")
    
    def __repr__(self):
        return "<%s: system = %s>"%(self.name, self.system)

################################################################################
# First order methods
################################################################################
class Euler(ABCSolver):
    """An implementation of the first order Euler method.

    Methods:
    advance - See the documentation for ABCSolver

    """
    name = 'Euler'
  
    def advance(self, t, u, dt):
        """Returns a tslice.TimeSlice object containing the evolved data, via
        the first order Euler method, at time t+dt.

        Returns - please read above... No.. Don't make me have to write it out
                  again... Oh god why are you doing this to me? *sniff*...
                  ok, what ever you say.

                  http://www.youtube.com/watch?v=SuJtAoADesE

                  Returns a tslice.TimeSlice object containing the evolved data, via
                  the first order Euler method, at time t+dt.

        """
        du = self.system.evaluate(t, u)
        r_time = t + dt        
        r_slice = u + dt*du
        r_slice.time = r_time        
        return (r_time, r_slice)

################################################################################
# Fourth order methods
################################################################################
class RungeKutta4(ABCSolver):
    """docstring for RungeKutta4"""

    name = "RK4"

    def __init__(self, *args, **kwds):
        super(RungeKutta4, self).__init__(*args, **kwds)


    def advance(self, t0, u0, dt):
        """
        Very simple minded implementation of the standard 4th order Runge-Kutta
        method to solve an ODE of the form
        fdot = rhs(t,f)
        """
        if __debug__:
            self.log.debug("In advance")
            self.log.debug("time = %s"%repr(t0))
            self.log.debug("data = %s"%(repr(u0)))
            self.log.debug("dt = %s"%repr(dt))
        eqn = self.system
        u = u0
        t = t0
        k = eqn.evaluate(t, u)
        if __debug__:
            self.log.debug("First evaluate, k = %s"%repr(k))
        u1 = u0 + (dt/6.0)*k

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u)
        if __debug__:
            self.log.debug("Second evaluate, k = %s"%repr(k))
        u1 += dt/3.0*k

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u)
        if __debug__:
            self.log.debug("Third evaluate, k = %s"%repr(k))
        u1 += dt/3.0*k

        u = u0 + dt*k
        t = t0 + dt
        k = eqn.evaluate(t, u)
        if __debug__:
            self.log.debug("Fourth evaluate, k = %s"%repr(k))
        u1 += dt/6.0*k
        u1.time = t
        
        return (t,u1)

class RungeKutta4Dirichlet(ABCSolver):
    """An implementation of the Runge Kutta 4 routine that calls
    system.dirichlet_boundary."""


    def __init__(self, **kwds):
        super(RungeKutta4Dirichlet, self).__init__(**kwds)
        if not hasattr(self.system, "dirichlet_boundary"):
            raise Exception("%s does not implement dirichlet_boundary method"
                %self.system)

    def advance(self, t0, u0, dt):
        """
        Very simple minded implementation of the standard 4th order Runge-Kutta
        method to solve an ODE of the form
        fdot = rhs(t,f) that allows for the implementation of
        Dirichlet conditions during evaluation.
        
        Ensure that the corresponding system file has a method called,
        "dirichlet_boundary".
        """
        eqn = self.theEqn
        u = u0
        t = t0
        k = eqn.evaluate(t, u, intStep = 1)
        u1 = u0 + (dt/6.0)*k
        u1 = eqn.dirichlet_boundary(u1, intStep = 1)

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u, intStep = 2)
        u1 += dt/3.0*k
        u1 = eqn.dirichlet_boundary(u1, intStep = 2)

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u, intStep = 3)
        u1 += dt/3.0*k
        u1 = eqn.dirichlet_boundary(u1, intStep = 3)

        u = u0 + dt*k
        t = t0 + dt
        k = eqn.evaluate(t, u, intStep = None)
        u1 += dt/6.0*k
        u1.time = t
        u1 = eqn.dirichlet_boundary(u1, intStep = None)
        
        return (t,u1)

