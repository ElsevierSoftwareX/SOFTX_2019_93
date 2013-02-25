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
        eqn = self.system
        u = u0
        t = t0
        k = eqn.evaluate(t, u)
        u1 = u0 + (dt/6.0)*k

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u)
        u1 += dt/3.0*k

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u)
        u1 += dt/3.0*k

        u = u0 + dt*k
        t = t0 + dt
        k = eqn.evaluate(t, u)
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

################################################################################
# This code has not been updated in some time. It needs to be checked before
# use.
################################################################################
class rk45(ABCSolver):
    """
    Solve a system of ODE using a Runge-Kutta method

    with adaptive step size control. The code is taken
    almost literally from Numerical Recipes Ch. 16
    """

    a2 = 0.5; a3 = 0.3; a4 = 0.6; a5 = 1.; a6 = 0.875;

    b21 = 0.2; b31 = 0.075; b32 = 0.225;
    b41 = 0.3; b42 = -0.9; b43 = 1.2;
    b51 = -11./54.; b52 = 2.5; b53 = -70./27.; b54 = 35./27.
    b61 = 1631./55296.; b62 = 175./512.; b63 = 575./13824.;
    b64 = 44275./110592.; b65 = 253./4096.

    c1 = 37./378.; c3 = 250./621.; c4 = 125./594.; c6 = 512./1771.

    # d_i = c_i - c_i^*   (cf. Numerical Recipes in C p. 717)
    d1 = c1 - 2825./27648.; d3 = c3 - 18575./48384.;
    d4 = c4 - 13525./55296.; d5 = - 277./14336.; d6 = c6 - 0.25
    SAFETY = 0.9
    PGROW = -0.2
    PSHRINK = -0.25
    ERRCON = 1.89e-4
    TINY = 1e-30
    EPS = 1e-8

    """ The problem here is that the methods rkdrv and rk45step are designed to
      operate on arrays, while advance operates on timeslices. This makes
      everything a bit awkward because we have to wrap arrays in timeslices and
      vice versa. Maybe one needs to rethink this bit of the code in a bit more
      detail. Since System.evaluate also expects timeslices we change things in
      the rk45 routines. """

    def __init__(self, eqn=None, eps = None, log = None):
        if eps is not None:
            self.EPS = eps
        super(rk45, self).__init__(eqn=eqn,log=log)


    def advance(self, t0, u0, dt,  eps=EPS):
        t = t0
        t1 = t0 + dt
        u = u0
        x = u0.x
        k = 0
        while (t < t1):
            k = k+1
            du = self.theEqn.evaluate(t, u)
            s = abs(u) + abs(du*dt) + self.TINY
            t, u, dt0, dt = self.rkdrv(t, u, dt, du, s)
            if (t+dt > t1): dt = t1-t
        return t, timeslice(u, x, time=t)



    def rkdrv(self, t, u, htry, du, scale):
        h = htry
        while(True):
            u1, uerr = self.rk45step(t, u, h, du)
            a = abs(uerr/scale)
            errmax = a.max()
            errmax = errmax/self.EPS
            if (errmax <= 1.): break
            ht = self.SAFETY*h*math.pow(errmax, self.PSHRINK)
            if (h > 0.0):
                h = max(ht, 0.1*h)
            else:
                h = min(ht, 0.1*h)

            t1 = t+h
            if (t1 == t):
                raise ValueError;

        if (errmax > self.ERRCON):
            hnext = self.SAFETY*h*math.pow(errmax, self.PGROW)
        else:
            hnext = 5.*h

        return (t + h, u1, h, hnext)


    def rk45step(self, t, u, h,  du):
        eqn = self.theEqn

        utmp = u + h*self.b21*du
        k2 = eqn.evaluate(t + self.a2*h, utmp)

        utmp = u + h*(self.b31*du + self.b32*k2)
        k3 = eqn.evaluate(t + self.a3*h, utmp)

        utmp = u + h*(self.b41*du + self.b42*k2 + self.b43*k3)
        k4 = eqn.evaluate(t + self.a4*h, utmp)

        utmp = u + h*(self.b51*du + self.b52*k2 + self.b53*k3 \
                 + self.b54*k4)
        k5 = eqn.evaluate(t + self.a5*h, utmp)

        utmp = u + h*(self.b61*du + self.b62*k2 + self.b63*k3 \
                 + self.b64*k4 + self.b65*k5)
        k6 = eqn.evaluate(t + self.a6*h, utmp)

        uout = u + h*(self.c1*du + self.c3*k3 + self.c4*k4 \
                 + self.c6*k6)
        uerr = h*(self.d1*du + self.d3*k3 + self.d4*k4 \
                 + self.d5*k5 + self.d6*k6)

        return uout, uerr
