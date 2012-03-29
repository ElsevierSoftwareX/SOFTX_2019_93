#!/usr/bin/env python
# encoding: utf-8 
"""
solvers.py

Created by Jörg Frauendiener on 2010-12-26.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

#import sys
#import os
import math
#from ..tslices import *
import logging

#############################################################################
class Solver(object):
    theEqn = None
    name = 'Generic Solver'

    def __init__(self, eqn=None):
        if eqn is not None:
            self.theEqn = eqn

    def useSystem(self, eqn):
        self.theEqn = eqn

    def advance(self, t, u, dt = None):
#        if __debug__:
#            self.log.debug("Advancing solution")
#            self.log.debug("t = %f, u = %s, dt = %f"%(t,repr(u),dt))
        if dt is None:
            dt = self.theEqn.timestep(u)
        eqn = self.theEqn
        du = eqn.evaluate(t, u)
#        if __debug__:
#            self.log.debug("Advanced solution is = %s"%repr((t+dt, u + dt*du)))
        return (t+dt, u + dt*du)

    def __repr__(self):
        return "<%s: equation = %s>"%(self.name,self.theEqn)


#############################################################################
class RungeKutta4(Solver):
    """docstring for RungeKutta4"""


    def __init__(self, log=None):
        super(RungeKutta4, self).__init__()
        self.log = logging.getLogger("rk4")


    def advance(self, t0, u0, dt):
        """
        Very simple minded implementation of the standard 4th order Runge-Kutta
        method to solve an ODE of the form
        fdot = rhs(t,f)
        """
#        if __debug__:
#            self.log.debug("solvers.advance(t0 = %f, u0 = %s, dt = %f"%(t0,repr(u0),dt))
        eqn = self.theEqn
        u = u0
        t = t0
        k = eqn.evaluate(t, u, intStep = 1)
        k.communicate()
        u1 = u0 + (dt/6.0)*k
        u1 = eqn.dirichlet_boundary(u1, intStep = 1)
#        if __debug__:
#            self.log.debug("k = %s"%repr(k))
#            self.log.debug("u1 after intStep = 1 is %s"%repr(u1))  

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u, intStep = 2)
        k.communicate()
        u1 += dt/3.0*k
        u1 = eqn.dirichlet_boundary(u1,intStep = 2)
#        if __debug__:
#            self.log.debug("k = %s"%repr(k))
#            self.log.debug("u1 after intStep = 1 is %s"%repr(u1))  

        u = u0 + dt/2.0*k
        t = t0 + dt/2.0
        k = eqn.evaluate(t, u, intStep = 3)
        k.communicate()
        u1 += dt/3.0*k
        u1 = eqn.dirichlet_boundary(u1,intStep = 3)
#        if __debug__:
#            self.log.debug("k = %s"%repr(k))
#            self.log.debug("u1 after intStep = 1 is %s"%repr(u1))  

        u = u0 + dt*k
        t = t0 + dt
        k = eqn.evaluate(t, u, intStep = None)
        k.communicate()
        u1 += dt/6.0*k
        u1.time = t
        u1 = eqn.dirichlet_boundary(u1,intStep = None)
#        if __debug__:
#            self.log.debug("k = %s"%repr(k))
#            self.log.debug("u1 after intStep = 1 is %s"%repr(u1))  
        
        return (t,u1)


class rk45(Solver):
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
