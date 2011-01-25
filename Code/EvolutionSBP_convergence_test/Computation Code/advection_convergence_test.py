#!/usr/bin/env python
# encoding: utf-8
"""
advection.py

Created by Jörg Frauendiener on 2011-01-14.
Copyright (c) 2011 University of Otago. All rights reserved.
"""

import numpy as np
import tslices
import system

def bump(r,m,s):
    a = r[0]
    b = r[-1]
    f = lambda (t): t*b + (1-t)*a
    r0 = f(m)
    r1 = f(m-s)
    r2 = f(m+s)
    print(a,r1,r0,r2,b)
    bmp = (( (r-r1)/(r0-r1) )*( (r-r2)/(r0-r2) ))**4
    bmp[ r < r1 ] = 0
    bmp[ r > r2 ] = 0
    return  ( bmp )

class advection_eqn(system.System):
    """docstring for advection_eqn"""
    def __init__(self, D, CFL = 0.5, tau = 2.5):
        super(advection_eqn, self).__init__()
        self.D = D
        self.tau = tau
        self.CFL = CFL
        self.name = "CFL = "+str(CFL)

    def evaluate(self, t, psi):
        u, v  = tuple(psi.fields[k] for k in range(psi.numFields))
        x   = psi.x
        dx  = psi.dx
        tau = self.tau
        
        # implementation follows Carpenter et al.
        # using the SAT method

        # impose the equation at every point
        du =  self.D(u,dx) 
        dv = -self.D(v,dx) 

        # at the boundaries we need boundary conditions
        # implemented as penalty terms
        # use g_{NN} = 1, g_{00} = -1,
        # note that g_{00} lambda >0 and g_NN lambda > 0 in those points
        # where the boundary condition needs to be imposed
        du[-1] -= tau/dx * (u[-1] - self.left(t))
        dv[ 0] -= tau/dx * (v[ 0] + u[0])

        # now all time derivatives are computed
        # package them into a time slice and return
        return tslices.timeslice((du, dv), x, time = t)

    def initialValues(self, t0, grid):
        assert(grid.dim == 1), "Grid must be 1-dimensional"
        x = np.linspace(-1.0,1.0,grid.shape[0])
        u =  bump(x,0.25,0.25)
        v = np.zeros_like(u)#-bump(x,0.75,0.125)
        return tslices.timeslice((u, v), x, time = t0)

    def left(self,t):
        return 0.0

    def right(self,t):
        return 0.0

    def timestep(self,u):
        return self.CFL*u.dx
