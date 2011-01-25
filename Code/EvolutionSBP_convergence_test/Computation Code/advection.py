#!/usr/bin/env python
# encoding: utf-8
"""
advection.py

Created by Jörg Frauendiener on 2011-01-14.
Copyright (c) 2011 University of Otago. All rights reserved.
"""

import numpy as np
import math
import h5py_array as h5py
import time
import sys

from ibvp import *
from tslices import *
from actions import *
from solvers import RungeKutta4
from system import System
from diffop import *


def bump(r,m,s):
    global g
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


class advection_eqn(System):
    """docstring for advection_eqn"""
    def __init__(self, D, CFL = 0.5, tau = 2.5):
        super(advection_eqn, self).__init__()
        self.D = D
        self.tau = tau
        self.CFL = CFL

    def evaluate(self, t, Psi):
        u, v  = tuple(Psi.fields[k] for k in range(Psi.numFields))
        x   = Psi.x
        dx  = Psi.dx
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
        dv[ 0] -= tau/dx * (v[ 0] + u[ -1])

        # now all time derivatives are computed
        # package them into a time slice and return

        return timeslice((du, dv), x, time = t)

    def initialValues(self, t0, grid):
        assert(grid.dim == 1), "Grid must be 1-dimensional"
        x = np.linspace(-1.0,1.0,grid.shape[0])
        v = -bump(x,0.75,0.125)
        u =  bump(x,0.25,0.25)
        return timeslice((u, v), x, time = t0)


    def left(self,t):
        return math.sin(t*5)


    def right(self,t):
        return 0.0


    def timestep(self,u):
        return self.CFL*u.dx



def main():
    """docstring for main"""
    rinterval = Grid((201,))
    rk4 = RungeKutta4()
    
    import advection
    system = advection.advection_eqn(D = D43_CNG(), CFL = 2., tau = 2.5)
    year = str(time.localtime()[0])
    month = str(time.localtime()[1])
    day = str(time.localtime()[2])
    hdf_file = h5py.H5pyArray("advection_"+day+"-"+month+"-"+year)
    problem = IBVP(rk4, system, rinterval, action = (\
            Plotter(frequency = 1, xlim = (1,2), ylim = (-5,5),\
                findex = (0,1), delay = 0.0),\
            HDFOutput(hdf_file,system)))
    problem.run(0.0, 10.0)





if __name__ == '__main__':
    main()
