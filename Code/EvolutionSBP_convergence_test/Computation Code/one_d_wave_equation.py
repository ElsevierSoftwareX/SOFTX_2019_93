#!/usr/bin/env python
# encoding: utf-8
"""
advection.py

Created by Ben Whale on 2011-01-25.
"""

import numpy as np
import tslices
import system

def smooth_non_analytic(x):
    def f(y): 
        if y<=0.0:
            return 0.0
        else:
            return np.exp(-1/y)
    return np.vectorize(f)(x)

def smooth_step(x, start, stop):
    y = (x-start)/(stop-start)
    return (smooth_non_analytic(y))/ \
        (smooth_non_analytic(y)+\
                smooth_non_analytic(1-y))

def smooth_bump(x, start_up, stop_up,start_down, stop_down):
    return smooth_step(x, start_up, stop_up)*\
        (1-smooth_step(x, start_down, stop_down))

def exp_bump(x):
    return np.exp(-20*x*x)

def jorgs_poly_bump(x, p):
    return (1-(3*x/4)**2)**p


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

def sin(x, period, translate):
    return np.sin(np.pi*x)

class wave_eqn(system.System):
    """docstring for advection_eqn"""
    def __init__(self,CFL = 0.5):
        super(wave_eqn, self).__init__()
        self.CFL = CFL
        self.name = "CFL = "+str(CFL)

    def evaluate(self, t, psi):
        u = psi.fields[0]
        x   = psi.x
        dx  = psi.dx
        N = u.shape[0]
        
        # implementation of 4th order
        # central difference for interior points
        # and 4th order forward or backward difference
        # for boundaries and one point in from the 
        # boundaries.

        du = np.zeros_like(u)
        # impose the equation at interior points
        for i in range(u.shape[0]):
            du[i] =  (1./12)*u[(i-2 )% N]-(2./3)*u[(i-1)% N]+(2./3)*u[(i+1)% N]-(1./12)*u[(i+2)% N]
        
        du = du/(dx)
        
        # apply difference for boundary points
        #du[0] = -(1/4.0)*u[4]+(4/3.0)*u[3]-(3.0)*u[2]+(4.0)*u[1]-(25/12.0)*u[0]
        #du[1] = (1/(12.0))*(u[4]-6*u[3]+18*u[2]-10*u[1]-3*u[0])
        #du[-1] = (1/4.0)*u[-5]-(4/3.0)*u[-4]+(3.0)*u[-3]-(4.0)*u[-2]+(25/12.0)*u[-1]
        #du[-2] = (1/(12.0))*(-u[-5]+6*u[-4]-18*u[-3]+10*u[-2]+3*u[-1])

        # impose boundary conditions, if any, on derivatives

        # now all time derivatives are computed
        # package them into a time slice and return
        return tslices.timeslice((du), x, time = t)

    def initialValues(self, t0, grid):
        return self.exactValues(t0, grid)

    def timestep(self,u):
        return self.CFL*u.dx
        
    def exactValues(self, t, grid):
        x = grid.getGrid(-1, 1)
        u =  exp_bump(x)#smooth_bump(x,-0.25,-0.1 ,0.1, 0.25)#sin(x, 2, -1)#bump(x,0.25,0.25)#jorgs_poly_bump(x,4)#
        v = np.zeros_like(u)
        N = u.shape[0]
        for i in range(N):
            v[i] = u[i+t%N]
        return tslices.timeslice((v), x, t)
