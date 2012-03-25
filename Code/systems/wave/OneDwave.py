#!/usr/bin/env python
# encoding: utf-8
"""
spin2cyl.py

Created by Jörg Frauendiener on 2011-02-03.
Modified by Ben since 2011-03-28.
Copyright (c) 2011 University of Otago. All rights reserved.
"""
from __future__ import division

# import standard modules
import logging
import numpy as np
import math

# import our modules
from skyline.tslices import tslices
from skyline.systems.system import System

class OneDwave(System):

    def timestep(self,grid):
        ssizes = grid.step_sizes
        spatial_divisor = (1/ssizes[0])
        dt = self.CFL/spatial_divisor
        return dt
        
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, D, CFL, tau = None, log_parent = None ):
        super(OneDwave, self).__init__(CFL)
        self.log = log_parent.getChild("OneDwave")
        self.D = D
        self.tau = tau
        self.name = """<OneDwave D = %s, CLF = %f, tau = %s>"""%\
        (D.name, CFL, repr(tau))
        if __debug__:
            self.log.debug("Costruction of %s successful"%self.name)
        
    ############################################################################
    # Initial Conditions
    ############################################################################
    def initialValues(self,t0,r):
        #self.log.info("Initial value routine = central bump")
        #return self.centralBump(t0,r)
        if __debug__:
            self.log.info("Initial value routine = exp_bump")
        return self.exp_bump(t0,r)
        #self.log.info("Initial value routine = sin")
        #return self.sin(t0,r)
    
    ############################################################################
    # Boundary functions
    ############################################################################
    def dirichlet_boundary(self,u,intStep = None):
        #u.fields[0][-1]=self.boundaryRight(u.time,u)
        #u.fields[0][0]=self.boundaryLeft(u.time,u)
        #u.fields[0][0] = u.fields[-1][0]
        return u
    
    def boundaryRight(self,t,Psi):
        return 0.0
        
    def boundaryLeft(self,t,Psi):
        return 0.0
        
    ############################################################################
    # Evolution Routine
    ############################################################################
    def evaluate(self, t, Psi, intStep = None):
        if __debug__:
            self.log.debug("Entered evaluation: t = %f, Psi = %s, intStep = %s"%\
            (t,Psi,intStep))
         
        # Define useful variables
        f0,Dtf0 = tuple(Psi.fields[k] for k in range(Psi.numFields))
        
        x   = Psi.domain
        dx  = Psi.dx
        tau = self.tau
        
        ########################################################################
        # Calculate derivatives
        ########################################################################
        if __debug__:
            self.log.debug("f0.shape = %s"%repr(f0.shape))
        
        DxDxf = np.real(self.D(f0,dx))
        DtDtf = DxDxf
                
        if __debug__:
            self.log.debug("""Derivatives are:
                DtDtf = %s"""%\
                (repr(DtDtf)))
        
        ########################################################################
        # Impose boundary conditions 
        ########################################################################
                
        DtDtf[-1] =  0 #DtDtf[0] #self.boundaryRight(t,Psi)
        DtDtf[0] =  0 #self.boundaryLeft(t,Psi)       
                
        #DtDtf = DtDtf + \
        #        tau*(f0[0] - self.boundaryLeft(t,Psi))*\
        #            self.D.penalty_boundary(1,dx[0],DtDtf.shape) + \
        #        tau*(f0[-1] - self.boundaryRight(t,Psi))*\
        #            self.D.penalty_boundary(-1,dx[0],DtDtf.shape)
                
                
        # now all time derivatives are computed
        # package them into a time slice and return
        rtslice = tslices.timeslice([Dtf0,DtDtf],Psi.domain,time=t)
        if __debug__:
            self.log.debug("Exiting evaluation with timeslice = %s"%repr(rtslice))
        return rtslice
    
    ############################################################################
    # Initial Value Routines
    ############################################################################
    def centralBump(self,t0,grid):
        r = grid.axes
        r3ind = int(r.shape[0]/6)
        r3val = r[r3ind]
        r6val = r[6*r3ind]
        def bump(p):
            v = float(max(0.0,(-p + r3val) * (p - r6val)))**4
            return v
        def deriv_bump(p):
            #if r3ind < p < r6ind:
            #    return float(-2*p+r6val+r3val)
            return float(0)
        rv = np.vectorize(bump)(grid)
        ru = np.vectorize(deriv_bump)(grid)
        rv = 0.5*rv/np.amax(rv)
        ru = 0.5*ru/np.amax(rv)
        rtslice = tslices.timeslice([rv,ru],grid,t0)
        return rtslice
        
    def sin(self,t0,grid):
        r = grid.axes
        print r
        rv = np.sin(2*math.pi*r/(grid[-1]-grid[0]))
        rtslice = tslices.timeslice([rv,np.zeros_like(rv)],grid,t0)
        return rtslice
        
    def exp_bump(self,t0,grid):
        mid_ind = int(grid.shape[0]/2)
        rv = 0.5*np.exp(-10*(grid-grid[mid_ind])*(grid-grid[mid_ind]))
        return tslices.timeslice([rv,np.zeros_like(rv)],grid,t0)
