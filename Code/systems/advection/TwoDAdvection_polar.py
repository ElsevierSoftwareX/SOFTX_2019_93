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
from coffee.tslices import tslices
from coffee.system import System

class TwoDadvection(System):

    def timestep(self,grid):
        ssizes = grid.step_sizes
        spatial_divisor = (1/ssizes[0])+(1/ssizes[1])
        dt = self.CFL/spatial_divisor
        return dt
        
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, xdirec, ydirec, D1, DFFT, CFL, tau = None, 
        equation_coords = 'Polar',
        log_parent = None ):
        super(TwoDadvection, self).__init__()
        self.CFL = CFL
        self.xcoef = xdirec
        self.ycoef = ydirec
        self.log = log_parent.getChild("TwoDAdvection")
        self.D1 = D1
        self.DFFT = DFFT
        self.tau = tau
        self.equation_coords = equation_coords
        self.name = """<TwoDadvection xdirec = %f, ydirec = %f, D1 = %s, 
        DFFT = %s, CLF = %f, tau = %s>"""%\
        (xdirec,ydirec,D1.name, DFFT.name, CFL, repr(tau))
        if __debug__:
            self.log.debug("Costruction of %s successful"%self.name)
        
    ############################################################################
    # Configuration for initial conditions and boundary conditions.
    ############################################################################
    def initial_data(self,t0,r):
        self.log.info("Initial value routine = central bump")
        return self.centralBump(t0,r)
    
    def first_right(self,t,Psi):
        return np.zeros_like(Psi.domain.axes[0])
        
    def first_left(self,t,Psi):
        return (0.0,0.0)
        
    ############################################################################
    # Evolution Routine
    ############################################################################
    def evaluate(self, t, Psi, intStep = None):
        #if __debug__:
        #    self.log.debug("Entered evaluation: t = %f, Psi = %s, intStep = %s"%\
        #        (t,Psi,intStep))
         
        # Define useful variables
        f0, = tuple(Psi.fields[k] for k in range(Psi.numFields))
        
        r   = Psi.domain.axes[0]
        theta = Psi.domain.axes[1]
        dr  = Psi.dx[0]
        dphi = Psi.dx[1]
        tau = self.tau
        csth = np.cos(theta)+np.sin(theta)
        phi_period = (2*math.pi)/(theta[-1]-theta[0])
        
        ########################################################################
        # Calculate derivatives and impose boundary conditions
        ########################################################################
        if self.equation_coords is 'Polar':
            Drf = np.apply_along_axis(lambda x:self.D1(x,dr),Psi.domain.r,f0)
#            penalty_term = np.lib.stride_tricks.as_strided(
#                self.D.penaly_boundary(dr,"left")
#            n = penalty_term.shape[1]
#            Drf[:n] -= self.tau * (f0[0] - self.right(t, Psi)) * penalty_term
            
            Dphif = np.apply_along_axis(lambda x: self.DFFT(x,dphi),Psi.domain.phi,f0)
            Dtf = (self.xcoef*np.cos(theta)+self.ycoef*np.sin(theta))*Drf+\
                (self.ycoef*np.cos(theta)-self.xcoef*np.sin(theta))*\
                    (((phi_period/r)*Dphif.swapaxes(0,1)).swapaxes(0,1))
        else:
            Dxf = np.apply_along_axis(lambda x:self.D1(x,dr), Psi.domain.r, f0)
            oned_pt = self.D1.penalty_boundary(dr, "left")
            oned_pt_shape = oned_pt.size
            penalty_term = np.lib.stride_tricks.as_strided(
                oned_pt,
                shape = (oned_pt_shape, r.size),
                strides = (oned_pt.itemsize, 0)
                )
            Dxf[:oned_pt_shape] -= self.tau * \
                (f0[:oned_pt_shape] - self.first_right(t, Psi)) * \
                penalty_term
            #if __debug__:
            #    self.log.debug("Dxf is %s"%repr(Dxf))
            Dyf = np.apply_along_axis(lambda x:self.DFFT(x,dphi),\
                Psi.domain.phi,f0)
            Dtf = self.xcoef*Dxf + self.ycoef*Dyf
            #PT_edges = Psi.domain.get_edge(f0)
            #for slice,v in PT_edges:
                #if __debug__:
                #    self.log.debug("Slice for penalty method is %s"%repr(slice))
                #    self.log.debug("Boundary values are %s"%repr(v))
            #    for i,v in enumerate(self.tau*(f0[slice][0]-v)):
            #        Dtf[:,i] -= v*self.D1.penalty_boundary(self.xcoef,dr,Dxf.shape)[:,i]
                
        #if __debug__:
        #    self.log.debug("""Derivatives are: Dtf0 = %s"""%(repr(Dtf)))
        
        ########################################################################
        # Impose boundary conditions 
        ########################################################################
                
        #Dtf0[-1], DtDtf[-1] = self.boundaryRight(t,Psi)
        #Dtf0[0], DtDtf[0] = self.boundaryLeft(t,Psi)
                
        # now all time derivatives are computed
        # package them into a time slice and return
        rtslice = tslices.timeslice([Dtf],Psi.domain,time=t)
        #if __debug__:
        #    self.log.debug("Exiting evaluation with timeslice = %s"%repr(rtslice))
        return rtslice
    
    ############################################################################
    # Boundary functions
    ############################################################################
    def dirichlet_boundary(self,u,intStep = None):
#        #r boundary
#        from mpi4py import MPI
#        rank = MPI.COMM_WORLD.rank
#        if rank == 0:
#            u.fields[0][0,:] = 0 # u.fields[0][-1,:]
        #phi boundary
        u.fields[0][:,0] = u.fields[0][:,-1]
        return u
    
    ############################################################################
    # Initial Value Routines
    ############################################################################
    def centralBump(self,t0,grid):
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank
        r, phi = grid.axes[0],grid.axes[1]
        r3ind = int(r.shape[0]/6)
        r3val = r[r3ind]
        r6val = r[6*r3ind]
        phi3ind = int(phi.shape[0]/5)
        phi3val = phi[phi3ind]
        phi6val = phi[4*phi3ind]
        rmid = int(r.shape[0]/2)
        phimid = int(phi.shape[0]/2)
        def exp_bump(p):
            rv = np.exp(-20*(p[0]-r[rmid])**2)*\
                np.exp(-5*(p[1]-phi[phimid])**2)
            return rv
        def bump(p):
            v = max(0.0,(-p[0] + r3val) * (p[0] - r6val))*\
                max(0.0,(-p[1] + phi3val) * (p[1] - phi6val))
            return float(v)**4
        #rv = np.apply_along_axis(bump,2,grid)
        rv = np.apply_along_axis(exp_bump,2,grid)
        #rv = rv/np.amax(rv)
        if rank == 0:
            rtslice = tslices.timeslice([rv],grid,t0)
        else:
            rtslice = tslices.timeslice([np.zeros_like(rv)],grid,t0)
        return rtslice
