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
from coffee.diffop import ghp

class SpinWave(System):

    def timestep(self,grid):
        ssizes = grid.step_sizes
        spatial_divisor = (1/ssizes[0])
        dt = self.CFL/spatial_divisor
        return dt
        
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, D, CFL, tau, iv_routine, lmax):
        self.D = D
        self.tau = tau
        self.CFL = CFL
        self.eth = ghp.eth
        self.ethp = ghp.ethp
        self.lmax = lmax
        self.name = """<SpinDwave D = %s, CLF = %f, tau = %s, iv_routine = %s>"""%\
        (D.name, CFL, repr(tau), iv_routine)
        if __debug__:
            self.log.debug("Costruction of %s successful"%self.name)
        
    ############################################################################
    # Initial Conditions
    ############################################################################
    def initial_data(self, t, grid):
        if __debug__:
            self.log.info("Initial value routine = %s"%self.iv_routine)
        values = getattr(self, self.iv_routine, None)(t,r)
        return tslices.timeslice([values], grid, t)
    
    ############################################################################
    # Boundary functions
    ############################################################################
    def dirichlet_boundary(self,u,intStep = None):
        """This method is called by some RK routines."""
        return u
    
    def SATphi0(self,t,Psi):
        return 0.0
        
    def SATphi2(self,t,Psi):
        return 0.0
        
    ############################################################################
    # Evolution Routine
    ############################################################################
    def evaluate(self, t, tslice, intStep = None):
        if __debug__:
            self.log.debug("Entered evaluation: t = %f, Psi = %s, intStep = %s"%\
            (t,Psi,intStep))
         
        # Define useful variables
        phi0, phi1, phi2 = tuple(tslice.fields[k] for k in range(Psi.numFields))
        
        r   = tslice.grid.mesh[0]
        dr  = tslice.grid.mesh_steps[0]
        tau = self.tau
        
        ########################################################################
        # Calculate derivatives
        ########################################################################
        dr_phi0 = self.D(phi0, dr)
        dr_phi1 = self.D(phi1, dr)
        dr_phi2 = self.D(phi2, dr)
        eth_phi1 = self.eth(phi1, r, dr, [1], self.lmax)
        eth_phi2 = self.eth(phi2, r, dr, [1], self.lmax)
        ethp_phi0 = self.ethp(phi0, r, dr, [1], self.lmax)
        ethp_phi1 = self.ethp(phi1, r, dr, [1], self.lmax)
        
        if __debug__:
            self.log.debug("""Derivatives are:
                dr_phi0 = %s
                dr_phi1 = %s
                dr_phi2 = %s
                eth_phi1 = %s
                eth_phi2 = %s
                ethp_phi0 = %s
                ethp_phi1 = %s"""%
                (repr(dr_phi0), repr(dr_phi1), repr(dr_phi2),
                repr(eth_phi1), repr(eth_phi2), repr(ethp_phi0), 
                repr(ethp_phi1)))
                
        dtphi0 = drphi0 + (1/r) * (phi0 + eth_phi1)
        dtphi1 = (0.5/r) * (ethp_phi0 + eth_phi2)
        dtphi2 = -drphi2 - (1/r) * (phi2 - ethp_phi1)
        
        ########################################################################
        # Impose boundary conditions 
        ########################################################################
                
        pt = self.D.penalty_boundary(dr, "right")
        pt_shape = pt.size
        
        dpsi0[-pt_shape:] -= tau * (psi0[-1] - self.SATphi0(t, Psi)) * pt
        dpsi2[:pt_shape] -= tau * (psi2[0] - self.SATphi2(t, Psi)) * pt
                
                
        # now all time derivatives are computed
        # package them into a time slice and return
        rtslice = tslices.timeslice([Dtf0,DtDtf],Psi.domain,time=t)
        if __debug__:
            self.log.debug("Exiting evaluation with timeslice = %s"%repr(rtslice))
        return rtslice
    
    ############################################################################
    # Initial Value Routines
    ############################################################################
    def exp_bump(self, t, grid):
        r = grid.mesh[0]
        f = 0.5*np.exp(-20*(r*r))
        return f 
        
    ############################################################################
    # Constraints
    ############################################################################
    def constraints(self, tslice):
        r = grid.mesh[0]
        dr = grid.mesh_steps[0]
        drphi1 = self.D(tslice.fields[1], tslice.grids.steps[0])
        ethpphi0 = self.ethp(phi0)
        ethphi2 = self.eth(phi2)
        constraint = r * drphi1 - 0.5 * ethpphi0 + 0.5 * ethphi2 - 2* phi1
        return constraint
