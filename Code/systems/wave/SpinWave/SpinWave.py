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
import itertools

# import our modules
from coffee.tslices import tslices
from coffee.system import System
from coffee.diffop import ghp

class SpinWave(System):

    def timestep(self, grid):
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
        self.eth = ghp.eth()
        self.ethp = ghp.ethp()
        self.lmax = lmax
        self.iv_routine = iv_routine
        self.name = """<SpinDwave D = %s, CLF = %f, tau = %s, iv_routine = %s>"""%\
        (D.name, CFL, repr(tau), iv_routine)
        if __debug__:
            self.log = logging.getLogger("SpinWave")
            self.log.debug("Costruction of %s successful"%self.name)
        
    ############################################################################
    # Initial Conditions
    ############################################################################
    def initial_data(self, t, grid):
        if __debug__:
            self.log.info("Initial value routine = %s"%self.iv_routine)
        values = getattr(self, self.iv_routine, None)(t, grid)
        return tslices.timeslice(values, grid, t)
    
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
            self.log.debug("Entered evaluation: t = %f, tslice = %s, intStep = %s"%\
            (t, tslice, intStep))
         
        # Define useful variables
        phi0, phi1, phi2 = tslice.fields
        
        r   = tslice.grid.meshes[0]
        dr  = tslice.grid.step_sizes[0]
        tau = self.tau
        
        ########################################################################
        # Calculate derivatives
        ########################################################################
        dr_phi0 = np.apply_along_axis(self.D, 0, phi0, dr)
        #dr_phi1 = np.apply_along_axis(self.D, 0, phi1, dr)
        dr_phi2 = np.apply_along_axis(self.D, 0, phi2, dr)
        eth_phi1 = np.asarray(
            [self.eth(data, [1], self.lmax) for data in phi1]
            )
        eth_phi2 = np.asarray(
            [self.eth(data, [1], self.lmax) for data in phi2]
            )
        ethp_phi0 = np.asarray(
            [self.ethp(data, [1], self.lmax) for data in phi0]
            )
        ethp_phi1 = np.asarray(
            [self.ethp(data, [1], self.lmax) for data in phi1]
            )
        
        if __debug__:
            self.log.debug("""Derivatives are:
                dr_phi0 = %s
                dr_phi2 = %s
                eth_phi1 = %s
                eth_phi2 = %s
                ethp_phi0 = %s
                ethp_phi1 = %s"""%
                (repr(dr_phi0), repr(dr_phi2),
                repr(eth_phi1), repr(eth_phi2), repr(ethp_phi0), 
                repr(ethp_phi1)))
                
        dtphi0 = dr_phi0 + (1/r) * (phi0 + eth_phi1)
        dtphi1 = (0.5/r) * (ethp_phi0 + eth_phi2)
        dtphi2 = -dr_phi2 - (1/r) * (phi2 - ethp_phi1)
        
        ########################################################################
        # Impose boundary conditions 
        ########################################################################
                
        pt = self.D.penalty_boundary(dr, "right")
        pt_shape = pt.size
        
        dtphi0[-pt_shape:] -= tau * (phi0[-1] - self.SATphi0(t, tslice)) * pt
        dtphi2[:pt_shape] -= tau * (phi2[0] - self.SATphi2(t, tslice)) * pt
                
                
        # now all time derivatives are computed
        # package them into a time slice and return
        rtslice = tslices.timeslice(
            [dtphi0, dtphi1, dtphi2], 
            tslice.grid, 
            time=t
            )
        if __debug__:
            self.log.debug("Exiting evaluation with timeslice = %s"%repr(rtslice))
        return rtslice
    
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
   
    ############################################################################
    # Initial Value Routines
    ############################################################################
    def exp_bump(self, t, grid):
        r = grid.meshes[0]
        phi0 = 0.5*np.exp(-20*(r*r))
        phi1 = np.zeros_like(phi0)
        phi2 = np.zeros_like(phi0)
        return [phi0, phi1, phi2]
