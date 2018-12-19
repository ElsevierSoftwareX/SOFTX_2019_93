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

class EthEthp(System):

    def timestep(self, grid):
        ssizes = grid.domain.step_sizes
        spatial_divisor = (1/ssizes[0])
        dt = self.CFL/spatial_divisor
        return dt
        
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, CFL, iv_routine, lmax):
        self.CFL = CFL
        self.eth = ghp.eth()
        self.ethp = ghp.ethp()
        self.lmax = lmax
        self.numvar = 2
        self.iv_routine = iv_routine
        self.name = """<SpinDwave CLF = %f, iv_routine = %s, lmax = %d>"""%\
            (CFL, iv_routine, lmax)
        self.log = logging.getLogger("SpinWave")
        if __debug__:
            self.log.debug("Costruction of %s successful"%self.name)
        
    ############################################################################
    # Initial Conditions
    ############################################################################
    def initial_data(self, t, grid):
        if __debug__:
            self.log.info("Initial value routine = %s"%self.iv_routine)
        values = getattr(self, self.iv_routine, None)(t, grid)
        return tslices.TimeSlice([values], grid, t)
    
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
        f = tslice.data[0]
        
        ########################################################################
        # Calculate derivatives
        ########################################################################
        ethf = self.eth(f, [0], self.lmax)
        dtf = self.ethp(ethf[0], [1], self.lmax)
        
        if __debug__:
            self.log.debug("""Derivatives are:
                dtf = %s
                """%
                (repr(dtf)))
        
        ########################################################################
        # Impose boundary conditions 
        ########################################################################
                
#        pt = self.D.penalty_boundary(dr, "right")
#        pt_shape = pt.size
#        
#        dtphi0[-pt_shape:] -= tau * (phi0[-1] - self.SATphi0(t, tslice)) * pt
#        dtphi2[:pt_shape] -= tau * (phi2[0] - self.SATphi2(t, tslice)) * pt
                
                
        ########################################################################
        # Packaging
        ########################################################################
        rtslice = tslices.TimeSlice(
            dtf, 
            tslice.domain, 
            time=t
            )
        if __debug__:
            self.log.debug("Exiting evaluation with timeslice = %s"%repr(rtslice))
        return rtslice
   
    ############################################################################
    # Initial Value Routines
    ############################################################################
    def Y000(self, t, grid):
        Y000 = np.zeros_like(grid.meshes[0], dtype=np.typeDict['complex128'])
        Y000[:] = 0.5*math.sqrt(1/math.pi)
        return Y000
        
    def Y01m1(self, t, grid):
        Y01m1 = np.zeros_like(grid.meshes[0], dtype=np.typeDict['complex128'])
        theta, phi = grid.meshes
        Y01m1 = 0.5*( 
            math.sqrt(3/(2*math.pi)) * np.sin(theta) * 
            np.exp(-complex(0,1)*phi)
            )
        return Y01m1
            
    def Y010(self, t, grid):
        Y010 = np.zeros_like(grid.meshes[0], dtype=np.typeDict['complex128'])
        theta, phi = grid.meshes
        Y010 = 0.5*(
            complex(1,0) * math.sqrt(3/(math.pi)) * np.cos(theta) 
            ) 
        return Y010
            
    def Y011(self, t, grid):
        Y011 = np.zeros_like(grid.meshes[0], dtype=np.typeDict['complex128'])
        theta, phi = grid.meshes
        Y011 = -0.5*(
            math.sqrt(3/(2*math.pi)) * np.sin(theta) * 
            np.exp(complex(0,1)*phi)
            );
        return Y011
