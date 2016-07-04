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
from functools import partial

# import our modules
from coffee.tslices import tslices
from coffee.system import System

def sbp_ghost_point_processor(
        data=None,
        b_values=None,
        speed=1,
        f0=None,
        Dtf=None,
        pt_r=None,
        pt_l=None,
        log=None
    ):
    if __debug__ and log:
        log.debug("b_values = %s"%repr(b_values))
    for d_slice, data in b_values:
        if __debug__ and log:
            log.debug("d_slice is %s"%(repr(d_slice)))
            log.debug("recieved_data is %s"%(data))
        if speed < 0:
            sigma1 = 0.25 
            sigma3 = sigma1 - 1
        else:
            sigma3 = 0.25 
            sigma1 = sigma3 - 1
        if d_slice[1] == slice(-1, None, None):
            if __debug__ and log:
                log.debug("Calculating right boundary")
            Dtf[-pt_r.size:] += sigma1 * speed * pt_r * (f0[d_slice[1]] - data[0])
        else:
            if __debug__ and log:
                log.debug("Calculating left boundary")
            Dtf[:pt_l.size] += sigma3 * speed * pt_l * (f0[d_slice[1]] - data[0])

class OneDAdvectionMpi(System):

    def timestep(self, tslice):
        ssizes = tslice.domain.step_sizes
        spatial_divisor = (1/ssizes[0])
        dt = self.CFL/spatial_divisor
        return dt
        
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, D, speed, CFL, tau = None):
#        super(OneDAdvection, self).__init__()
        self.log = logging.getLogger("OneDAdvection")
        self.D = D
        self.speed = speed
        self.tau = tau
        self.CFL = CFL
        self.numvar = 1
        self.name = """<OneDAdvection D = %s, CLF = %f, tau = %s>"""%\
        (D.name, CFL, repr(tau))
        if __debug__:
            self.log.debug("Costruction of %s successful"%self.name)
        
    ############################################################################
    # Configuration for initial conditions and boundary conditions.
    ############################################################################
    def initial_data(self,t0,r):
        return self.centralBump(t0,r)
        #self.log.info("Initial value routine = exp_bump")
        #return self.exp_bump(t0,r)
        #self.log.info("Initial value routine = sin")
        #return self.sin(t0,r)
        #self.log.info("Initial value routine = data")
        #return self.data(t0,r)
    
    def boundaryRight(self,t,Psi):
        #return 0.5 * math.sin( 
            #2*math.pi*(t+2) / ( 
                #Psi.domain.axes[0][-1] - Psi.domain.axes[0][0] 
                #) 
            #)
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
        f0 = Psi.data[0]
        x   = Psi.domain.axes[0]
        dx  = Psi.domain.step_sizes[0]
        tau = self.tau
        
        ########################################################################
        # Calculate derivatives
        ########################################################################
        Dxf = np.real(self.D(f0,dx))
        Dtf = self.speed * Dxf
        
        #First do internal boundaries
        pt_r = self.D.penalty_boundary(dx, "right")
        pt_l = self.D.penalty_boundary(dx, "left")
        gp_processor = partial(
            sbp_ghost_point_processor,
            speed=self.speed,
            f0=f0,
            Dtf=Dtf,
            pt_r=pt_r,
            pt_l=pt_l,
            log=self.log
        )
        if __debug__:
            self.log.debug("Implementing internal boundary")
        b_values = Psi.communicate(gp_processor)

        #Now do the external boundaries
        if __debug__:
            self.log.debug("Implementing external boundary")
        b_data = Psi.external_slices()
        if __debug__:
            self.log.debug("b_data = %s"%repr(b_data))
        for dim, direction, d_slice in b_data:
            if __debug__:
                self.log.debug("Boundary slice is %s"%repr(d_slice))
            if self.speed > 0 and direction == 1:
                if __debug__:
                    self.log.debug("Doing external boundary on right")
                Dtf[-pt_r.size:] -= tau * self.speed * (
                    f0[-1] - self.boundaryRight(t,Psi)
                    ) * pt_r
            elif self.speed < 0 and direction == -1:
                if __debug__:
                    self.log.debug("Doing external boundary on left")
                Dtf[:pt_l.size] -= tau * self.speed * (
                    f0[0] - self.boundaryLeft(t,Psi)
                    ) * pt_l
        
        if __debug__:
            self.log.debug("""Derivatives are:
                Dtf = %s"""%\
                (repr(Dtf)))

        ########################################################################
        # Impose boundary conditions 
        ########################################################################
        #Dtf[-1]= 0.#self.boundaryRight(t,Psi)
        #Dtf[0]= self.boundaryLeft(t,Psi)
        #Dtf[-1] = Dtf[0]
                
        # now all time derivatives are computed
        # package them into a time slice and return
        rtslice = tslices.TimeSlice(np.array([Dtf]), Psi.domain, time=t)
        if __debug__:
            self.log.debug("Exiting evaluation with timeslice = %s"%
                repr(rtslice))
        return rtslice
    
    ############################################################################
    # Boundary functions
    ############################################################################
    def dirichlet_boundary(self, u, intStep = None):
        #u.fields[0][-1] = self.boundaryRight(u.time,u)
        return u
    
    ############################################################################
    # Initial Value Routines
    ############################################################################
    def centralBump(self, t0, grid):
        self.log.info("Initial value routine = central bump")
        r = grid.axes[0]
        length = grid.bounds[0][1] - grid.bounds[0][0]
        rv = np.maximum(0.0, 
            (36) * (1/length)**2 * (-r + length/3) * (r - 2*length/3)
            )**8
        if __debug__:
            self.log.debug(repr(rv))
        #if np.amax(rv is not 0:
            #rv = 0.5*rv/np.amax(rv)
        rtslice = tslices.TimeSlice(np.array([rv]),grid,t0)
        return rtslice
        
    def exp_bump(self, t, grid):
        axes = grid.axes[0]
        mid_ind = int(axes.shape[0]/2)
        rv = (1/72.0) * length**2  * np.exp(-40*(axes-axes[mid_ind])*(axes-axes[mid_ind]))
        return tslices.TimeSlice(np.array([rv]), grid, t)
    
    def sin(self,t0,grid):
        r = grid.axes
        rv = np.sin(2*math.pi*r/(grid[-1]-grid[0]))
        rtslice = tslices.TimeSlice([0.5*rv],grid,t0)
        return rtslice
        
    def data(self,t0,grid):
        rv = np.array([ 0.00000401632684942,  0.00000192499658763,  0.00000099772870147,\
         0.00000022765183832,  0.00000046507080863, -0.0000003300452312 ,\
         0.00000062974172828, -0.00000084085150719,  0.00000122978111633,\
        -0.00000173153916076,  0.00000243852897086, -0.00000339401167924,\
         0.00000468353699582, -0.00000640006545601,  0.00000866039005127,\
        -0.00001159615978749,  0.00001534736518352, -0.00002004222246138,\
         0.00002576538282592, -0.00003251084480869,  0.0000401219186533 ,\
        -0.00004822399684474,  0.00005616161979072, -0.00006295603141441,\
         0.00006730239613382, -0.00006762518639408,  0.00006220491467334,\
        -0.0000493790556335 ,  0.00002780564387487,  0.00000323813794329,\
        -0.00004356481100534,  0.00009184500062725, -0.00014548888853252,\
         0.00020068538741031, -0.00025262465531646,  0.00029590251389135,\
        -0.00032507451254041,  0.00033529828730872, -0.00032298121854387,\
         0.00028634089954959, -0.00022579126761677,  0.00014408737801006,\
        -0.00004619390760402, -0.00006111857691737,  0.00016990621035476,\
        -0.00027183673081992,  0.00035902855096032, -0.00042483915235696,\
         0.00046450680546802, -0.00047557203841141,  0.00045803716953868,\
        -0.00041425815985836,  0.00034859741282364, -0.00026689395793278,\
         0.00017582532588945, -0.00008224165993041, -0.00000745225032476,\
         0.00008780553643091, -0.00015467906027939,  0.00020543401573565,\
        -0.00023895456368999,  0.0002555262007749 , -0.00025660403132089,\
         0.0002445120127969 , -0.00022211440620396,  0.00019249677896301,\
        -0.00015868547023992,  0.00012342541952096, -0.00008902553443739,\
         0.0000572726977001 , -0.00002940740612892,  0.00000615033664036,\
         0.00001223462391232, -0.00002585437390634,  0.00003509147037659,\
        -0.00004050974746794,  0.00004276960709606, -0.00004255693092361,\
         0.0000405291526316 , -0.00003727739758106,  0.0000333046546635 ,\
        -0.00002901606283444,  0.00002472000591966, -0.00002063537528648,\
         0.0000169039610633 , -0.00001360432584562,  0.00001076564712092,\
        -0.00000838344097786,  0.00000642403682553, -0.00000485353496625,\
         0.00000359675704848, -0.00000265847140611,  0.00000186187684214,\
        -0.00000143563257268,  0.00000077349425006, -0.00000096984553866,\
        -0.00000014450388622, -0.00000125698079451, -0.00000141060384138,\
        -0.00000214380180203, -0.0000024309521466 ])
        return tslices.TimeSlice(np.array([rv]),grid,t0)
