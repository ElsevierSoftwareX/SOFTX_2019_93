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

    def timestep(self, tslice):
        ssizes = tslice.domain.step_sizes
        spatial_divisor = (1/ssizes[0])+(1/ssizes[1])
        dt = self.CFL/spatial_divisor
        return dt
        
    ############################################################################
    # Constructor
    ############################################################################
    def __init__(self, xdirec, ydirec, Dx, Dy, CFL, tau = None, 
        equation_coords = 'Polar',
        log_parent = None ):
        super(TwoDadvection, self).__init__()
        self.CFL = CFL
        self.xcoef = xdirec
        self.ycoef = ydirec
        self.log = log_parent.getChild("TwoDAdvection")
        self.Dx = Dx
        self.Dy = Dy
        self.numvar = 1
        self.tau = tau
        self.equation_coords = equation_coords
        self.name = """<TwoDadvection xdirec = %f, ydirec = %f, Dx = %s, 
        Dy = %s, CLF = %f, tau = %s>"""%\
        (xdirec,ydirec,Dx.name, Dy.name, CFL, repr(tau))
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
        f0, = Psi.data
        
        r = Psi.domain.axes[0]
        theta = Psi.domain.axes[1]
        dr  = Psi.domain.step_sizes[0]
        dphi = Psi.domain.step_sizes[1]
        tau = self.tau
        
        ########################################################################
        # Calculate derivatives and impose boundary conditions
        ########################################################################
        Dxf = np.apply_along_axis(
            lambda x:self.Dx(x, dr),
            1, 
            f0
            )
        #if __debug__:
        #    self.log.debug("Dxf is %s"%repr(Dxf))
        Dyf = np.apply_along_axis(
            lambda x:self.Dy(x,dphi),
            0,
            f0
            )
        Dtf = self.xcoef*Dxf + self.ycoef*Dyf
                
        if __debug__:
            self.log.debug("""Derivatives are: Dtf0 = %s"""%(repr(Dtf)))
        
        ########################################################################
        # Impose boundary conditions 
        ########################################################################
                
        # implementation follows Carpenter et al.
        # using the SAT method
        # at the boundaries we need boundary conditions
        # implemented as penalty terms the objects in diffop.py know how to 
        # do this. 
        #
        # tau is the penalty parameter and will need to take on different
        # values depending on the operator.
        #
        
        pt_x_r = self.Dx.penalty_boundary(dr, "right")
        pt_x_r_shape = pt_x_r.size 
        pt_x_l = self.Dx.penalty_boundary(dr, "left")
        pt_x_l_shape = pt_x_l.size 

        pt_y_r = self.Dy.penalty_boundary(dr, "right")
        pt_y_r_shape = pt_y_r.size 
        pt_y_l = self.Dy.penalty_boundary(dr, "left")
        pt_y_l_shape = pt_y_l.size 

        #First do internal boundaries
        if __debug__:
            self.log.debug("Implementing internal boundaries")
        b_values = Psi.communicate()
        if __debug__:
            self.log.debug("b_values = %s"%repr(b_values))
        for d_slice, data in b_values:
            if __debug__:
                self.log.debug("d_slice is %s"%(repr(d_slice)))
                self.log.debug("recieved_data is %s"%(data))
            #the calculation of sigma constants is taken from Carpenter,
            #Nordstorm and Gottlieb. Note that in this paper the metric H is
            #always set to the identity matrix. Beware: in some presentations
            #of SBP operators it is not the identitiy. This is accounted for
            #in the calculation of pt below.
            #I think that this paper implicitly assumes that 'a' is positive
            #hence the difference for psi4 from the calculations given in 
            #the paper. This change accounts for the negative eigenvalue
            #associated to psi4.
            #Note that sigma3 = sigma1 - eigenvalue_on_boundary, at least when
            #the eigenvalue is positive. For negative eigenvalue it seems to me
            #that the roles of sigma3 and sigma1 are reversed.
            psi0_chara = kappa[d_slice[1]]/(1+tkappap[d_slice[1]])
            psi4_chara = -kappa[d_slice[1]]/(1-tkappap[d_slice[1]])
            sigma3psi0 = 0
            sigma1psi0 = sigma3psi0 - 1
            sigma1psi4 = 0
            sigma3psi4 = sigma1psi4 - 1
            #sigma1drpsi4r0 = 1 #a2/(1 - t)
            #sigma3drpsi4r0 = 1 #sigma1drpsi4r0 - a2/(1 - t)
            #if __debug__:
                #self.log.debug("psi0_chara = %f"%psi0_chara)
                #self.log.debug("psi4_chara = %f"%psi4_chara)
                #self.log.debug("sigma1psi0 = %f"%sigma1psi0)
                #self.log.debug("sigma3psi0 = %f"%sigma3psi0)
                #self.log.debug("sigma1psi4 = %f"%sigma1psi4)
                #self.log.debug("sigma3psi4 = %f"%sigma3psi4)
            if d_slice[1] == slice(-1, None, None):
                if __debug__:
                    self.log.debug("Calculating right boundary")
                dpsi0[-pt_r_shape:] += sigma1psi0 * psi0_chara * pt_r * (
                    psi0[d_slice[1]] - data[0]
                    )
                dpsi4[-pt_r_shape:] += sigma1psi4 * psi4_chara * pt_r * (
                    psi4[d_slice[1]] - data[4]
                    )
                #dtdrpsi4r0[:] = data[5]
            else:
                if __debug__:
                    self.log.debug("Calculating left boundary")
                dpsi0[:pt_l_shape] += sigma3psi0 * psi0_chara * pt_l * (
                    psi0[d_slice[1]] - data[0]
                    )
                dpsi4[:pt_l_shape] += sigma3psi4 * psi4_chara * pt_l * (
                    psi4[d_slice[1]] - data[4]
                    )
                dtdrpsi4r0[:] = 0.0

        #Now do the external boundaries
        if __debug__:
            self.log.debug("Implementing external boundary")
        b_data = Psi.boundary_slices()
        if __debug__:
            self.log.debug("b_data = %s"%repr(b_data))
        for dim, direction, d_slice in b_data:
            if __debug__:
                self.log.debug("Boundary slice is %s"%repr(d_slice))
            if direction == 1:
                if __debug__:
                    self.log.debug("Doing external boundary")
                dpsi0[-pt_r_shape:] -= tau * (kappa[-1]/(1 + tkappap[-1])) \
                    * (psi0[-1] - self.right(t,Psi)) * pt_r
        #oned_pt = self.Dx.penalty_boundary(dr, "left")
        #oned_pt_shape = oned_pt.size
        #penalty_term = np.lib.stride_tricks.as_strided(
            #oned_pt,
            #shape = (oned_pt_shape, r.size),
            #strides = (oned_pt.itemsize, 0)
            #)
        #Dxf[:oned_pt_shape] -= self.tau * \
            #(f0[:oned_pt_shape] - self.first_right(t, Psi)) * \
            #penalty_term
                
        # now all time derivatives are computed
        # package them into a time slice and return
        rtslice = tslices.TimeSlice([Dtf],Psi.domain,time=t)
        if __debug__:
            self.log.debug("Exiting evaluation with TimeSlice = %s"%repr(rtslice))
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
    def centralBump(self, t0, grid):
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
        r_mesh, phi_mesh = grid.meshes
        def exp_bump(p):
            rv = np.exp(-20*(p[0]-r[rmid])**2)*\
                np.exp(-5*(p[1]-phi[phimid])**2)
            return rv
        def bump(p):
            v = max(0.0,(-p[0] + r3val) * (p[0] - r6val))*\
                max(0.0,(-p[1] + phi3val) * (p[1] - phi6val))
            return float(v)**4
        #rv = np.apply_along_axis(bump,2,grid)
        rv = np.exp(-20*(r_mesh - r[rmid])**2)*\
            np.exp(-5*(phi_mesh - phi[phimid])**2)
        #rv = np.apply_along_axis(exp_bump,2,grid)
        #rv = rv/np.amax(rv)
        if rank == 0:
            rtslice = tslices.TimeSlice([rv],grid,t0)
        else:
            rtslice = tslices.TimeSlice([np.zeros_like(rv)],grid,t0)
        return rtslice
