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

    def boundary(self, t, Psi):
        return np.zeros_like(Psi.data[0])
    
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
        
        x = Psi.domain.axes[0]
        y = Psi.domain.axes[1]
        dx  = Psi.domain.step_sizes[0]
        dy = Psi.domain.step_sizes[1]
        tau = self.tau
        
        ########################################################################
        # Calculate derivatives and impose boundary conditions
        ########################################################################
        Dxf = np.apply_along_axis(
            lambda x:self.Dx(x, dx),
            0, 
            f0
            )
        #if __debug__:
        #    self.log.debug("Dxf is %s"%repr(Dxf))
        Dyf = np.apply_along_axis(
            lambda y:self.Dy(y,dy),
            1,
            f0
            )
                
        if __debug__:
            self.log.debug("""Derivatives are: Dxf = %s"""%(repr(Dxf)))
            self.log.debug("""Derivatives are: Dyf = %s"""%(repr(Dyf)))
        
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
        
        pt_x_r = self.Dx.penalty_boundary(dx, "right")
        pt_x_r_shape = pt_x_r.size 
        pt_x_l = self.Dx.penalty_boundary(dx, "left")
        pt_x_l_shape = pt_x_l.size 

        pt_y_r = self.Dy.penalty_boundary(dy, "right")
        pt_y_r_shape = pt_y_r.size 
        pt_y_l = self.Dy.penalty_boundary(dy, "left")
        pt_y_l_shape = pt_y_l.size 

        #First do internal boundaries
        if __debug__:
            self.log.debug("Implementing internal boundaries")
        _, b_values = Psi.communicate() # compare to OneDAdvection for an 
                                        # alternative way to handle this.
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
            x_chara = self.xcoef
            y_chara = self.ycoef
            if x_chara > 0:
                sigma3x = 0
            else:
                sigma3x = 1
            sigma1x = sigma3x - 1
            if y_chara > 0:
                sigma1y = 0
            else:
               sigma1y = 1
            sigma3y = sigma1y - 1
            if d_slice[1] == slice(-1, None, None):
                if __debug__:
                    self.log.debug("Calculating right x boundary")
                Dxf[-pt_x_r_shape:] += sigma1x * x_chara * pt_x_r * (
                        f0[d_slice[1:]] - data[0]
                    )
            elif d_slice[1] == slice(None, 1, None):
                if __debug__:
                    self.log.debug("Calculating left x boundary")
                Dxf[:pt_x_l_shape] += sigma3x * x_chara * pt_x_l * (
                        f0[d_slice[1:]] - data[0]
                        )
            elif d_slice[1] == slice(None, None, None):
                if d_slice[2] == slice(-1, None, None):
                    if __debug__:
                        self.log.debug("Calculating right y boundary")
                    Dyf[-pt_y_r_shape:] += sigma1y * y_chara * pt_y_r * (
                            f0[d_slice[1:]] - data[0]
                        )
                elif d_slice[2] == slice(None, 1, None):
                    if __debug__:
                        self.log.debug("Calculating left y boundary")
                    Dyf[:pt_y_l_shape] += sigma3y * y_chara * pt_y_l * (
                            f0[d_slice[1:]] - data[0]
                            )

        #Now do the external boundaries
        if __debug__:
            self.log.debug("Implementing external boundary")
        b_data = Psi.external_slices()
        if __debug__:
            self.log.debug("b_data = %s"%repr(b_data))
        for dim, direction, d_slice in b_data:
            if __debug__:
                self.log.debug("Boundary slice is %s"%repr(d_slice))
                self.log.debug("Dimension is %d"%dim)
                self.log.debug("Direction is %d"%direction)
            d_slice = d_slice[1:]
            if dim == 0:
                if self.xcoef > 0 and direction == 1:
                    if __debug__:
                        self.log.debug("Doing right hand x boundary")
                    Dxf[-pt_x_r_shape:] -= tau * self.xcoef * \
                        (f0[d_slice] - self.boundary(t,Psi)[d_slice]) * pt_x_r
                if self.xcoef < 0 and direction == -1:
                    if __debug__:
                        self.log.debug("Doing left hand x boundary")
                    Dxf[:pt_x_l_shape] += tau * self.xcoef * \
                        (f0[d_slice] - self.boundary(t,Psi)[d_slice]) * pt_x_l
            elif dim == 1:
                if self.ycoef > 0 and direction == 1:
                    if __debug__:
                        self.log.debug("Doing right hand y boundary")
                    Dyf[:,-pt_y_r_shape:] -= tau * self.ycoef * \
                        (f0[d_slice] - self.boundary(t,Psi)[d_slice]) * pt_y_r
                if self.ycoef < 0 and direction == -1:
                    if __debug__:
                        self.log.debug("Doing left hand y boundary")
                    Dyf[:,:pt_y_l_shape] += tau * self.ycoef * \
                        (f0[d_slice] - self.boundary(t,Psi)[d_slice]) * pt_y_l

        Dtf = self.xcoef*Dxf + self.ycoef*Dyf

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
