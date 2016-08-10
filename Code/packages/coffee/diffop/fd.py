#!/usr/bin/env python
# encoding: utf-8
"""
diffop.py

Created by Jörg Frauendiener on 2011-01-10.
Copyright (c) 2011 University of Otago. All rights reserved.
"""
from __future__ import division

import numpy as np
import logging


################################################################################
# Finite difference default ghost point processor
################################################################################
def ghost_point_processor(data, b_values, log=None):
    if __debug__ and log:
        log.debug("original data is " + repr(data))
    for b_slice, b_data in b_values:
        if __debug__ and log:
            log.debug("b_slice is %s"%(repr(b_slice)))
            log.debug("b_data is %s"%(repr(b_data)))
        data[b_slice] = b_data
    if __debug__ and log:
        log.debug("new data is " + repr(data))


################################################################################
# Finite difference stencil base class.
################################################################################

class FD_stencil(object):
    
    def __init__(self,stencil_coefs,loffset,roffset):
        self.stencil_coefs = stencil_coefs
        self.loffset = loffset
        self.roffset = roffset
        
    def __repr__(self):
        return "<FD_stencil %s, [%d,%d]>"%(repr(self.stencil_coefs),\
            self.loffset,self.roffset)
        
    def __call__(self, u, apply_at=None):
        if apply_at is not None:
            lbound = apply_at - self.loffset
            rbound = apply_at + self.roffset + 1
            if rbound == 0: 
                rbound = None
            return np.dot(self.stencil_coefs, u[lbound : rbound])
        else:
            """The convolve method does not quite do the right thing, it
            flips the direction of iteration. Hence the ::-1 below.
            
            Note also that the convolve method is based on the 
            multiarray.correlate a C routine."""
            return np.convolve(u,self.stencil_coefs[::-1],mode='same')
           
def flip(stencil):
    """This method takes, for example, a forward difference operator and returns
    a backwards difference operator using `the same' stencil.
    
    E.g. flip(F12_stencil) = B12_stencil,
    E.g. flip(C12_stencil) = C12_stencil"""
    return FD_stencil(-(stencil.stencil_coefs[::-1]),\
        stencil.roffset,stencil.loffset)
        
################################################################################
# First derivative stencils
################################################################################
"""Further stencil's can be calculated using 
`finite_difference_weights_generator.py' located in Code/Utils.
I suggest reading Fornberg's paper for the notation."""

class C12_stencil(FD_stencil):

    def __init__(self):
        stencil = np.array([-1.0/2.0,0.0,1.0/2.0])
        loff = 1
        roff = 1
        super(C12_stencil,self).__init__(stencil,loff,roff)
   
class C14_stencil(FD_stencil):

    def __init__(self):
        stencil = np.array([1.0/12, -2.0/3, 0, 2.0/3,-1.0/12])
        loff = 2
        roff = 2
        super(C14_stencil,self).__init__(stencil,loff,roff)
   
class F12_stencil(FD_stencil):
    
    def __init__(self):
        stencil = np.array([-3.0/2.0, 2, -1.0/2.0])
        loff = 0
        roff = 2
        super(F12_stencil,self).__init__(stencil,loff,roff)
        
class F13_stencil(FD_stencil):
    
    def __init__(self):
        stencil = np.array([-11.0/6.0, 3.0, -3.0/2.0, 1.0/3])
        loff = 0
        roff = 3
        super(F13_stencil,self).__init__(stencil,loff,roff)
        
class F13_stencil(FD_stencil):
    
    def __init__(self):
        stencil = np.array([-11.0/6.0, 3.0, -3.0/2.0, 1.0/3])
        loff = 0
        roff = 3
        super(F13_stencil,self).__init__(stencil,loff,roff)
        
class F14_stencil(FD_stencil):
    
    def __init__(self):
        stencil = np.array([-25.0/12, 4.0, -3.0, 4.0/3, -1.0/4])
        loff = 0
        roff = 4
        super(F14_stencil,self).__init__(stencil,loff,roff)

class MF14_stencil(FD_stencil):

    def __init__(self):
        stencil = np.array([-1.0/4, -15.0/18, 3.0/2, -1.0/2, 15.0/180])
        loff = 1
        roff = 3
        super(MF14_stencil,self).__init__(stencil,loff,roff)

class B12_stencil(FD_stencil):
    
    def __init__(self):
        stencil = np.array([1.0/2.0,-2.0,3.0/2.0])
        loff = 2
        roff = 0
        super(B12_stencil,self).__init__(stencil,loff,roff)

################################################################################
# Second derivative stencils
################################################################################

class C22_stencil(FD_stencil):

    def __init__(self):
        stencil = np.array([1.0,-2.0,1.0])
        loff = 1
        roff = 1
        super(C22_stencil,self).__init__(stencil,loff,roff)
        
class F22_stencil(FD_stencil):

    def __init__(self):
        stencil = np.array([2.0,-5.0,4.0,-1.0])
        loff = 0
        roff = 3
        super(F22_stencil,self).__init__(stencil,loff,roff)
        
class B22_stencil(FD_stencil):

    def __init__(self):
        stencil = np.array([1.0,-4.0,5.0,-2.0])
        loff = 3
        roff = 0
        super(B22_stencil,self).__init__(stencil,loff,roff)

################################################################################
# Finite Difference base class
################################################################################

class FD_diffop(object):
    name = "FD_diffop"

    def __init__(self):
        self.log = logging.getLogger('FD')

    def __call__(self, u, dx):
        ru = self.central(u)
        for i,b in self.boundaries:
            if __debug__:
                self.log.debug(
                    "Applying boundary: i = " + repr(i) + ", b = " + repr(b)
                )
            ru[i] = b(u, apply_at = i)
        return ru/(dx**self.order)

    def __str__(self):
        return "Differential operator "%self.name
        
    def save(self):
        filename = os.path.expanduser("~/" + self.name)
        print filename
        np.savetxt(filename + "_left.txt", self.central)
        np.savetxt(filename + "_right.txt", self.boundaries)

    def ghost_points(self):
        return self.central.loffset, self.central.roffset

    def internal_points(self):
        return self.ghost_points()

################################################################################
# First Derivative operators
################################################################################
class FD12(FD_diffop):
    """Implements a second order FD routine for first order derivatives. 
    On the boundaries uses forward/backward differences. Sufficiently
    away from the boundaries uses a central schemes. Assumes that the
    FD grid is evenly spaced."""
    
    name = "FD12"
    central = C12_stencil()
    boundaries = np.array(\
        [(0,F12_stencil()),\
        (-1,B12_stencil())]\
        )
    order = 1

class FD14(FD_diffop):
    """Implements a fourth order FD routine for first order derivatives. 
    On the boundaries uses forward/backward differences. On
    interor points where the central difference scheme cannot be applied,
    skewed forward/backward difference stencil's are used. Sufficiently
    away from the boundaries uses a central schemes. Assumes that the
    FD grid is evenly spaced."""
    
    name = "FD14"
    central = C14_stencil()
    boundaries = np.array(\
        [(0, F14_stencil()),\
        (1, MF14_stencil()),\
        (-2, flip( MF14_stencil() )),\
        (-1, flip( F14_stencil() ))])
    order = 1

################################################################################
# Second Derivative operators
################################################################################

class FD22(FD_diffop):
    """Implements a second order FD routine for second order derivatives. 
    On the boundaries uses forward/backward differences. Sufficiently
    away from the boundaries uses a central schemes. Assumes that the
    FD grid is evenly spaced."""
    
    name = "FD22"
    central = C22_stencil()
    boundaries = np.array(\
        [(0,F22_stencil()),\
        (-1,B22_stencil())]\
        )
    order = 2
