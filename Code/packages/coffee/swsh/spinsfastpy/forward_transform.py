""" 
This module is part of spinsfastpy. Please see the package documentation. It
contains python bindings for revision 90 of the forward transformations
implemented in Huffenberger's & Wandelt's spinsfast code. See "Fast and exact 
spin-s Spherical Harmonic Transformations" in Astrophysical Journal Supplement 
Series (2010) 189:255-260. All references to sections, appendices or equations 
in the documentation below refer to equations given in this paper.

This module implements the methods of the file spinsfast_forward.h required
for spin s spherical transformations of functions on S^2.

Methods:

Imm -- Returns Imm, see equation (8)
Jmm -- Returns Jmm, see equation (10)
forward -- Returns the spin spherical harmonic coefficients, equation (6),
    using either Imm or Jmm which can be specified via keyword arguments

Copyright 2012
Ben Whale 
version 3 - GNU General Public License
<http://www.gnu.org/licenses/>
"""
################################################################################
# setup
################################################################################

# standard imports
import ctypes
import numpy as np
from numpy import ctypeslib, typeDict
import inspect
import os

# imports from this module
import salm
import delta_matrix as dm

# load a shared object library version of spinsfast
sf = ctypes.CDLL(
         os.path.abspath(
             os.path.join(
                 os.path.dirname(inspect.getfile(inspect.currentframe())),
                 "..", "lib", "libspinsfast.so.1"
                 )
             )
         )

sf.spinsfast_forward_multi_Imm.restype = ctypes.c_void_p
sf.spinsfast_forward_multi_Imm.argtypes = [ \
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned') ]
#sf.spinsfast_forward_multi_Imm.argtypes = [ \
    #ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        #flags='contiguous, writeable, aligned'), \
    #ctypeslib.ndpointer(dtype=typeDict['int'], ndim=1, \
        #flags='contiguous, writeable, aligned'), \
    #ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    #ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        #flags='contiguous, writeable, aligned') ]

sf.spinsfast_forward_multi_Jmm.restype = ctypes.c_void_p
sf.spinsfast_forward_multi_Jmm.argtypes = [ \
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned') ]

sf.spinsfast_forward_transform.restype = ctypes.c_void_p
sf.spinsfast_forward_transform.argtypes = [
    ctypeslib.ndpointer(
        dtype=typeDict['complex'], 
        ndim=1,
        flags='contiguous, writeable, aligned'
        ),
    ctypes.c_int,
    ctypeslib.ndpointer(
        dtype=ctypes.c_int, 
        ndim=1,
        flags='contiguous, writeable, aligned'
        ),
    ctypes.c_int,
    ctypeslib.ndpointer(
        dtype=typeDict['complex'], 
        ndim=1,
        flags='contiguous, writeable, aligned'
        ),
    ctypes.c_int, 
    ctypes.c_void_p ]
#sf.spinsfast_forward_transform.argtypes = [
    #ctypeslib.ndpointer(
        #dtype=typeDict['complex'], 
        #ndim=1,
        #flags='contiguous, writeable, aligned'
        #),
    #ctypes.c_int,
    #ctypeslib.ndpointer(
        #dtype=typeDict['int'], 
        #ndim=1,
        #flags='contiguous, writeable, aligned'
        #),
    #ctypes.c_int,
    #ctypeslib.ndpointer(
        #dtype=typeDict['complex'], 
        #ndim=1,
        #flags='contiguous, writeable, aligned'
        #),
    #ctypes.c_int, 
    #ctypes.c_void_p ]

sf.spinsfast_forward_transform_eo.restype = ctypes.c_void_p
sf.spinsfast_forward_transform_eo.argtypes = [ \
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypes.c_int, \
    ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypes.c_int, \
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'), \
    ctypes.c_int, ctypes.c_void_p ]
#sf.spinsfast_forward_transform_eo.argtypes = [ \
    #ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        #flags='contiguous, writeable, aligned'), \
    #ctypes.c_int, \
    #ctypeslib.ndpointer(dtype=typeDict['int'], ndim=1, \
        #flags='contiguous, writeable, aligned'), \
    #ctypes.c_int, \
    #ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        #flags='contiguous, writeable, aligned'), \
    #ctypes.c_int, ctypes.c_void_p ]

################################################################################
# the python functions
################################################################################
def forward(f, spins, lmax, delta_method="TN_PLANE", real=False):
    Jmm_set = _Jmm(f, spins, lmax)
    #print "Jmm"
    #print repr(Jmm_set)
    return _forward_Jmm(spins, lmax, Jmm_set, delta_method, real)

def _Imm(f, spins, lmax):
    """
    Returns Imm.
    
    Arguments:
    f -- the values of a function on S^2 represented as a numpy.ndarry
         of shape (Ntheta, Nphi) according to the ecp discretisation, see
         Section 2.3 
    spins -- a float or a numpy.ndarray of shape (Nmap,) of floats giving 
             the required spins for the calculation. Note that allowed
             floats are signed half integers. This is not checked.
    lmax -- the maximum l needed.
    
    Returns:
    Imm -- a numpy.ndarray of shape (Nmap,2*lmax + 1,2*lmax + 1). The ndarray
           Imm[i,:,:] corresponds to Imm for spins[i]. The calculation for
           Imm is done according to appendix A.
    """
    if len(f.shape) == 2:
        f = np.array([f])
    Nmaps, Ntheta, Nphi = f.shape
    f_flat = np.asarray(f.flatten('C'), dtype = typeDict['complex'])
    spins = np.atleast_1d(spins)
    if len(spins.shape)!=1:
        raise ValueError('spins must be an int or a one dimensional array of ints')
    Nm = 2*lmax + 1
    NImm = Nm*Nm
    Imm = np.empty(NImm, dtype = typeDict['complex'])
    sf.spinsfast_forward_multi_Imm(f_flat, spins, Nmaps, Ntheta, Nphi, lmax, Imm)
    return Imm.reshape(Nm,Nm)

def _Jmm(f, spins, lmax):
    """
    Returns Jmm.
    
    Arguments:
    f -- the values of a function on S^2 represented as a numpy.ndarry
         of shape (Ntheta, Nphi) according to the ecp discretisation, see
         Section 2.3 
    spins -- a float or a numpy.ndarray of shape (Nmap,) of floats giving 
             the required spins for the calculation. Note that allowed
             floats are signed half integers. This is not checked.
    lmax -- the maximum l needed.
    
    Returns:
    Jmm -- a numpy.ndarray of shape (Nmap,2*lmax + 1,lmax+1). The ndarray
           Jmm[i,:,:] corresponds to Jmm for spins[i]. The calculation for
           Jmm is done according to equation (10).
    """
    if len(f.shape) == 2:
        f = np.array([f])
    Nmaps, Ntheta, Nphi = f.shape
    f_flat = np.asarray(f.flatten('C'), dtype = typeDict['complex'])
    spins = np.atleast_1d(spins)
    if len(spins.shape)!=1:
        raise ValueError('spins must be an int or a one dimensional array of ints')
    Nm = 2 * lmax + 1
    NJmm = Nm * (lmax + 1)
    #import pdb; pdb.set_trace()
    Jmm = np.empty(Nmaps*NJmm, dtype = typeDict['complex'])
    sf.spinsfast_forward_multi_Jmm(f_flat, spins, Nmaps, Ntheta, Nphi, lmax, Jmm)
    return Jmm

def _forward_Jmm(spins, lmax, Jmm, delta_method="TN_PLANE", real=False):
    """
    Returns the spin spherical harmonic coefficients for f given Jmm.
    
    Arguments:
    spins -- a float or a numpy.ndarray of shape (Nmap,) of floats giving 
             the required spins for the calculation. Note that allowed
             floats are signed half integers. This is not checked.
    lmax -- the maximum l needed.
    Jmm -- a numpy.ndarray of shape (Nmap,2*lmax + 1,lmax+1). The ndarray
           Jmm[i,:,:] corresponds to Jmm for spins[i]. The calculation for
           Jmm is done according to equation (10).
                   
    Key word arguments:
    delta_method -- a constant indicating which of the available methods to use 
                    in calculation of the deltas: the Wigner d functions 
                    evaluated at pi/2, see equation (4). The available methods 
                    can be accessed via delta_methods. Defaults to "TN".
    real -- Is the original function f real? Defaults to False. If real
            then the optimisations given in the last paragraph of section 2.1
            are used.
    
    Returns:
    salm -- an instance of the class alm.salm. This class gives acces to the 
            spin coefficients of the function f from which Jmm was calculated.
    """
    spins = np.atleast_1d(spins)
    if len(spins.shape)!=1:
        raise ValueError('spins must be an int or a one dimensional array of ints')
    Ntransform = spins.shape[0]
    a = np.empty(sf.N_lm(lmax) * Ntransform, dtype = typeDict['complex'])
    DeltaMethod, Deltawork = dm.methods[delta_method](lmax)
    if real:
        sf.spinsfast_forward_transform_eo(a, Ntransform, spins, lmax, Jmm, \
            DeltaMethod, Deltawork)
    else:    
        sf.spinsfast_forward_transform(
            a, 
            Ntransform, 
            spins, 
            lmax, 
            Jmm,
            DeltaMethod, 
            Deltawork
            )
    return salm.sfpy_salm(a.reshape(Ntransform,sf.N_lm(lmax)),spins,lmax)

#a = np.zeros((2,8,8), dtype="complex")
#print forward(a, [1,3], 3)
