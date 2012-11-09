""" 
This module is part of spinsfastpy. Please see the package documentation. It
contains python bindings for revision 90 of the backward transformations
implemented in Huffenberger's & Wandelt's spinsfast code. See "Fast and exact 
spin-s spherical harmonic transformations" in Astrophysical Journal Supplement 
Series (2010) 189:255-260. All references to sections, appendices or equations 
in the documentation below refer to equations given in this paper.

This module implements the methods of the file spinsfast_backward.h required
for spin s spherical transformations of functions on S^2.

Methods:

Gmm -- Returns Gmm, see equation (13), given the spin s spherical harmonic
       coefficients
backward -- Returns the values of the function, equation (12), on an ecp grid, 
            section 2.3, using Gmm.

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
import os
import inspect

# imports from this package
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

sf.spinsfast_backward_transform.restype = ctypes.c_void_p
sf.spinsfast_backward_transform.argtypes = [\
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'),\
    ctypes.c_int, ctypes.c_int, ctypes.c_int, \
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned')]

sf.spinsfast_backward_Gmm.restype = ctypes.c_void_p
sf.spinsfast_backward_Gmm.argtypes = [\
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'),\
    ctypes.c_int, \
    ctypeslib.ndpointer(dtype=typeDict['int'], ndim=1, \
        flags='contiguous, writeable, aligned'),\
    ctypes.c_int, \
    ctypeslib.ndpointer(dtype=typeDict['complex'], ndim=1, \
        flags='contiguous, writeable, aligned'),\
    ctypes.c_int, ctypes.c_void_p]

################################################################################
# the python functions
################################################################################
def backward(salm, Ntheta, Nphi, delta_method="TN_PLANE"):
    """
    Returns a function on S^2 given it's spin coefficients in the form of an
    alm.salm object.
    
    Arguments:
    salm -- the alm.salm object that stores the spin coefficients
    Ntheta -- an int giving the number of points discritising the theta 
              variable of S^2
    Nphi -- an int giving the number of points discritising the phi variable of
            S^2
    
    Key work arguments:
    delta_method -- a constant indicating which of the available methods to use 
                    in calculation of the deltas: the Wigner d functions 
                    evaluated at pi/2, see equation (4). The available methods 
                    can be accessed via delta_matrix.py. Defaults to "TN".
    
    Returns:
    f -- a numpy.ndarray of shape (Ntheta, Nphi) containing the values of the
         function on S^2, parameterised via the ecp discretisation, see
         section 2.3
    """
    Gmm_set = _Gmm(salm, delta_method)
    return _backward_Gmm(Ntheta, Nphi, salm.lmax, salm.spins.shape[0], Gmm_set)

def _backward_Gmm(Ntheta, Nphi, lmax, Nmaps, Gmm):
    """
    Returns a function on S^2 given Gmm.
    
    Arguments:
    Ntheta -- an int giving the number of points discritising the theta 
              variable of S^2
    Nphi -- an int giving the number of points discritising the phi variable of
            S^2
    Nmaps -- an int that should be equal to alm.salm.spins.shape[0]
    Gmm -- a numpy.ndarray of shape (Nmap,2*lmax + 1,2*lmax + 1). The ndarray
           Gmm[i,:,:] corresponds to Gmm for spins[i]. The array Gmm is 
           calculated from equation (13).
    
    Returns:
    f -- a numpy.ndarray of shape (Ntheta, Nphi) containing the values of the
         function on S^2, parameterised via the ecp discretisation, see
         section 2.3
    """
    f = np.empty((Nmaps,Ntheta*Nphi), dtype = typeDict['complex'])
    Nm = 2 * lmax + 1
    NGmm = Nm * Nm
    for i in range(Nmaps):
        sf.spinsfast_backward_transform(f[i], Ntheta, Nphi, lmax, Gmm[i*NGmm:(i+1)*NGmm])
    if Nmaps == 1:
        return f.reshape(Nmaps,Ntheta,Nphi)[0]
    else:
        return f.reshape(Nmaps,Ntheta,Nphi)
    
def _Gmm(salm, delta_method="TN"):
    """
    Returns Gmm.
    
    Arguments:
    salm -- the spin spherical harmonic coefficients represented as an alm.salm
            object
    
    Key work arguments:
    delta_method -- a constant indicating which of the available methods to use 
                    in calculation of the deltas: the Wigner d functions 
                    evaluated at pi/2, see equation (4). The available methods 
                    can be accessed via delta_methods. Defaults to "TN".
    
    Returns:
    Gmm -- a numpy.ndarray of shape (Nmap,2*lmax + 1,2*lmax + 1). The ndarray
           Gmm[i,:,:] corresponds to Gmm for spins[i]. The array Gmm is 
           calculated from equation (13).
    """
    spins = salm.spins
    Ntransform = spins.shape[0]
    lmax = salm.lmax
    if len(spins.shape) != 1:
        raise ValueError('spins must be an int or a one dimensional array of ints')
    Nm = 2*lmax+1
    Gmm_set = np.empty(Nm*Nm*Ntransform, dtype=typeDict['complex'])
    alm_flat = np.asarray(salm.coefs,dtype=typeDict['complex']).flatten('C')
    DeltaMethod, Deltawork = dm.methods[delta_method](lmax)
    sf.spinsfast_backward_Gmm(alm_flat, Ntransform, spins, lmax, Gmm_set, DeltaMethod, Deltawork)
    return Gmm_set
    
    
    
