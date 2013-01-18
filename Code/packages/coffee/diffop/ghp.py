from __future__ import division
import math
import os
import numpy as np

from coffee.swsh import spinsfastpy as sfpy

#This module could be reimplemented using meta-classes as eth and ethp
#essentially just differ by a boolean value. I have already tried to avoid
#having copies of essentially the same code, but this would completely
#remove this remainin duplicates

class eth(object):
    
    def __init__(self):
        """
        Implements the differential operator given by
        \eth{}_sY_{lm} = \sqrt{l(l+1)-s(s+1)}{}_{s+1}Y_{lm}
        """

    def __call__(self, u, spins = None, lmax = None):
        if isinstance(u, sfpy.salm.sfpy_sralm):
            return _eval_sralm(self, u)
        else:
            return self._eval_salm(u, spins, lmax)

    def _eval_salm(self, u, spins = None, lmax = None):
        Ntheta = None
        Nphi = None
        salm, spins, lmax = _transform_to_harmonic_space(
            u, spins, lmax, Ntheta, Nphi
            )
        r_salm = sfpy.salm.sfpy_salm(
            np.empty_like(salm, dtype=np.typeDict['complex']), 
            spins + 1, 
            lmax
            )
        _do_derivative(salm, spins, lmax, _eth_eigen, r_salm)
        return _transform_from_harmonic_space(r_salm, Ntheta, Nphi)
  
    def __repr__(self):
        return "eth"


class ethp(object):
    
    def __init__(self):
        """
        Implements the differential operator given by
        \eth'{}_sY_{lm} = -\sqrt{l(l+1)-s(s-1)}{}_{s-1}Y_{lm}
        """

    def __call__(self, u, spins = None, lmax = None):
        if isinstance(u, sfpy.salm.sfpy_sralm):
            return _eval_sralm(self, u)
        else:
            return self._eval_salm(u, spins, lmax)

    def _eval_salm(self, u, spins = None, lmax = None):
        Ntheta = None
        Nphi = None
        salm, spins, lmax = _transform_to_harmonic_space(
            u, spins, lmax, Ntheta, Nphi
            )
        r_salm = sfpy.salm.sfpy_salm(
            np.empty_like(salm, dtype=np.typeDict['complex']), 
            spins - 1, 
            lmax
            )
        _do_derivative(salm, spins, lmax, _ethp_eigen, r_salm)
        return _transform_from_harmonic_space(r_salm, Ntheta, Nphi)
  
    def __repr__(self):
        return "ethp"

def _eval_sralm(self, u):
    rv_temp = self._eval_salm(u[:,0,:])
    rv = np.empty(
        (rv_temp.shape[0],) + (u.shape[1],) + (rv_temp.shape[1],),
        dtype = u.dtype
        )
    rv[:,0,:] = np.asarray(rv_temp)
    for i in range(1,u.shape[1]):
        rv[:, i, :] = self._eval_salm(u[:, i, :])
    return sfpy.salm.sfpy_sralm(
        rv,
        u.spins,
        u.lmax,
        cg=u.cg,
        bandlimit_multiplication=u.bl_mult
        )

def _eth_eigen(s, j):
    if abs(s+1) > j:
        return 0
    else:
        return math.sqrt(j*(j+1)-s*(s+1))
        
def _ethp_eigen(s, j):
    if abs(s-1) > j:
        return 0
    else:
        return -math.sqrt(j*(j+1)-s*(s-1))

def _transform_to_harmonic_space(u, spins, lmax, Nthetea, Nphi):
    if spins is not None and lmax is not None:
        Ntheta, Nphi = u.shape
        salm = sfpy.forward(u, spins, lmax)
        spins = np.asarray(spins)
    else:
        salm = u
        spins = u.spins
        lmax = u.lmax
    return salm, spins, lmax

def _do_derivative(salm, spins, lmax, eigen_calc, r_salm):
    for j in range(lmax + 1):
        sm_values = salm[:,j]
        eigenvalues = np.vectorize(eigen_calc)(spins, j)
        r_salm[:,j] = eigenvalues[:, np.newaxis] * sm_values

def _transform_from_harmonic_space(r_salm, Ntheta, Nphi):
    if Ntheta is not None:
        return sfpy.backward(r_salm, Ntheta, Nphi)
    return r_salm
