from __future__ import division
import math
import os
import numpy as np

from coffee.swsh import spinsfastpy as sfpy

class eth(object):
    
    def __init__(self):
        """
        Implements the differential operator given by
        \eth{}_sY_{lm} = \sqrt{l(l+1)-s(s+1)}{}_{s+1}Y_{lm}
        """

    def __call__(self, u, spins = None, lmax = None):
        Ntheta = None
        if spins is not None and lmax is not None:
            Ntheta, Nphi = u.shape
            salm = sfpy.forward(u, spins, lmax)
            spins = np.asarray(spins)
        else:
            salm = u
            spins = u.spins
            lmax = u.lmax
        r_salm = sfpy.sfpy_salm(
            np.empty_like(np.asarray(salm), dtype=np.typeDict['complex']), 
            spins + 1, 
            lmax
            )
        #print "salm = %s"%salm
        for j in range(lmax):
            sm_values = salm[:,j]
            eigenvalues = np.vectorize(_eth_eigen)(spins, j)
            eigenvalues = np.lib.stride_tricks.as_strided(
                eigenvalues,
                sm_values.shape,
                (eigenvalues.itemsize, 0)
                )
            r_salm[:,j] = eigenvalues * sm_values
        #print "r_salm = %s"%r_salm
        if Ntheta is not None:
            r_u = sfpy.backward(r_salm, Ntheta, Nphi)
            return r_u
        else:
            return r_salm
        
  
    def __repr__(self):
        return "eth"

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

class ethp(object):
    
    def __init__(self):
        """
        Implements the differential operator given by
        \eth'{}_sY_{lm} = -\sqrt{l(l+1)-s(s-1)}{}_{s-1}Y_{lm}
        """

    def __call__(self, u, spins = None, lmax = None):
        Ntheta = None
        if spins is not None and lmax is not None:
            Ntheta, Nphi = u.shape
            salm = sfpy.forward(u, spins, lmax)
            spins = np.asarray(spins)
        else:
            salm = u
            spins = u.spins
            lmax = u.lmax
        r_salm = sfpy.sfpy_salm(np.empty_like(salm, dtype=np.typeDict['complex']), spins - 1, lmax)
        #print "salm = %s"%salm
        for j in range(lmax):
            sm_values = salm[:,j]
            eigenvalues = np.vectorize(_ethp_eigen)(spins, j)
            eigenvalues = np.lib.stride_tricks.as_strided(
                eigenvalues,
                sm_values.shape,
                (eigenvalues.itemsize, 0)
                )
            r_salm[:,j] = eigenvalues * sm_values
        #print "r_salm = %s"%r_salm
        if Ntheta is not None:
            r_u = sfpy.backward(r_salm, Ntheta, Nphi)
            return r_u
        else:
            return r_salm
  
    def __repr__(self):
        return "ethp"
