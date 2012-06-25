from __future__ import division
import math
import os
import numpy as np

from coffee.swsh import spinsfastpy as sfpy

class eth(object):
    
    def __init__(self):
        """
        Implements the differential operator given by
        \eth{}_sY_{lm} = -\sqrt{l(l+1)-s(s+1)}{}_{s+1}Y_{lm}
        """

    def __call__(self, u, spins = None, lmax = None):
        Ntheta = None
        if spins is not None and lmax is not None:
            Ntheta, Nphi = u.shape
            salm = sfpy.forward(u, spins, lmax)
            print salm
        else:
            salm = u
            spins = u.spins
            lmax = u.lmax
        r_salm = sfpy.sfpy_salm(np.empty_like(salm, dtype=np.typeDict['complex']), spins + 1, lmax)
        r_salm[:,0] = 2
        print r_salm
        for j in range(lmax):
            sm_values = salm[:,j]
            eigenvalues = -np.sqrt(j*(j+1)-spins*(spins+1))
            eigenvalues = np.lib.stride_tricks.as_strided(
                eigenvalues,
                sm_values.shape,
                (eigenvalues.itemsize, 0)
                )
            r_salm[:,j] = eigenvalues * sm_values
        print r_salm
        if Ntheta is not None:
            r_u = sfpy.backward(r_salm, Ntheta, Nphi)
            return r_u
        else:
            return r_salm
        
  
    def __repr__(self):
        return "eth"

#    def _multiply(j, eigenvalues, spins, sm_values):
#        r_sm_values = np.empty_like(sm_values)
#        for i, v in enumerate(eigenvalues):
#            s = spins[i]
#            if s == j:
#                r_sm_values[i] = np.zeros_like(sm_values[i])
#                continue
#            sp1 = s + 1
#            ind_sp1 = spins[spins==sp1]
#            if ind_sp1.shape[0]!=1:
#                raise ValueError("Array of spins either contains duplicates or does not contain a spin value needed for this calculation"
#            ind_sp1 = ind_sp1[0]
#            r_sm_values[i] = eigenvalues[i]*sm_values[ind_sp1]
#        return r_sm_values
        
class ethp(object):
    
    def __init__(self):
        """
        Implements the differential operator given by
        \eth'{}_sY_{lm} = \sqrt{l(l+1)-s(s-1)}{}_{s-1}Y_{lm}
        """

    def __call__(self, u, spins = None, lmax = None):
        if spins is not None and lmax is not None:
            Ntheta, Nphi = u.shape
            salm = sfpy.forward(u, spins, lmax)
        else:
            salm = u
            spins = u.spins
            lmax = u.lmax
        r_salm = alm.salm(np.empty_like(u), spins - 1, lmax)
        for j in range(lmax):
            sm_values = salm[:,j]
            eigenvalues = np.sqrt(j*(j+1)-spins*(spins+1))
            r_salm[:,j] = eigenvalues * sm_values
        if not spectral:
            r_u = sfpy.backward(r_salm, Ntheta, Nphi)
            return r_u
        else:
            return r_salm
  
    def __repr__(self):
        return "ethp"
        
if __name__ == "__main__":
    Nphi = 50
    Ntheta = 50
    spins = np.array([0])
    lmax = 3
    f = np.empty((Ntheta, Nphi))
    f[:] = 0.5 * (1/math.sqrt(math.pi))
    eth_test = eth()
    eth_f = eth_test(f, spins, lmax)
    print "f = %s"%repr(f)
    print "eth_f = %s"%repr(eth_f)
    f_salm =  sfpy.forward(f, spins, lmax)
    eth_f_salm = eth_test(f_salm)
    print "f_salm = %s"%repr(f_salm)
    print "eth_f_salm = %s"%repr(eth_f_salm)
