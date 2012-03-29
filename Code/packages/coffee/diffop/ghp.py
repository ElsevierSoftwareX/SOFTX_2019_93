import math
import os

import spinsfast as sfpy

class eth(object):
    name = "eth"

    def __init__(self, radius, spectral=False, spins=None, lmax=None):
        """
        If spectral = False then a list of spins and the maximum l must be
        provided
        """
        self.radius = radius
        self.spectral = spectral
        self.spins = spins

    def __call__(self,u,dx):
        if not spectral:
            Nmaps, Ntheta, Nphi = u
            salm = sfpy.forward(u, spins, lmax)
            spins = self.spins
            lmax = self.lmax
        else:
            salm = u
            spins = u.spins
            lmax = u.lmax
        r_salm = alm.salm(np.empty_like(u), spins, lmax)
        factor = -math.sqrt(1/(2*self.radius*self.radius))
        for j in range(lmax):
            sm_values = salm.l(j)
            eigenvalues = factor * math.sqrt( (j+spins+1) * (j-spins) )
            r_salm.l(j) = _multiply(j, eignvalues, spins, sm_values)
        if not spectral:
            r_u = sfpy.backward(r_salm, Ntheta, Nphi)
            return r_u
        else:
            return r_salm
  
    def __str__(self):
        return "eth"

    def _multiply(j, eigenvalues, spins, sm_values):
        r_sm_values = np.empty_like(sm_values)
        for i, v in enumerate(eigenvalues):
            s = spins[i]
            if s == j:
                r_sm_values[i] = np.zeros_like(sm_values[i])
                continue
            sp1 = s + 1
            ind_sp1 = spins[spins==sp1]
            if ind_sp1.shape[0]!=1:
                raise ValueError("Array of spins either contains duplicates or does not contain a spin value needed for this calculation"
            ind_sp1 = ind_sp1[0]
            r_sm_values[i] = eigenvalues[i]*sm_values[ind_sp1]
        return r_sm_values
        

