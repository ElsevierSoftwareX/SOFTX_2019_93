#!/usr/bin/env python
# encoding: utf-8
"""
spin2firstorder.py

Created by Jörg Frauendiener on 2010-12-24.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

import numpy as np
import math

from ibvp import *
from tslices import *
from actions import *
from solvers import RungeKutta4
from system import System
from diffop import *



def bump(r,m,s):
    global g
    a = r[0]
    b = r[-1]
    f = lambda (t): t*b + (1-t)*a
    r0 = f(m)
    r1 = f(m-s)
    r2 = f(m+s)
    print(a,r1,r0,r2,b)
    bmp = (( (r-r1)/(r0-r1) )*( (r-r2)/(r0-r2) ))**4
    bmp[ r < r1 ] = 0
    bmp[ r > r2 ] = 0
    
    return  ( bmp )



#############################################################################
class spin2_SBP(System):
	"""docstring for spin2_SBP"""
	def __init__(self, D, CFL = 1.0, tau = 2., L = 2):
		super(spin2_SBP, self).__init__()
		self.L = L
		self.D = D
		self.tau = tau
	
	def evaluate(self, t, Psi):
		psi0, psi1, psi2, psi3, psi4 = tuple(Psi.fields[k] for k in range(Psi.numFields))
		r	= Psi.x
		dr	= Psi.dx
		dr2 = dr*dr
		l = self.L
		ll	= math.sqrt(l*(l+1))
		ll2 = math.sqrt(l*(l+1)-2)
		tau = self.tau/dr
		
		dpsi1 = (-psi1 + 0.5*(ll2*psi0 - ll*psi2)/r)
		dpsi2 = (0.5*ll*(psi1 - psi3)/r)
		dpsi3 = ( psi3 - 0.5*(ll2*psi4 - ll*psi2)/r)

		
		# implementation follows Carpenter et al.
		# using the SAT method
		
		# impose the equation at every point
		dpsi0 =  self.D(psi0,dr) + (psi0 - ll2 * psi1)/r
		dpsi4 = -self.D(psi4,dr) - (psi4 - ll2 * psi3)/r

		
		# at the boundaries we need boundary conditions
		# implemented as penalty terms
		# use g_{NN} = 1, g_{00} = -1, and |lambda| = 1 here
		# note that g_{00} lambda >0 and g_NN lambda > 0 in those points
		# where the boundary condition needs to be imposed
		
		dpsi4[ 0] -= tau*(psi4[ 0] + psi0[ 0])
		dpsi0[-1] -= tau*(psi0[-1] + psi4[-1])
		
		# now all time derivatives are computed
		# package them into a time slice and return
		
		return timeslice((dpsi0,dpsi1,dpsi2,dpsi3,dpsi4),r,time=t)
	
	def initialValues(self, t0, grid):
		assert(grid.dim == 1), "Grid must be 1-dimensional"
		r = np.linspace(1.0,2.0,grid.shape[0])
		psi0 = bump(r,0.3,0.25)
		psi1 = np.zeros_like(r)
		psi2 = np.zeros_like(r)#bump(r,0.5,0.25)
		psi3 = np.zeros_like(r)
		psi4 = np.zeros_like(r)#bump(r,0.7,0.25)
		return timeslice((psi0, psi1, psi2, psi3, psi4),r,time=t0)
	
	def left(self,t):
		return 0.0

	
	def right(self,t):
		return math.sin(t)




#############################################################################
def main():
	"""docstring for main"""

	rinterval = Grid((201,))
	rk4 = RungeKutta4()
	system = spin2_SBP(D = D43_CNG(), CFL = 2., tau = 2.5)
	problem = IBVP(rk4, system, rinterval, \
			action = Plotter(frequency = 1, xlim = (1,2), ylim = (-5,5), findex = (0,2,4), delay = 0.0))

	problem.run(0.0, 10.0)	







if __name__ == '__main__':
	main()
