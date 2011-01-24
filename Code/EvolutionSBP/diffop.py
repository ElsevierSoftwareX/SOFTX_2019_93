#!/usr/bin/env python
# encoding: utf-8
"""
diffop.py

Created by Jörg Frauendiener on 2011-01-10.
Copyright (c) 2011 University of Otago. All rights reserved.
"""

import sys
import os
import unittest
import math
import numpy as np

class diffop(object):
    name = "Dx"
    def __init__(self):
        self.bdyRegion = self.Ql.shape
    
    def __call__(self,u,dx):
        r,c = self.bdyRegion
        du = np.zeros_like(u)
        du = np.convolve(u, self.A, mode='same')
        du[0:r] = np.dot(self.Ql, u[0:c])
        du[-r:] = np.dot(self.Qr, u[-c:])
        return du/dx

    def __str__(self):
        return "Differential operator "%self.name
        
    def save(self):
        filename = os.path.expanduser("~/" + self.name)
        print filename
        np.savetxt(filename + "_left.txt", self.Ql)
        np.savetxt(filename + "_right.txt", self.Qr)
        np.savetxt(filename + "_mid.txt", self.A)

class D42(diffop):
    """docstring for D42"""
    
    def __init__(self):
        self.name = "D42"
        self.A = np.array([-1./12.,2./3.,0.,-2./3.,1./12.])
        self.Ql = np.array( \
            [[-24.0/17.0, 59.0/34.0, -4.0/17.0, -3.0/34.0,         0,         0 ],\
             [-1.0/2.0,           0,   1.0/2.0,         0,         0,         0 ],\
             [4.0/43.0,  -59.0/86.0,         0, 59.0/86.0, -4.0/43.0,         0 ],\
             [3.0/98.0,           0,-59.0/98.0,         0, 32.0/49.0, -4.0/49.0 ]])
        self.Qr = -self.Ql[::-1,::-1]
        super(D42, self).__init__()
        


class D43_Tiglioetal(diffop):
    """D43 is a finite difference operator which has the SBP property.
    It is 4th order accurate in the interior and 3rd order accurate at the boundaries."""
    def __init__(self):
        self.name = "D43_Tiglioetal"
        self.A = np.array([-1./12.,2./3.,0.,-2./3.,1./12.])
        self.Ql = np.array( \
        [\
         [ -2.09329763466349871588733,  4.0398572053206615302160,  -3.0597858079809922953240,   1.37319053865399486354933, -0.25996430133016538255400,   0,                           0],\
         [ -0.31641585285940445272297, -0.53930788973980422327388,  0.98517732028644343383297, -0.05264665989297578146709, -0.113807251750624235013258,  0.039879767889849911803103, -0.0028794339334846531588787 ],\
         [  0.13026916185021164524452, -0.87966858995059249256890,  0.38609640961100070000134,  0.31358369072435588745988,  0.085318941913678384633511, -0.039046615792734640274641,  0.0034470016440805155042908 ],\
         [ -0.01724512193824647912172,  0.16272288227127504381134, -0.81349810248648813029217,  0.13833269266479833215645,  0.59743854328548053399616,  -0.066026434346299887619324, -0.0017244594505194129307249 ],\
         [ -0.00883569468552192965061,  0.03056074759203203857284,  0.05021168274530854232278, -0.66307364652444929534068,  0.014878787464005191116088,  0.65882706381707471953820,  -0.082568940408449266558615  ]\
        ])
        self.Qr = -self.Ql[::-1,::-1]
        super(D43_Tiglioetal, self).__init__()




class D43_CNG(diffop):
    """This operator looks somewhat suspicious. It does not converge at the boundaries"""
    r1 = -(2177.*math.sqrt(295369.) - 1166427.)/(25488.)
    r2 = (66195.*math.sqrt(53.*5573.) - 35909375.)/101952.
    A = np.array([-1./12.,2./3.,0.,-2./3.,1./12.])
    name = "D43_CNG"
    def __init__(self):
        a = self.r1
        b = self.r2
        Q = np.mat(np.zeros((4,7)))
        
        Q[0,0] = -0.5
        Q[0,1] = -(864.*b + 6480*a + 305)/4320.
        Q[0,2] = (216*b + 1620*a + 725)/540.
        Q[0,3] = -(864*b + 6480*a + 3335)/4320
        
        Q[1,0] = -Q[0,1]
        Q[1,1] = 0.0
        Q[1,2] = -(864.*b + 6480*a + 2315)/1440.
        Q[1,3] = (108*b + 810*a + 415)/270
        
        Q[2,0] = -Q[0,2]
        Q[2,1] = -Q[1,2]
        Q[2,2] = 0.0
        Q[2,3] = -(864*b + 6480*a + 785)/4320
        
        Q[3,0] = -Q[0,3]
        Q[3,1] = -Q[1,3]
        Q[3,2] = -Q[2,3]
        Q[3,3] = 0.0
        
        Q[2,4] = -1./12.
        Q[3,5] = -1./12.
        Q[3,4] =  8./12.
        
        P = np.mat(np.zeros((4,4)))
        P[0,0] = -(216*b + 2160*a - 2125)/(12960)
        P[0,1] = (81*b + 675*a + 415)/540
        P[0,2] = -(72*b + 720*a + 445)/(1440)
        P[0,3] = -(108*b + 756*a + 421)/1296
        
        P[1,0] = P[0,1]
        P[1,1] = -(4140*b + 32400*a + 11225)/4320
        P[1,2] = (1836*b + 14580*a + 7295)/2160
        P[1,3] = -(216*b + 2160*a + 665)/(4320)
        
        P[2,0] = P[0,2]
        P[2,1] = P[1,2]
        P[2,2] = -(4104*b + 32400*a + 12785)/4320
        P[2,3] = (81*b + 675*a + 335)/(540)
        
        P[3,0] = P[0,3]
        P[3,1] = P[1,3]
        P[3,2] = P[2,3]
        P[3,3] = -(216*b + 2160*a - 12085)/(12960)
        
        self.Ql = np.dot(np.linalg.inv(P),Q)
        self.Qr = -self.Ql[::-1,::-1]
        
        super(D43_CNG, self).__init__()
        




class D43_Strand(diffop):
    """docstring for D43_Strand"""
    A = np.array([-1./12.,2./3.,0.,-2./3.,1./12.])
    name = "D43_Strand"
    def __init__(self):
        Q = np.mat(np.zeros((5,7)))
        Q[0,0] = -1.83333333333333333333333333333
        Q[0,1] = 3.00000000000000000000000000000
        Q[0,2] = -1.50000000000000000000000000000
        Q[0,3] = 0.333333333333333333333333333333
        Q[0,4] = 0
        Q[0,5] = 0
        Q[0,6] = 0
        Q[1,0] = -0.389422071485311842975177265599
        Q[1,1] = -0.269537639034869460503559633378
        Q[1,2] = 0.639037937659262938432677856167
        Q[1,3] = 0.0943327360845463774750968877551
        Q[1,4] = -0.0805183715808445133581024825052
        Q[1,5] = 0.00610740835721650092906463755990
        Q[1,6] = 0
        Q[2,0] = 0.111249966676253227197631191911
        Q[2,1] = -0.786153109432785509340645292042
        Q[2,2] = 0.198779437635276432052935915726
        Q[2,3] = 0.508080676928351487908752085966
        Q[2,4] = -0.0241370624126563706018867104954
        Q[2,5] = -0.00781990939443926721678719106507
        Q[2,6] = 0
        Q[3,0] = 0.0190512060948850190478223587421
        Q[3,1] = 0.0269311042007326141816664674713
        Q[3,2] = -0.633860292039252305642283500163
        Q[3,3] = 0.0517726709186493664626888177616
        Q[3,4] = 0.592764606048964306931634491846
        Q[3,5] = -0.0543688142698406758774679261355
        Q[3,6] = -0.00229048095413832510406070952285
        Q[4,0] = -0.00249870649542362738624804675220
        Q[4,1] = 0.00546392445304455008494236684036
        Q[4,2] = 0.0870248056190193154450416111553
        Q[4,3] = -0.686097670431383548237962511314
        Q[4,4] = 0.0189855304809436619879348998899
        Q[4,5] = 0.659895344563505072850627735853
        Q[4,6] = -0.0827732281897054247443360556719
        
        self.Ql = Q
        self.Qr = -self.Ql[::-1,::-1]
        
        
        super(D43_Strand, self).__init__()




def main():
    D = D43_CNG()
    D.save()

from matplotlib.pylab import *

def f(x):
    return (x-2)**3
    
def df(x):
    return 3*(x-2)**2

class diffopTests(unittest.TestCase):
    def setUp(self):
        self.D = D43_Strand()
        
    def test_pointconvergence(self):
        power2 = (1, 2, 4, 8, 16, 32)
        D = self.D
        def compute_delta(n):
            N = 20*n
            x = np.linspace(-1,1,N+1)
            dx = 2./N
            ff = f(x)
            dff = df(x)
            delta = np.log(np.abs(D(ff,dx) - dff))*math.log(2)
            return delta[0::n]
            
        delta = map(compute_delta,power2)
        plot(np.asarray(delta).T,'o-')
        grid(True)
        gca().set_title("Pointwise convergence  for %s" % D.name)
        show()
        self.assertTrue(True)
        
    def test_globalconvergence(self):
        power2 = (1, 2, 4, 8, 16, 32)
        D = self.D
        def compute_delta(n):
            N = 20*n
            x = np.linspace(-1,1,N+1)
            dx = 2./N
            ff = f(x)
            dff = df(x)
            delta = np.abs(D(ff,dx) - dff)**2
            l2 = math.sqrt(delta.sum()*dx)
            return math.log(l2)*math.log(2.)

        err = map(compute_delta,power2)
        plot(np.log(np.asarray(power2))*math.log(2.), np.asarray(err),'o-')
        grid(True)
        gca().set_title("Global convergence for %s" % D.name)
        show()
        self.assertTrue(True)
        
        
        
if __name__ == '__main__':
    unittest.main()
#   main()
