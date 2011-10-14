#!/usr/bin/env python
# encoding: utf-8
"""
FFT_diffop.py

Created by Ben Whale on 2011-09-05.
Copyright (c) 2011 University of Otago. All rights reserved.
"""
from __future__ import division

import cmath
import numpy as np
import scipy
from scipy import fftpack


################################################################################
# Finite difference differential operators.
################################################################################

class FFT_lagrange1(object):
    
    name = "FFT_lagrange1"
    
    def __init__(self,num_grid_points,period):
        N = num_grid_points
        assert N%2 == 0
        """The derivative matrix is calculated using the 'Negative Sum
        Trick'."""
        M = np.empty((N,N))
        for i in range(N):
            M[i,i] = 0
            for j in range(N):
                if j != i:
                    M[i,j]= (1/2)*(-1)**(i+j)*\
                        (1/np.tan( ((i-j)*np.pi) / (N ) ))
        for i in range(N):
            M[i,i] = np.sum(sorted(M[i,:]))
        self.M = M*(2*np.pi)/period
            
    def __call__(self,u,dx):
        ru = np.empty_like(u)
        ru[:-1] = np.dot(self.M,u[:-1])
        ru[-1] = ru[0]
        return ru

class FFT(object):
           
    name = "FFT"
    
    def __init__(self,order,period):
        self.order = order
        self.period = period
           
    def __call__(self,u,dx):
        #transform into fourier space 
        ufft = np.fft.fft(u)
        #collect frequencies at each index
        ufreq = np.fft.fftfreq(ufft.size,d=dx)
        #compute derivative coefficient for each frequency
        dufreq = (2*cmath.pi*cmath.sqrt(-1)*ufreq)**(self.order)
        #compute fourier domain derivatives
        dufft = dufreq*ufft
        #transform into 'normal' space
        rdufft = np.fft.ifft(dufft)
        return rdufft
   
class RFFT(object):
           
    name = "RFFT"
           
    def __init__(self,order,period):
        self.order = order
        self.period = period
           
    def __call__(self,u,dx):
        #get length of array
        n = u.shape[0]
        #transform into fourier space
        ufft = np.fft.rfft(u)
        #collect frequencies at each index
        ufreq = np.fft.fftfreq(ufft.size, d=dx)
        #compute derivative coefficient for each frequency
        dufreq = (2*cmath.pi*cmath.sqrt(-1)*ufreq)**(self.order)
        #compute fourier domain derivatives
        dufft = dufreq*ufft
        #transform into 'normal' space
        rdufft = np.fft.irfft(dufft,n)
        return rdufft

class FFT_diff_scipy(object):

    name = "FFT_diff_scipy"
    
    def __init__(self,order,period):
        self.order = order
        self.period = period
    
    def __call__(self,u,dx):
        du = np.empty_like(u)
        du[:-1] = fftpack.diff(u[:-1],self.order,self.period)
        du[-1] = du[0]
        return du

class FFT_scipy(object):
    
    name = "FFT_scipy"
    
    def __init__(self,order,period):
        self.order = order
        self.period = period
    
    def __call__(self,u,dx):
        ufft = fftpack.fft(u)
        ufreq = fftpack.fftfreq(ufft.size,d=dx)
        dufreq = ((2*cmath.pi*cmath.sqrt(-1)*ufreq)**(self.order))
        dufft = dufreq*ufft
        rdufft = fftpack.ifft(dufft)
        return rdufft
        
class RFFT_scipy(object):
    """The scipy implementation of RFFT doesn't seem to behave as well as
    the numpy RFFT implementation I suggest using FFT_diff_scipy instead."""
    
    name = "RFFT_scipy"
    
    def __init__(self,order):
        self.order = order        
    
    def __call__(self,u,dx):
        n = u.shape[0]
        ufft = fftpack.rfft(u)
        ufreq = fftpack.rfftfreq(ufft.size)
        dufreq = ((2*cmath.pi*cmath.sqrt(-1)*ufreq)**(self.order))
        dufft = dufreq*ufft
        rdufft = fftpack.ifft(dufft,n)
        return rdufft/(dx**self.order)
