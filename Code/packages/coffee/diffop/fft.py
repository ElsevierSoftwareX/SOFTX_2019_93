#!/usr/bin/env python
# encoding: utf-8
"""
FFT_diffop.py

Created by Ben Whale on 2011-09-05.
Copyright (c) 2011 University of Otago. All rights reserved.
"""
from __future__ import division

import cmath, math
import numpy as np
import scipy
import fftw3 as fftw
from scipy import fftpack
import pdb

#import logging



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
        dufreq = (2*np.pi*1j*ufreq)**(self.order)
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
        du = fftpack.diff(u,self.order,self.period)
        #du[:-1] = fftpack.diff(u[:-1],self.order,self.period)
        #du[-1] = du[0]
        return du

class FFTW(object):

    name = "FFTW"
    
    def __init__(self,order,period = None,fftw_flags = ['estimate']):
        self.order = order
        self.period = period
        self.fftw_flags = fftw_flags
        self.u = None
        self.dufreq = None
        #self.log = logging.getLogger("FFTW3")
        
    def __call__(self,u,dx):
        if self.u is None:
            self.u = np.empty_like(u, dtype = np.dtype(np.complex128))
            self.ufft = np.empty_like(self.u)
            self.dufft = np.empty_like(self.u)
            self.du = np.empty_like(self.u)
            self.fft = fftw.Plan(self.u,self.ufft,direction="forward")
            self.ifft = fftw.Plan(self.dufft,self.du,direction="backward")
        self.fft.guru_execute_dft(u.astype(np.dtype(np.complex128)),self.ufft)
        if self.dufreq is None:
            ufreq = np.array([self._compute_freq(i,self.ufft.shape[0],dx[0]) for i in range(self.ufft.shape[0])])
            self.dufreq = np.power(2*np.pi*1j*ufreq, self.order)
        self.dufft = self.dufreq*self.ufft
        self.ifft.guru_execute_dft(self.dufft,self.du)
        return self.du/u.shape[0]

    def _compute_freq(self,index,size,dx):
        if size%2 == 0:
            mid = size/2
        else:
            mid = (size-1)/2+1
        if index < mid:
            rfreq = index
        else:
            rfreq =  -(size - index)
        return rfreq/(size*dx)

class FFTW_real(object):

    name = "FFTW_real"
    
    def __init__(self,order,period = None,fftw_flags = ['estimate']):
        self.order = order
        self.period = period
        self.fftw_flags = fftw_flags
        self.u = None
        self.dufreq = None
        #self.log = logging.getLogger("FFTW3")
        
    def __call__(self,u,dx):
        if self.u is None:
            self.u = np.empty_like(u)
            self.ufft = np.empty((math.floor((u.shape[0]/2)+1),), dtype = np.dtype(np.complex128))
            self.dufft = np.empty_like(self.ufft)
            self.du = np.empty_like(u)
            self.fft = fftw.Plan(self.u,self.ufft,direction="forward")
            self.ifft = fftw.Plan(self.dufft,self.du,direction="backward")
        self.fft.guru_execute_dft(u,self.ufft)
        if self.dufreq is None:
            ufreq = np.array([self._compute_freq(i,self.ufft.shape[0],dx[0]) for i in range(self.ufft.shape[0])])
            self.dufreq = np.power(2*np.pi*1j*ufreq, self.order)
        self.dufft = self.dufreq*self.ufft
        self.ifft.guru_execute_dft(self.dufft,self.du)
        return self.du/u.shape[0]

    def _compute_freq(self,index,size,dx):
        if size%2 == 0:
            mid = size/2
        else:
            mid = (size-1)/2+1
        if index < mid:
            rfreq = index
        else:
            rfreq =  -(size - index)
        return rfreq/(size*dx)

class FFT_scipy(object):
    
    name = "FFT_scipy"
    
    def __init__(self,order,period = None):
        self.order = order
        self.period = period
        #self.log = logging.getLogger("FFT_scipy")
    
    def __call__(self,u,dx):
        ufft = fftpack.fft(u)
        ufreq = fftpack.fftfreq(ufft.size,d=dx)
        #self.log.debug("ufft = %s"%repr(ufft))
        #self.log.debug("ufreq = %s"%repr(ufreq))        
        dufreq = np.power(2*np.pi*1j*ufreq,self.order)
        #self.log.debug("dufreq = %s"%repr(dufreq))        
        dufft = dufreq*ufft
        #self.log.debug("dufft = %s"%repr(dufft))
        du = fftpack.ifft(dufft)
        #self.log.debug("du = %s"%repr(du))
        return du
        
class RFFT_scipy(object):
    """The scipy implementation of RFFT doesn't seem to behave as well as
    the numpy RFFT implementation I suggest using FFT_diff_scipy instead."""
    
    name = "RFFT_scipy"
    
    def __init__(self,order,period):
        self.order = order        
    
    def __call__(self,u,dx):
        n = u.shape[0]
        ufft = fftpack.rfft(u)
        ufreq = fftpack.rfftfreq(ufft.size)
        dufreq = ((2*np.pi*1j*ufreq)**(self.order))
        dufft = dufreq*ufft
        rdufft = fftpack.ifft(dufft,n)
        return rdufft/(dx**self.order)
        
class FFTW_convolve(object):

    name = "FFTW"
    
    def __init__(self,order,period = None,fftw_flags = ['estimate']):
        self.order = order
        self.period = period
        self.fftw_flags = fftw_flags
        #self.log = logging.getLogger("FFTW3")
        
    def __call__(self,u,dx):
        du = np.empty_like(u)
        du = fftpack.diff(u,self.order,self.period)
        return du

    _cache = {}
    def _diff(x,order=1,period=None,
                _cache = _cache):
        """This the is same as fftpack.diff, with the difference that fftw3
        routines are used in the convolution.
        """
        tmp = np.asarray(x)
        if order==0:
            return tmp
        if iscomplexobj(tmp):
            return diff(tmp.real,order,period)+1j*diff(tmp.imag,order,period)
        if period is not None:
            c = 2*pi/period
        else:
            c = 1.0
        n = len(x)
        omega = _cache.get((n,order,c))
        if omega is None:
            if len(_cache)>20:
                while _cache: _cache.popitem()
            def kernel(k,order=order,c=c):
                if k:
                    return pow(c*k,order)
                return 0
            omega = fftpack.convolve.init_convolution_kernel(n,kernel,d=order,
                                                     zero_nyquist=1)
            _cache[(n,order,c)] = omega
        return _convolve_FFTW3(tmp,omega,swap_real_imag=order%2)
    del _cache
    
    def _convolve_FTW3(tmp, omega, swap_real_imag = 0):
        n = tmp.shape[0]
        if self.in_array is None:
            self.in_array = np.empty_like(tmp, dtype = np.dtype(np.complex128))
            self.out_array = np.empty_like(self.in_array)
            self.fft = fftw.Plan(self.in_array, self.out_array, direction="forward")
            self.ifft = fftw.Plan(self.out_array, self.in_array, direction="backward")
        self.fft.guru_execute_dft(tmp, self.out_array)
        if swap_real_imag:
            out_array[0] = out_array[0]*omega[0]
            if not n%2:
                out_array[n-1] = out_array[n-1]*omega[n-1]
            for i in range(1,n-1,2):
                c = out_array[i]*omega[i]
                out_array[i] = out_array[i+1]*omega[i+1]
                out_array[i+1] = c
        else:
            for i in range(n):
                out_array[i] = out_array[i]*omega[i]
        self.ifft.guru_execute_dft(self.out_array, self.in_array)
        return self.in_array/n
    
