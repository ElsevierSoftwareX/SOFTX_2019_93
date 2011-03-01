from __future__ import division
import numpy
import numpy as np
import Gnuplot
import math
import sys

sys.path.append("../bin")
import h5py_array
import datagroup_utils

#sys.argv[1] = 'exp_-20xx_simple.hdf'

def exp_bump(x):
    return numpy.exp(-20*x*x)

def dexp_bump(x):
    return -20*x*exp_bump(x)
    
def smooth_non_analytic(x):
    def f(y): 
        if y<=0.0:
            return 0.0
        else:
            return np.exp(-1/y)
    return np.vectorize(f)(x)

def dsmooth_non_analytic(x):
    def f(y): 
        if y<=0.0:
            return 0.0
        else:
            return np.exp(-1/y)/(y**2)
    return np.vectorize(f)(x)

def dsmooth_step(x, start, stop):
    y = (x-start)/(stop-start)
    first_term = (dsmooth_non_analytic(y)) / (smooth_non_analytic(y)+smooth_non_analytic(1-y))
    second_term = -smooth_non_analytic(y)*(dsmooth_non_analytic(y)-dsmooth_non_analytic(1-y))/\
        (smooth_non_analytic(y)+smooth_non_analytic(1-y))**2
    return (first_term+second_term)/(stop-start)
        
def smooth_step(x, start, stop):
    y = (x-start)/(stop-start)
    return (smooth_non_analytic(y))/ \
        (smooth_non_analytic(y)+\
                smooth_non_analytic(1-y))

def smooth_bump(x, start_up, stop_up,start_down, stop_down):
    return smooth_step(x, start_up, stop_up)*\
        (1-smooth_step(x, start_down, stop_down))

def dsmooth_bump(x, start_up, stop_up,start_down, stop_down):
    return dsmooth_step(x, start_up, stop_up)*(1-smooth_step(x, start_down, stop_down))-\
        smooth_step(x, start_up, stop_up)*dsmooth_step(x, start_down, stop_down)

with h5py_array.H5pyArray(sys.argv[1]) as file:
    rawData = file.require_group("/Raw Data")
    eigenDatas = file.require_group("Eigen Data")
    derivative_error = []
    for data in rawData:        
            #Get the number of grid intervals
            #Our matrix with by square with this number of rows/columns
            datagroup = rawData.require_datagroup(data)
            num_grid_intervals = datagroup[0].attrs['domain'].shape[0]
            domain = datagroup.attrs['grid'].getGrid(-1,1)
            dx = domain[1]-domain[0]
            
            #Set up the 2d array for the operator
            D = numpy.zeros((num_grid_intervals, num_grid_intervals))
            
            #Construct operator for 4th order spatial difference
            for i in range(D.shape[0]):
                for j in range(D[i].shape[0]):
                    if (i-2)%num_grid_intervals == j:
                        D[i, j] = 1/12
                    elif (i-1)%num_grid_intervals == j:
                        D[i, j] = -2/3
                    elif (i+1)%num_grid_intervals == j:
                        D[i, j] = 2/3
                    elif (i+2)%num_grid_intervals == j:
                        D[i, j] = -1/12
            
            #u = numpy.dot(D,exp_bump(domain))/(2*dx)
            #u = numpy.dot(D,smooth_step(domain,-0.9,-0.1))/(dx)
            u = numpy.dot(D,smooth_bump(domain,-0.9,-0.1 ,0.1, 0.9))/(dx)
            #v = dexp_bump(domain)
            #v = dsmooth_step(domain,-0.9,-0.1)
            v = dsmooth_bump(domain,-0.9,-0.1 ,0.1, 0.9)
            derivative_error +=[(numpy.abs(u-v),domain)]
            
    x = derivative_error[-1]
    derivative_error[1:] = derivative_error[0:-1]
    derivative_error[0] = x
    g = Gnuplot.Gnuplot()
    g('set style data lines')
    plots = []
    for i, data  in enumerate(derivative_error):
        plots += [Gnuplot.Data(data[1],numpy.log2(data[0])+4*i,title = "%i"%i)]
    g.title("Error in spatial derivative with +4*i scaling")
    g('set yrange [-50:-0]')
    g('set ylabel "log2 of absolute value of error with scaling"')
    #g('set xrange [-0.91:-0.09]')
    g.plot(*plots)
    raw_input("press key to finish")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
