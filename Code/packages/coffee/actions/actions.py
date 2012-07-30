import Gnuplot
import time
import os
#from pylab import *
import numpy as np
import logging

class Prototype(object):
    """The prototype of all actions. 
    
    This class provides basic functionality that all actions required.
    
    To subclass this class:
    1) define the method _doit in the subclass
    2) call the Prototypes constructor to ensure that frequency, start and stop
       are properly initialised.
    """

    def __init__(self, frequency = 1, start = -float('infinity'),\
            stop = float('infinity')\
            ):
        self.frequency = frequency
        self.stop = stop
        self.start = start
    
    def will_run(self,it,u):
        return (it % self.frequency) == 0 and self.start<=u.time<=self.stop
    
    def __call__(self, it, u):
        if self.will_run(it,u):
            self._doit(it, u)
    
    def _doit(self, it, u):
        pass

class BlowupCutoff(Prototype):

    def __init__(self, cutoff = 10, frequency = 1, start = -float('infinity'),\
        stop = float('infinity') ):
        super(BlowupCutoff,self).__init__(frequency,start,stop)
        self.cutoff = cutoff

    def above_cutoff(self,u):
        for component in u.fields:
            if np.any(component>=self.cutoff):
                return True
        return False

    def _doit(self, it,u):
        if self.above_cutoff(u):
            raise Exception("Function values are above the cutoff")

#class Plotter(Prototype):
#    """docstring for Plotter"""
#    def __init__(self, frequency = 1, xlim = (-1,1), ylim = (-1,1), findex = None, delay = 0.0):
#        super(Plotter, self).__init__(frequency)
#        if findex is not None:
#            self.index = findex
#            self.delay = delay
#            self.colors = ('b','g','r','c','m','y','k','coral') 
#            from numpy import asarray
#            ion()
#            fig = figure(1)
#            ax = fig.add_subplot(111)
#            self.lines = [ ax.add_line(Line2D(xlim,ylim)) for k in findex ] 
#            ax.set_xlim(xlim)
#            ax.set_ylim(ylim)       
#            ax.grid(True)
#            self.axes = ax
#    
#    def _doit(self, it, u):
#        x = u.x
#        index = np.asarray(self.index)
#        f = u.fields[index]
#        mx = np.max(f.flat)
#        mn = np.min(f.flat)
#        self.axes.set_title("Iteration: %d, Time: %f" % (it,u.time))
#        #self.axes.set_xlim(x[0],x[-1])
#        #self.axes.set_ylim(mn,mx,auto=True)
#
#        l = len(self.index)
#        ioff()
#        for k in range(l):
#            line = self.lines[k]
#            line.set_xdata(x)
#            line.set_ydata(f[k])
#            line.set_color(self.colors[k])
#        ion()
#        draw()
#        time.sleep(self.delay)


class Info(Prototype):
    """docstring for info"""
    
    def _doit(self, it, u):
        print("Iteration: %d, Time: %f" % (it, u.time))
