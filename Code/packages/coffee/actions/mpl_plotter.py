import time
from pylab import *
import numpy as np

class Plotter(Prototype):
    """docstring for Plotter"""
    def __init__(self, frequency = 1, xlim = (-1,1), ylim = (-1,1), findex = None, delay = 0.0):
        super(Plotter, self).__init__(frequency)
        if findex is not None:
            self.index = findex
            self.delay = delay
            self.colors = ('b','g','r','c','m','y','k','coral') 
            from numpy import asarray
            ion()
            fig = figure(1)
            ax = fig.add_subplot(111)
            self.lines = [ ax.add_line(Line2D(xlim,ylim)) for k in findex ] 
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)       
            ax.grid(True)
            self.axes = ax
    
    def _doit(self, it, u):
        x = u.x
        index = np.asarray(self.index)
        f = u.fields[index]
        mx = np.max(f.flat)
        mn = np.min(f.flat)
        self.axes.set_title("Iteration: %d, Time: %f" % (it,u.time))
        #self.axes.set_xlim(x[0],x[-1])
        #self.axes.set_ylim(mn,mx,auto=True)

        l = len(self.index)
        ioff()
        for k in range(l):
            line = self.lines[k]
            line.set_xdata(x)
            line.set_ydata(f[k])
            line.set_color(self.colors[k])
        ion()
        draw()
        time.sleep(self.delay)

