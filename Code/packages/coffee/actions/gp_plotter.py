import Gnuplot
import time
import os
import numpy as np
import logging

from coffee.actions import Prototype

class Plotter1D(Prototype):
    
    def __init__(self, system, *args, **kwds):
        if 'start' in kwds:
            start = kwds.pop('start')
        else:
            start = -float('Infinity')
        if 'frequency' in kwds:
            frequency = kwds.pop('frequency')
        else:
            frequency = 1
        super(Plotter1D, self).__init__(frequency = frequency, start = start)
        if 'delay' in kwds:
            self.delay = kwds.pop('delay')
        else:
            self.delay = 0.0
        if 'title' in kwds:
            self.title = kwds.pop('title')
        if self.title is None:
            self.title = r"Time %f"
        self.system = system
        self.log = logging.getLogger("GNUplotter")
        try:
            if kwds['data_function'] is not None:
                self.datafunc = kwds.pop('data_function') 
            else:
                self.datafunc = lambda y,x,z:x.data
        except:
            self.datafunc = lambda y,x,z:x.data
        if __debug__:
            self.log.debug("Initialising plotter...")
        self.Device = Gnuplot.Gnuplot()
        g = self.Device
        for arg in args:
            g(arg)
        if __debug__:
            self.log.debug("Done.-")

    def _doit(self, it, u):
        g = self.Device
        x = u.domain.axes[0]
#        if __debug__:
#            self.log.debug("Plotting iteration %i with data %s"%(it,str(u)))
        f = np.atleast_2d(self.datafunc(it,u,self.system))
        if __debug__:
            self.log.debug("Data after processing by self.datafunc is %s"%f)
            self.log.debug(
                "Shape of domain to plot over is %s"%x.shape
                )
            self.log.debug("Domain to plot over is %s"%repr(x))
        graphs = []
        g(self.title%u.time)
        for i, val in enumerate(f):
            if __debug__:
                self.log.debug(
                    "Shape of data to be plotted is %s"%val.shape
                    )
                self.log.debug("Data to be plotted is %s"%repr(val))
            graphs += [Gnuplot.Data(x, val,\
                title = "Component %i"%i)]
        g.plot(*graphs)
        time.sleep(self.delay)
    
    def __del__(self):
        del self.Device

class Plotter2D(Prototype):
    
    def __init__(self, *args, **kwds):
        if 'delay' in kwds:
            self.delay = kwds.pop('delay')
        else:
            self.delay = 0.0
            
        if 'title' in kwds:
            self.title = kwds.pop('title')
        else:
            self.title = "Iteration %d, Time %f"
        
        if 'components' in kwds:
            self.components = kwds.pop('components')
        else:
            self.components = None
        
        self.system = kwds.pop('system')
        #self.log = logging.getLogger("GNUPlotter2D")
        try:
            if kwds['data_function'] is not None:
                self.datafunc = kwds.pop('data_function') 
            else:
                self.datafunc = lambda y,x,z:x.data
        except:
            self.datafunc = lambda y,x,z:x.data

#        if 'start' in kwds:
#            start = kwds.pop('start')
#        else:
#            start = -float('Infinity')
#        if 'frequency' in kwds:
#            frequency = kwds.pop('frequency')
#        else:
#            frequency = 1
#        super(GNUPlotter2D,self).__init__(frequency = frequency, start = start)
#        if __debug__:
#            self.log.debug("Initialising plotter...")
        self.device = Gnuplot.Gnuplot()
        for arg in args:
            self.device(arg)
        super(Plotter2D, self).__init__(**kwds)
#        if __debug__:
#            self.log.debug("Done.-")

    def _doit(self, it, u):
        x = u.x
#        if __debug__:
#            self.log.debug("Plotting iteration %i with data %s"%(it,str(u)))
        f = np.atleast_2d(self.datafunc(it, u, self.system))
#        if __debug__:
#            self.log.debug("Data after processing by self.datafunc is %s"%f)
        graphs = []
        self.device('set title "%s" enhanced'%self.title%(it, u.time))
        for i in range(f.shape[0]):
            if self.components is None:
                graphs += [Gnuplot.GridData(
                    f[i,:,:],
                    xvals = x.axes[0],
                    yvals = x.axes[1],
                    filename="deleteme.gp",
                    title = "Component %i"%i,
                    binary = 0
                    )]
            else:
                graphs += [Gnuplot.GridData(
                    f[i,:,:],
                    xvals = x.axes[0],
                    yvals = x.axes[1],
                    filename="deleteme.gp",
                    title = self.components[i],
                    binary = 0
                    )]
        self.device.splot(*graphs)
        time.sleep(self.delay)
    
    def __del__(self):
        try:
            os.remove("deleteme.gp")
        except:
            print "The file deleteme.gp was unable to be deleted. Please do so manually."
        del self.device
