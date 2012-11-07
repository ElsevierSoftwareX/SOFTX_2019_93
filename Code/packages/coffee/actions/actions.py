import Gnuplot
import time
import os
#from pylab import *
import numpy as np
import logging

import os.path


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

class Info(Prototype):
    def __init__(self, *args, **kwds):
        self.log = logging.getLogger("Info")
        super(Info, self).__init__(*args, **kwds)

    def _doit(self, it, u):
        self.log.info("Iteration: %d, Time: %f" % (it, u.time))
