#!/usr/bin/env python
# encoding: utf-8 
"""
tslices.py

Created by Jörg Frauendiener on 2010-11-17, modifications by Ben Whale since
then.
Additional development by Ben Whale.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

import abc

###############################################################################
# TimeSlice Abstract Base Class
##############################################################################
class ABCTimeSlice(object):

    __metaclass__ = abc.ABCMeta
    
    def __init__(self, data, domain, time, *args, **kwds):
        self.data = data
        self.domain = domain
        self.time = time
        super(ABCTimeSlice, self).__init__(*args, **kwds)

    #@property
    #def numFields(self):
        #return len(self.data)
    
    def __repr__(self):
        s = "TimeSlice(data = %s, domain = %s, time = %s)"%(
            repr(self.data), 
            repr(self.domain), 
            repr(self.time)
            )
        return s

    #@property
    #def dx(self):
        #return self.step_sizes
      
    #@property
    #def step_sizes(self):
        #return self.domain.step_sizes
        
    #@property
    #def fields(self):
        #return self.data

    #@property        
    #def x(self):
        #return self.domain

    def communicate(self):
        self.domain.sendrecv(self.data)

###############################################################################
# Concrete implementations
###############################################################################
class TimeSlice(ABCTimeSlice):
    """This implementation assumes that fields is an object that understands
    addtion."""   

    def __add__(self, other):
        try:
            rv = self.data + other
        except:
            return NotImplementedError(
                "Addition of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __iadd__(self, other):
        self.data += other
        return self

    def __radd__(self, other):
        try:
            rv = other + self.data
        except:
            return NotImplementedError(
                "Reflected addition of %s and %s is not implemented"
                %(other, self)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __sub__(self, other):
        try:
            rv = self.data - other
        except:
            return NotImplementedError(
                "Subtraction of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __isub__(self, other):
        try:
            self.data -= other
        except:
            return NotImplementedError(
                "In place subraction of %s and %s is not implemented"
                %(self, other)
                )
        return self

    def __rsub__(self, other):
        try:
            rv = other - self.data
        except:
            return NotImplementedError(
                "Reflected subtraction of %s and %s is not implemented"
                %(other, self)
                )
        return TimeSlice(rv, self.domain, self.time)
    
    def __mul__(self, other):
        try:
            rv = self.data * other
        except:
            return NotImplementedError(
                "Multiplicatio of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __imul__(self, other):
        try:
            self.data *= other
        except:
            return NotImplementedError(
                "In place multiplicatio of %s and %s is not implemented"
                %(self, other)
                )
        return self

    def __rmul__(self, other):
        try:
            rv = other * self.data
        except:
            return NotImplementedError(
                "Reflected multiplicatio of %s and %s is not implemented"
                %(other, self)
                )
        return TimeSlice(rv, self.domain, self.time)
    
    def __div__(self, other):
        try:
            rv = self.data / other
        except:
            return NotImplementedError(
                "Division of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __idiv__(self, other):
        try:
            self.data /= other
        except:
            return NotImplementedError(
                "In place division of %s and %s is not implemented"
                %(self, other)
                )
        return self

    #def __rdiv__(self, other):
        #try:
            #rv = other / self.data
        #except:
            #return NotImplementedError(
                #"Reflected division of %s and %s is not implemented"
                #%(other, self)
                #)
        #return TimeSlice(rv, self.domain, self.time)
    
    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.idiv(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other) 
    
    def __pow__(self, other):
        try:
            rv = self.data ** other
        except:
            return NotImplementedError(
                "Exponentiation of %s by %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __ipow__(self, other):
        try:
            self.data **= other
        except:
            return NotImplementedError(
                "In place exponentiation of %s by %s is not implemented"
                %(self, other)
                )
        return self

if __name__ == "__main__":
    import numpy as np
    t = TimeSlice(np.arange(10), np.linspace(0,1,11), 1.4)
    print t
    print t+2
    print t/2
    print t/2.0
    print t.__truediv__(2)
    print t*2
    t += 2
    print t
    t -= 3.0
    print t
    t /= 2.0
    print t
