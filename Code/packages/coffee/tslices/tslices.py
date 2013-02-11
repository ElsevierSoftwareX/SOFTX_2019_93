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
import logging
import numpy as np
import functools

###############################################################################
# TimeSlice Abstract Base Class
##############################################################################
class ABCTimeSlice(object):

    __metaclass__ = abc.ABCMeta
 
    def __init__(self, data, domain, time, name=None, *args, **kwds):
        self.data = np.array(data)
        self.domain = domain
        self.time = time
        if name is None:
            self.name = "ABCTimeSlice"
        else:
            self.name = name
        self.log = logging.getLogger(self.name)
        super(ABCTimeSlice, self).__init__(*args, **kwds)

    def __repr__(self):
        s = "%s(data = %s, domain = %s, time = %s)"%(
            self.name, 
            repr(self.data), 
            repr(self.domain), 
            repr(self.time)
            )
        return s

    def boundary_slices(self):
        return self.domain.boundary_slices(self.data.shape)
    
    def communicate(self):
        return self.domain.communicate(self.data)

    def barrier(self):
        return self.domain.barrier()

    def collect_data(self):
        data_all = self.domain.collect_data(self.data)
        domain_all = self.domain.full_grid
        r_tslice = self.__class__(
            data_all, domain_all, self.time
            )
        #if __debug__:
            #self.log.debug(
                #"r_tslice is %s"%(repr(r_tslice))
                #)
        return r_tslice

###############################################################################
# Concrete implementations
###############################################################################
class TimeSlice(ABCTimeSlice):
    """This implementation assumes that fields is an object that understands
    addtion."""   

    def __init__(self, *args, **kwds):
        if "name" not in kwds:
            kwds["name"] = "TimeSlice"
        super(TimeSlice, self).__init__(*args, **kwds) 

    #def _check_valid(other):
        #if isinstance(other, ABCTimeSlice):
            #if other.domian != self.domain and self.time != other.time:
                #return ValueError("%s and %s cannot be added as they either
                #have different domains or differnt times"%(
                #repr(self), repr(other)
                #)

    #def require_valid_tslice(f):
        #@functools.wraps
        #def is_valid_for_arithmatic(f):
            #def wrapper(*args, **kwds):

    def __add__(self, other):
        # It is very important that the rv is other + self.data not
        # self.data + other.
        # The reason (seems) is because as self.data can be an 
        # nd.array the sum self.data + other
        # ends up being computed as the elements of self.other
        # plus other, itself an nd.array.
        # This screws up the array of elements.
        # Putting other first ensures that if other is a timeslice
        # then the sum isn't distributed over the elements of self.data.
        try:
            rv =  other + self.data
        except:
            return NotImplementedError(
                "Addition of %s and %s is not implemented"
                %(self, other)
                )
        if isinstance(rv, TimeSlice):
            return rv
        else:
            return TimeSlice(rv, self.domain, self.time, name=self.name)

    def __iadd__(self, other):
        if isinstance(other, TimeSlice):
            self.data += other.data
        else:
            try:
                self.data += other
            except:
                return NotImplementedError(
                    "Addition of %s and %s is not implemented"
                    %(self, other)
                    )
        return self

    def __radd__(self, other):
        return self + other
        #try:
            #r_time_slice = other + self.data
        #except:
            #return NotImplementedError(
                #"Reflected addition of %s and %s is not implemented"
                #%(other, self)
                #)
        #return r_time_slice

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
            r_time_slice = other - self.data
        except:
            return NotImplementedError(
                "Reflected subtraction of %s and %s is not implemented"
                %(other, self)
                )
        return r_time_slice
    
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
        return self * other
        #try:
            #r_time_slice = other * self.data
        #except:
            #return NotImplementedError(
                #"Reflected multiplicatio of %s and %s is not implemented"
                #%(other, self)
                #)
        #return r_time_slice
    
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

    #def __rtruediv__(self, other):
        #return self.__rdiv__(other) 
    
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
