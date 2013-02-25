#!/usr/bin/env python
# encoding: utf-8 
"""
tslices.py

Created by Jörg Frauendiener on 2010-11-17.
Additional development by Ben Whale since then.

This module contains the abstract base class (abc) for TimeSlice objects and a
default implementation that makes certain assumptions. It is highly recomended
that you subclass either of these classes.

As usual the abc is intended to make you think twice about doing things
differently. All other classes that interact with TimeSlice objects assume that
the interface is as given below.

Now it should be noted that the variables ABCTimeSlice.data,
ABCTimeSlice.domain and ABCTimeSlice.time should have been declared as
abc.abstractproperty. Doing this is possible, but because of a limitation in
the abc class for Python 2 a substantial amount of additional code is needed.
So this hasn't been done. In fact NONE of the methods or variables are declared
as abstract. 

What is recommended is that you subclass ABCTimeSlice and call it's functions
once appropriate checks have been carried out.

At some point this will need to be changed... I assume after the move to Python
3 is made.

TimeSlice objects should contain all information needed to interperet the
values of the functions being numerically evolved.

Classes:
ABCTimeSlice - the abc for TimeSlice objects
TimeSlice - a default implementation of ABCTimeSlice that assume that all data
            is stored in a numpy array.
"""

import abc
import logging
import numpy as np

###############################################################################
# TimeSlice Abstract Base Class
##############################################################################
class ABCTimeSlice(object):
    """The abstract base class for TimeSlice objects.

    The interface below is assumed by all other classed that interact with
    TimeSlice objects (which is just about everything).

    """
    __metaclass__ = abc.ABCMeta
 
    def __init__(self, data, domain, time, name=None, *args, **kwds):
        """Make a TimeSlice instance!

        It is recommended to call this method once your subclass has done
        appropriate vetting of data, domain and time. Not only because this
        ensure things are stored in the right places, but also because it sets
        up a logging object, logging.getLogger(self.name).

        Positional Arguments:
        data - the values of the functions being solved for
        domain - the grid over which the values are calcualted
        time - the time for which the values in data are valid

        Keyword Arguments:
        name - the name of your subclass. This defaults to ABCTimeSlice

        """
        self.data = data
        self.domain = domain
        self.time = time
        if name is None:
            self.name = "ABCTimeSlice"
        else:
            self.name = name
        self.log = logging.getLogger(self.name)
        super(ABCTimeSlice, self).__init__(*args, **kwds)

    def __repr__(self):
        """Returns a naive string representation of the slice.

        Returns - read above!

        """
        s = "%s(data = %s, domain = %s, time = %s)"%(
            self.name, 
            repr(self.data), 
            repr(self.domain), 
            repr(self.time)
            )
        return s

    def boundary_slices(self):
        """Returns a list of tuples that describe boundaries of grids for
        external boundaries. 

        Returns - returns the result of a call to
                  self.domain.boundary_slices(self.data.shape)

        """
        return self.domain.boundary_slices(self.data.shape)
    
    def communicate(self):
        """Returns a list of tuples that describe the boundaries of
        grids for internal boundaries.

        This is currently used for when the full domain is divided into
        subdomains for use with mpi.

        Returns - returns the result of a call to
                  self.domain.communicate(self.data)

        """
        return self.domain.communicate(self.data)

    def collect_data(self):
        """Returns a timeslice containing the almalgum of all data and domains
        across all subdomains.

        This is currently used to pass non-distributed data to actions in the
        ibvp method.

        Returns - a single timeslice which represents the complete data and
                  domain for that data.

        """
        data_all = self.domain.collect_data(self.data)
        domain_all = self.domain.full_grid
        r_tslice = self.__class__(
            data_all, domain_all, self.time
            )
        if __debug__:
            self.log.debug(
                "r_tslice is %s"%(repr(r_tslice))
                )
        return r_tslice

###############################################################################
# Concrete implementations
###############################################################################
class TimeSlice(ABCTimeSlice):
    """A default subclass of ABCTimeSlice.

    This implementation assumes that all data are numpy arrays of the same
    shape. If your data is not like this then you should subclass ABCTimeSlice.

    Due to the assumption that the data is represented as a numpy array, this
    class also implements a large number of methods which allow this object to
    be added, multiplied, etc...

    To be honest rather than implementing all the additional methods by hand
    it'd be easier just to make this default TimeSlice a subclass of np.ndarray
    itself and allow TimeSlice.data to access the underlying array.

    Methods:
    __init__ - constructor

    """

    def __init__(self,data, *args, **kwds):
        """Returns a TimeSlice object.

        Arguments:
        data - converts data to a numpy array before passing it on to
               ABCTimeSlice.

        """
        data = np.array(data)
        if "name" not in kwds:
            kwds["name"] = "TimeSlice"
        super(TimeSlice, self).__init__(data, *args, **kwds) 


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
            raise NotImplementedError(
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
                raise NotImplementedError(
                    "Addition of %s and %s is not implemented"
                    %(self, other)
                    )
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            rv = self.data - other
        except:
            raise NotImplementedError(
                "Subtraction of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __isub__(self, other):
        try:
            self.data -= other
        except:
            raise NotImplementedError(
                "In place subraction of %s and %s is not implemented"
                %(self, other)
                )
        return self

    def __rsub__(self, other):
        try:
            r_time_slice = other - self.data
        except:
            raise NotImplementedError(
                "Reflected subtraction of %s and %s is not implemented"
                %(other, self)
                )
        return r_time_slice
    
    def __mul__(self, other):
        try:
            rv = self.data * other
        except:
            raise NotImplementedError(
                "Multiplication of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __imul__(self, other):
        try:
            self.data *= other
        except:
            raise NotImplementedError(
                "In place multiplicatio of %s and %s is not implemented"
                %(self, other)
                )
        return self

    def __rmul__(self, other):
        return self * other
    
    def __div__(self, other):
        try:
            rv = self.data / other
        except:
            raise NotImplementedError(
                "Division of %s and %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __idiv__(self, other):
        try:
            self.data /= other
        except:
            raise NotImplementedError(
                "In place division of %s and %s is not implemented"
                %(self, other)
                )
        return self
    
    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.idiv(other)

    def __pow__(self, other):
        try:
            rv = self.data ** other
        except:
            raise NotImplementedError(
                "Exponentiation of %s by %s is not implemented"
                %(self, other)
                )
        return TimeSlice(rv, self.domain, self.time)

    def __ipow__(self, other):
        try:
            self.data **= other
        except:
            raise NotImplementedError(
                "In place exponentiation of %s by %s is not implemented"
                %(self, other)
                )
        return self
