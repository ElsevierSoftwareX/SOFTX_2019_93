#!/usr/bin/env python
# encoding: utf-8

# imports from external libraries
from __future__ import division
import math
import numpy as np
import abc
import logging

# imports from coffee
from coffee.mpi import mpiinterfaces

################################################################################
# Base Boundary data class
################################################################################
class ABCBoundary(object):
    """The abstract base class for boundary classes.

    A boundary class manages data associated to ghost points and the slices
    specifying which data points should be communicated"""

    __metaclass__ = abc.ABCMeta

    # I use -1 and 1 to reduce friction with the mpi direction variable
    LEFT = -1
    RIGHT = 1

    DIRECTION_ERROR = ValueError("Direction must be +/- 1")

    @staticmethod
    def _direction_to_index(direction):
        if direction == -1:
            return 0
        elif direction == 1:
            return 1
        raise DIRECTION_ERROR

    @staticmethod
    def _empty_slice(dimensions):
        return [slice(None, None, None) for i in range(dimensions)]

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def ghost_points(self, dimension, direction):
        """Returns the number of ghost points for the specified dimension and
        direction."""
        return 0

    @abc.abstractmethod
    def internal_slice(self, shape, dimension, direction):
        """Returns a tuples of slices which, when applied
        to a data array, gives the data to be communicated to 
        neighbouring grids."""
        return self._empty_slice(shape)

    def internal_slices(self, shape):
        """Returns a list of tuples. Each tuple contains an integer,
        representing the dimension, a direction and the result of 
        calling internal_slice(dimension, direction).

        This is a convience method to make iteration over external
        boundaries easy for the user."""
        neg_slices = [
            (i, -1, self.internal_slice(shape, i, -1)) 
            for i in range(self.number_of_dimensions)
        ]
        pos_slices = [
            (i, 1, self.internal_slice(shape, i, 1)) 
            for i in range(self.number_of_dimensions)
        ]
        return pos_slices + neg_slices

    @abc.abstractmethod
    def external_slice(self, shape, dimension, direction):
        """
        Returns a tuple which, when applied
        to a data array, gives the data on the extenal boundaries of the grid.
        
        """
        slices = self._empty_slice(len(shape))

        # +1 because for some reason I decided that vector data should be in the
        # first dimension.
        dim_index = dimension + 1
        if direction == -1:
            slices[dim_index] = slice(None, 1, None)
            return slices

        if direction == 1:
            slices[dim_index] = slice(-1, None, None)
            return slices

        raise DIRECTION_ERROR

    def external_slices(self, shape):
        """Returns a list of tuples. Each tuple contains an integer,
        representing the dimension, a direction and the result of 
        calling external_slice(dimension, direction).

        This is a convience method to make iteration over external
        boundaries easy for the user."""

        # -1 to account for first dimension of shape being number of data 
        # components
        neg_slices = [
            (i, -1, self.external_slice(shape, i, -1)) 
            for i in range(len(shape) - 1 ) 
            if self.external_slice(shape, i , -1) != self._empty_slice(len(shape))
        ]
        pos_slices = [
            (i, 1, self.external_slice(shape, i, 1)) 
            for i in range(len(shape) - 1)
            if self.external_slice(shape, i , -1) != self._empty_slice(len(shape))
        ]
        return pos_slices + neg_slices
 

################################################################################
# Base Boundary data class
################################################################################
class SingleCartesianGridBoundary(ABCBoundary):
    """Implements ABCBoundary for a simulation on a single grid with no 
    mpi dependence.

    That is a grid with no internal boundaries and every grid "edge" an 
    external boundary."""
    pass

class MPIBoundary(ABCBoundary):
    """MPIBoundary implements the ABCBoundary class. The numbers of ghost points
    and "internal points" can be specified directly using arrays of two tuples."""
    def __init__(
            self, 
            ghost_points, 
            internal_points, 
            mpi_comm=None, 
            *args, 
            **kwargs
        ):
        """ 
        The ghost points and internal points variables must both be an array 
        or tuple with one entry for each dimension. Each entry is a 2-tuple giving 
        the number of points in the negative and positive directions for 
        that axis.
        """
        super(MPIBoundary, self).__init__(*args, **kwargs)
        self._ghost_points = ghost_points
        self._internal_points = internal_points
        self.mpi_comm = mpi_comm

    def ghost_points(self, dimension, direction):
        return self._ghost_points[dimension][self._direction_to_index(direction)]

    def source_and_dest(self, dimension, direction):
        return self.mpi_comm.Shift(dimension, direction)

    def internal_slice(self, shape, dimension, direction):
        source, dest = self.source_and_dest(dimension, direction)

        if dest < 0:
            rsend_slice = None
        else:
            send_slice = self._empty_slice(len(shape))
        if source < 0:
            rrecv_slice = None
        else:
            recv_slice = self._empty_slice(len(shape))

        i_point = self._internal_points[dimension][
            self._direction_to_index(direction)
        ]
        g_point = self._ghost_points[dimension][
            self._direction_to_index(direction)
        ]
        total_g_points = sum(self._ghost_points[dimension])

        # +1 because the first dimension of the data array is reserved for vector
        # valued data. This may not have been a good desicion
        #
        # The details of why the start and end points for the slices are as they
        # are is complicated. Double check things before you change anything.
        dim_index = dimension + 1

        # This functions are used to to avoid difficulties resulting from the
        # difference between slice(-1, 0) and slice(-1, None), we don't need one
        # for "pos_or_none" as slice(0, x) and slice(None, x) behave the same
        def neg_or_none(number):
            return number if number < 0 else None

        if direction == 1:
            if not dest < 0:
                send_slice[dim_index] = slice(
                    -total_g_points, 
                    neg_or_none(-total_g_points + i_point), 
                    None
                )
                rsend_slice = tuple(send_slice)
            if not source < 0:
                recv_slice[dim_index] = slice(
                    None,
                    i_point,
                    None
                )
                rrecv_slice = tuple(recv_slice)
        elif direction == -1:
            if not dest < 0:
                send_slice[dim_index] = slice(
                    max(0, total_g_points - i_point),
                    total_g_points, 
                    None
                )
                rsend_slice = tuple(send_slice)
            if not source < 0:
                recv_slice[dim_index] = slice(
                    -i_point,
                    None,
                    None
                )
                rrecv_slice = tuple(recv_slice)
        return rsend_slice, rrecv_slice

    def external_slice(self, shape, dimension, direction):
        if self.mpi_comm.periods[dimension]:
            return self._empty_slice(len(shape))
        coords = self.mpi_comm.Get_coords(self.mpi_comm.rank)
        dim = self.mpi_comm.dims[dimension]
        if coords[dimension] == 0 and direction == -1:
            return super(MPIBoundary, self).external_slice(
                shape, 
                dimension, 
                -1
            )
        if coords[dimension] == dim - 1 and direction == 1:
            return super(MPIBoundary, self).external_slice(
                shape,
                dimension, 
                1
            )
        return self._empty_slice(len(shape))

class SimpleMPIBoundary(MPIBoundary):
    """SimpleMPIBoundary is for boundaries in mpi contexts with a fixed number of
    ghost points in every direction and dimension and the "internal points" equal
    to the ghost points."""

    def __init__(self, ghost_points, *args, **kwargs):
        number_of_dimensions = kwargs["number_of_dimensions"]
        points_tuple = [
            (ghost_points, ghost_points) for d in range(number_of_dimensions)
        ]
        super(SimpleMPIBoundary, self).__init__(
            points_tuple,
            points_tuple,
            *args,
            **kwargs
        )
    
class GeneralBoundary(ABCBoundary):
    """MPIBoundary implements ABCBoundary for grids in an mpi context with arbitrary
    ghost points and arbitrary interal and external slices fixed at run time.
    
    The ghost points variable must be an array or tuple with one entry for each
    dimension. Each entry is a 2-tuple giving the number of ghost points in the
    negative and positive directions for that axis.

    The data slices variable is an array with one entry for each dimension. Each
    entry is a tuple of slices which when applied to the data stored in a tslice
    gives the data to be communicated / received.
    """

    def __init__(self, ghost_points, internal_slices, external_slices):
        self._ghost_points = ghost_points
        self._interal_slices = internal_slices
        self._external_slices = external_slices

    def ghost_points(self, dimension, direction):
        return self._ghost_points[dimension][self._direction_to_index(direction)]

    def internal_slices(self, shape, dimension, direction):
        return self._internal_slices[dimension][self._direction_to_index(direction)]

    def external_slices(self, shape, dimension, direction):
        return self._external_slices[dimension][self._direction_to_index(direction)]


################################################################################
# Base Grid class
################################################################################
class ABCGrid(object):
    """The abstract base class for Grid objects."""
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, 
        shape, bounds, name = "Grid", comparison = None,
        mpi = None, boundary_data = None, *args, **kwds
        ):
        self.mpi = mpi
        self.dim = len(shape)
        self.name = name
        self.log = logging.getLogger(name)
        self.comparison = comparison
        self.shape = shape
        self.bounds = bounds
        self.boundary_data = boundary_data
    
    def __strs__(self):
        return self.name

    def __repr__(self):
        return "<%s shape=%s, bounds=%s, comparison=%s, mpi=%s>"%(
            self.name, self.shape, self.bounds, self.comparison, self.mpi
            )

    def communicate(self, data, ghost_point_processor=None):
        if self.mpi is None:
            return []
        b_values = self.mpi.communicate(data, self.boundary_data)
        if ghost_point_processor:
            ghost_point_processor(data, b_values)
        return data, b_values

    def barrier(self):
        if self.mpi is None:
            return 
        return self.mpi.barrier()

    def external_slices(self, data_shape):
        if __debug__:
            self.log.debug("In grid.external_slices")
        return self.boundary_data.external_slices(data_shape)

    def collect_data(self, data):
        if self.mpi is None:
            return data
        return self.mpi.collect_data(data)
    
    @abc.abstractproperty
    def axes(self): pass
    
    @abc.abstractproperty
    def step_sizes(self): pass
    
    @property
    def meshes(self):
        axes = self.axes
        grid_shape = tuple([axis.size for axis in axes])
        mesh = []
        for i, axis in enumerate(axes):
            strides = np.zeros((len(self.shape), ))
            strides[i] = axis.itemsize
            mesh += [np.lib.stride_tricks.as_strided(
                axis,
                grid_shape,
                strides)
                ]
        return mesh

################################################################################
# Constructors for specific cases
################################################################################
class UniformCart(ABCGrid):
    
    def __init__(self, 
            shape, 
            bounds, 
            mpi_comm=None, 
            comparison=None, 
            name=None, 
            *args, **kwds):
        _shape = tuple([s+1 for s in shape])
        if mpi_comm is None:
            mpi = None
        else:
            mpi = mpiinterfaces.EvenCart(
                _shape, 
                kwds.get("boundary_data", None),
                mpi_comm=mpi_comm, 
                )
        if name is None:
            name = "UniformCart%s%s%s"%(shape, bounds, comparison)
        super(UniformCart, self).__init__(
            shape, bounds, 
            name=name, comparison=comparison,
            mpi=mpi, *args, **kwds
            ) 
        _axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            )
            for i in range(len(self.bounds))
            ]
        self._step_sizes = [axis[1]-axis[0] for axis in _axes]
        if self.mpi is None:
            self._axes = _axes
        else:
            self._axes = [
                axis[self.mpi.subdomain[i]] 
                for i, axis in enumerate(_axes)
                ]

    @property
    def axes(self):
        return self._axes

    @property
    def full_grid(self):
        if self.mpi is None:
            return self
        return UniformCart(
            self.shape, self.bounds, 
            comparison=self.comparison, name=self.name
            )
        
    @property
    def step_sizes(self):
        return self._step_sizes

class GeneralGrid(ABCGrid):

    def __init__(self, 
            shape, 
            bounds, 
            periods,
            #mpi_comm=None, 
            comparison=None, 
            name=None, 
            *args, 
            **kwds
        ):
        _shape = []
        for i,p in enumerate(periods):
            if p:
                _shape.append(shape[i])
            else:
                shape.append(shape[i]+1)
        _shape = tuple(_shape)
        mpi=None
        #mpi = mpiinterfaces.EvenCart(
            #_shape, 
            #kwds.get("boundary_data", None),
            #mpi_comm=mpi_comm, 
            #)
        if name is None:
            name = "<GeneralGrid shape=%s, bounds=%s,periods=%s, comparison=%s>"%(
                shape, 
                bounds, 
                periods,
                comparison)
        super(UniformCart, self).__init__(
            shape, bounds, 
            name=name, 
            comparison=comparison,
            mpi=mpi, 
            *args, 
            **kwds
            ) 
        _axes = [
            np.linspace(
                self.bounds[i][0], self.bounds[i][1], self.shape[i]+1
            )
            for i in range(len(self.bounds))
            ]
        self._step_sizes = [axis[1]-axis[0] for axis in _axes]
        #self._axes = [axis[self.mpi.subdomain] for axis in _axes]

    @property
    def axes(self):
        return self._axes

    @property
    def full_grid(self):
        if self.mpi is None:
            return self
        return UniformCart(
            self.shape, self.bounds, 
            comparison=self.comparison, name=self.name
            )
        
    @property
    def step_sizes(self):
        return self._step_sizes
