grid
====

.. contents::

boundary classes
----------------

Boundaries are classes that understand how computational domains relate to the
MPI network topology. They manage the constructions of lists of tuples of slices.
Each tuple of slices describes the location of data that is sufficiently close
to the edge of the computational domain to require either communication with
other MPI nodes or boundary conditions.

.. autoclass:: coffee.grid.grid.ABCBoundary
    :members: 
    :special-members:
    :private-members: _direction_to_index, _empty_slicempi_comm: mpi_commmpi_comm: mpi_comm

.. autoclass:: coffee.grid.grid.SingleCartesianGridBoundary
    :members: 
    :special-members:

.. autoclass:: coffee.grid.grid.MPIBoundary
    :members: 
    :special-members:

.. autoclass:: coffee.grid.grid.SimpleMPIBoundary
    :members: 
    :special-members:

.. autoclass:: coffee.grid.grid.GeneralBoundary
    :members: 
    :special-members:

grid classes
------------

A grid class understands the data associated with the some subset of the
computational domain of the system. They wrap a boundary class that manages the
information associated to the edges of the computational domain.

.. autoclass:: coffee.grid.grid.ABCGrid
    :members:
    :undoc-members: __strs__, __repr__
    :special-members: __init__, __strs__, __repr__

.. autoclass:: coffee.grid.grid.UniformCart
    :members:
    :special-members: __init__

.. autoclass:: coffee.grid.grid.GeneralGrid
    :members:
    :special-members: __init__


