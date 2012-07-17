"""
This is the package spinsfastpy. It contains python bindings to Huffenberger's &
Wandelt's spinsfast C code. We recommend using the import statement: 
import spinsfast as sfpy.

The principle objective of this code is to provide a simple interface to the C
code. For this reason we expose only as much of the API of spinsfast as 
necessary.

Importing this module provides two methods, forward and backward, and one 
dictionary, methods. Please see the individual documentation for how to use
forward and backward. The keys of the methods dictionary are the valid arguments
for the keyword argument delta_method in forward and backward. This controls
how the wigner d function, evaluated at math.pi/2, are calculated.

The package contains three modules forward_transform, backward_transform
and alm. Each module contains bindings to the spinsfast code that can be 
accessed by importing these modules, e.g.: 
from spinsfast import forward_transform.

Some example code, both C and Python can be found in ./examples.
"""
__all__ = ['forward','backward','salm']

from forward_transform import forward
from backward_transform import backward
from delta_matrix import methods
from salm import sfpy_salm

del forward_transform
del backward_transform
del delta_matrix
del salm

def set_clebsch_gordan_default(cg):
    sfpy_salm.cg_default = cg

