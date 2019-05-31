swsh
====
The `swsh` module contains code handling the requirements for representation
of functions as a sum of spin weighted spherical harmonics. COFFEE code is
able to perform conversion of functions valued over spheres into such sums
and vice versa. Depending on the differential operator used it is possible
to perform computations only using the sum representation. For this purpose
routines to handle Wigner 3j and Clebsch-Gordan symbols are included.

Use of this module requires compilation and installation of c code. Please
refer to the `c_code/` and `spinsfastpy/` directories for installation 
instructions.

.. contents::

salm
----
.. automodule:: coffee.swsh.salm
    :members:
    :special-members:

w3j
---
.. autodata:: coffee.swsh.w3j.NUMERICAL_ERROR_TOLERANCE
    :annotation: The numerical tolerance used in float comparisons.


.. autofunction:: coffee.swsh.w3j.valid
.. autofunction:: coffee.swsh.w3j.trivial_zero

.. autoclass:: coffee.swsh.w3j.W3jStone
    :members:
    :special-members:

.. autoclass:: coffee.swsh.w3j.W3jBen
    :members:
    :special-members:

.. autoclass:: coffee.swsh.w3j.W3jBoris
    :members:
    :special-members:


clebschgordan
-------------
.. autodata:: coffee.swsh.clebschgordan.NUMERICAL_ERROR_TOLERANCE
    :annotation: The numerical tolerance used in float comparisons.

.. autofunction:: coffee.swsh.clebschgordan.valid

.. autoclass:: coffee.swsh.clebschgordan.CGW3j
    :members:
    :special-members:

.. autoclass:: coffee.swsh.clebschgordan.CGStone
    :members:
    :special-members:

.. autoclass:: coffee.swsh.clebschgordan.CGBoris
    :members:
    :special-members:

spinsfastpy
-----------
.. automodule:: coffee.swsh.spinsfastpy
    :members:
    :special-members:

.. autofunction:: coffee.swsh.spinsfastpy.forward
.. autofunction:: coffee.swsh.spinsfastpy.backward
