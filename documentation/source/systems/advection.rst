Advection examples
==================

The four examples under the directory `systems/advection` provide examples of using
COFFEE to solve an advection equation. 
Each subdirectory contains an example.
Each such directory contains two files:

    `<name of example>.py`
    `<name of example>_setup.py`

The file ending with `_setup.py` is the setup file. It contains a variety of
settings which control the nature of the simulation. The other file contains
the system class which describes the equation being solved. The example
can be run by calling the setup file:

    `python <name of example>_setup.py`

OneDAdvection
-------------
A basic advection equation.

OneDAdvectionMpi_fd
-------------------
An implementation of the one dimensional advection equation suitable for use with
MPI using a finite difference operator.

OneDAdvectionMpi_sbp
--------------------
An implementation of the one dimensional advection equation suitable for use with
MPI using a summation by parts operator.

TwoDAdvection
-------------
The two dimensional advection equation on a sphere using summation by parts and
penalty boundaries.
