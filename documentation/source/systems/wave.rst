Wave examples
=============

There are two examples of wave equation systems under the `systems/wave` 
directory.
Each subdirectory contains an example.
Each such directory contains two files:

    `<name of example>.py`
    `<name of example>_setup.py`

The file ending with `_setup.py` is the setup file. It contains a variety of
settings which control the nature of the simulation. The other file contains
the system class which describes the equation being solved. The example
can be run by calling the setup file:

    `python <name of example>_setup.py`

ethethp
-------
This is an example of the wave equation on a sphere implemented using the eth
and ethp differential operators.

OneDwave
--------
A simple one dimensional wave equation. The system file contains a variety
of boundary functions.
