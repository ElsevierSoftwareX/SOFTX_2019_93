Welcome to COFFEE's documentation
=================================

COFFEE stands for COnFormal Field Equation Evolver. Originally the code
was designed to compute solutions to Friedrich's conformal field equations.
Over time, however, the library developed into a collection of code used to solve
hyperbolic systems of differential equations that implemented finite differences or
spectral methods and the method of lines.

The repository contains the COFFEE module itself as well as a handful of 
scripts for working with the output of COFFEE and several example systems
of ODEs and PDEs that can be solved using COFFEE. 

It is assumed that in reading this documentation you have also read the
accompanying SoftwareX paper. This documentation provides technical information
about the code while the paper gives an overview of the structure of the code
as well as information about how to use COFFEE to solve a system of equations.

.. toctree::
   :maxdepth: 2

   installation
   examples
   using_coffee
   coffee
   scripts

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
