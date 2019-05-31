Scripts
=======

COFFEE has a handful of useful scripts located in the `coffee/scripts` directory.
These scripts manipulate the hdf files that coffee produces from a simulation (if
instructed to do so). Documentation for each script can be found by executing:

    `python <script_name> -h`

The scripts are listed below along with a short description. 

.. contents::

compare
-------
Takes two hdf files and prints a comparison of their file structure.

dataOverTime
------------
Takes 5 arguments: the hdf file, the data type (`raw` or `weyl`), an optional
output name (defaults to `gif`), an optional file name and an optional derived
attribute name. 

The script then plots the specified data as it evolves over each computational
iteration using `Gnuplot`.

details
-------
Prints out information about the file structure of a single hdf file.

error
-----
Computes the relative errors of a series of computations against the 'best'
simulation. Best in this case means the simulation with the largest comparison
parameter.

extract
-------
Prints data values from a given hdf file. 

Takes 5 arguments: the hdf file, an identifier to a particular simulation, the
timeslice index, the component of the data to print and the index of the grid
point whose data is desired.

simulate
--------
This script automates initiation of computation for a simulation, extraction
of data and visualisation.

viewFunction3D
--------------
Produces visualisation of 3D data in an hdf file.

visualise
---------
Supports the creation of simple visualisations of data.

