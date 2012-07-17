"""
An action is a class which performs some additinonal processing to a time_slice
between time steps.

Please see the help for the prototype action for implementation details.

Implemented actions:
BlowupCutoff -- Terminates evolution if a value of the evolution crosses a 
                 given threshhold
GNUPlotter1D -- Plots a vector of unknowns over a 1-d grid using Gnuplot
GNUPlotter2D -- Plots a vector of unknowns over a 2-d grid using Gnuplot
                 EXPERIMENTAL
Plotter -- Plots a vector of unknowns over a 1-d grid using matplotlib
           EXPERIMENTAL
Info -- Prints information to std_out regarding the simulation
SimOutput -- Writes time_slices to HDD using h5py
"""

from actions import Prototype, BlowupCutoff, Plotter, Info
from gnuplot import GNUPlotter1D, GNUPlotter2D
from hdf_output import SimOutput

del hdf_output
del actions
del gnuplot
