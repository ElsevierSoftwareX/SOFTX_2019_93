import numpy
import sys
import h5py_array
import datagroup_utils
import Gnuplot

sys.argv[1] = 'advection_24-1-2011.hdf'

with h5py_array.H5pyArray(sys.argv[1]) as file:
    compData = file.require_datagroup("Comparison Data")
    datagroup = file[file.keys()[0]]
    datagroup_utils.GNUplot(datagroup)


    
