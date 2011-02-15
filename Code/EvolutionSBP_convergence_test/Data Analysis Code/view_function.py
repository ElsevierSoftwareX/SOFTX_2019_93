import sys
sys.path.append("../bin")
import numpy
import h5py_array
import h5py
import datagroup_utils
import tslices

#sys.argv[1] = 'exp_-20xx.hdf'
#sys.argv[2] = '2.0'

with h5py_array.H5pyArray(sys.argv[1]) as file:
    rawData = file.require_group("/Raw Data")
    for data in rawData:
        if  raw_input("Would you like to view %s (y,n)?" %data) =='y':
            datagroup_utils.GNUplot(rawData.require_datagroup(data))
    print "Finished"



    
