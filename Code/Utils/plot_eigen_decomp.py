from __future__ import division
import numpy
import Gnuplot
import math
import sys

sys.path.append("../bin")
import h5py_array
import datagroup_utils

#sys.argv[1] = 'exp_-20xx_simple.hdf'

with h5py_array.H5pyArray(sys.argv[1]) as file:
    rawData = file.require_group("/Raw Data")
    eigenDatas = file.require_group("Eigen Data")
    for data in eigenDatas:
        if raw_input('Calculate for %s (y/n)?' % data)=='y':                        
            #plot this data
            eigenData = eigenDatas[data]
            plots = []
            domain = eigenData['domain']
            for i in range(5):
                plots += [Gnuplot.Data(range(len(, numpy.real(eigenDecomp[i][0]))]
            g = Gnuplot.Gnuplot()
            g('set style data lines')
            g.plot(*plots)
            raw_input("press key to finish")
