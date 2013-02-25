from __future__ import division
import numpy
import Gnuplot
import math
import sys
import time

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
            vect = eigenData['eigenvectors'].value
            val = eigenData['eigenvalues'].value
            domain = eigenData['domain']
            g = Gnuplot.Gnuplot()
            g('set style data lines')
            for i,v in enumerate(vect.swapaxes(0,1)):
                g.title("abs(eigenvalue) = %f"%val[i])
                g.plot(Gnuplot.Data(domain, numpy.real(v)))#ect[:,float(sys.argv[2])])))
                time.sleep(0.1)
            raw_input("press key to finish")
