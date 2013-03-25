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
    for data in rawData:
        if raw_input('Calculate for %s (y/n)?' % data)=='y':
        
            #Get the number of grid intervals
            #Our matrix with by square with this number of rows/columns
            datagroup = rawData.require_datagroup(data)
            dt = datagroup[1].attrs['time']+datagroup[0].attrs['time']
            num_grid_intervals = datagroup[0].attrs['domain'].shape[0]
       
            #Set up the 2d array for the operator
            D = numpy.zeros((num_grid_intervals, num_grid_intervals))
            
            #Construct operator for 4th order spatial difference
            for i in range(D.shape[0]):
                for j in range(D[i].shape[0]):
                    if (i-2)%num_grid_intervals == j:
                        D[i, j] = 1/12
                    elif (i-1)%num_grid_intervals == j:
                        D[i, j] = -2/3
                    elif (i+1)%num_grid_intervals == j:
                        D[i, j] = 2/3
                    elif (i+2)%num_grid_intervals == j:
                        D[i, j] = -1/12
            identity = numpy.matrix(numpy.identity(num_grid_intervals))
            D = numpy.matrix(D)
            
            #Include RK4 terms in operator
            RK4 = numpy.zeros_like(D)
            for i in range(5):
                RK4 += (dt*D)**i/math.factorial(i)
            RK4 = numpy.asarray(RK4)
            
            #collect eigenvalues and vectors sorted by norm
            #of eigenvalue    
            evalues, evectors = numpy.linalg.eig(RK4)
            evalvect = zip(evalues,evectors.T)
            evalvect = numpy.array(sorted(evalvect,lambda x,y:\
                cmp(abs(x[0]),abs(y[0]))),numpy.dtype('object'))
                      
                      
            #check that sorted was done correctly
            #for val,vect in evalvect:
            #    j = numpy.where(evalues == val)[0][0]
            #    print "Difference"
            #    print "Value Dif = %f"%(val-evalues[j])
            #    print "Vector Dif = %s"%str(vect-evectors[:,j])
            
            #add data to hdf file
            try: del eigenDatas[data]
            except: pass
            eigenData = eigenDatas.require_group(data)
            eigenData['operator'] = RK4
            eigenData['eigenvalues'] = evalues
            eigenData['eigenvectors'] = evectors
            domain = datagroup.attrs['grid'].getGrid(-1,1)
            eigenData['domain'] = domain
            
            #plot egienvalues and unit circle
            y = []
            x = []
            for num in evalues:
                y += [num.imag]
                x += [num.real]
            eigen =  Gnuplot.Data(x, y)
            u = []
            v = []
            for i in range(200):
                u += [numpy.sin(numpy.pi*i/100)]
                v += [numpy.cos(numpy.pi*i/100)]
            unitc = Gnuplot.Data(u, v)
            g = Gnuplot.Gnuplot()
            g('set terminal gif')
            g(r'set output "RK4_eigenvalues_dt'+str(dt)+r'.gif"')
            g.plot(eigen)
            
            #Do eigen vector decomposition of solution and
            #add to hdf file
            eigenDecomp = eigenData.require_datagroup('eigen decomposition')
            for i in range(len(datagroup)):
                v = datagroup[i].value
                eigenDecomp[i] = numpy.dot(v,evectors)
