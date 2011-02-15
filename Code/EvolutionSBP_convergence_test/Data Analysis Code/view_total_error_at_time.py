import sys
sys.path.append("../bin/Code")
import numpy
import h5py_array
import Gnuplot
import datagroup_utils

#sys.argv[1] = 'advection_periodic_boundary31-1-2011.hdf'
#sys.argv[2] = 'Exact 2.0'
#sys.argv[3] = 'gif'
#sys.argv[4] = 'L1_error_1.5_periodic.gif'

def convergence_rate(x, n, y, m):
    return -numpy.log2(x/y)/numpy.log2(n/m)

with h5py_array.H5pyArray(sys.argv[1]) as file:
    data = file.require_group("Comparison Data")
    compData = data.require_datagroup(sys.argv[2])
    totalErrors = datagroup_utils.Lp_norms(compData, 1)
    g = Gnuplot.Gnuplot()
    g('set terminal '+sys.argv[3])
    g('set out \"'+sys.argv[4]+"\"")
    g('set style data linespoints')
    g('set title "L_1 norm from %s at time %.3f" enhanced' % (sys.argv[1],  compData.attrs['time']))
    g('set xlabel "Log base 2 of grid step size"')
    g('set ylabel "Log base 2 of Sum of all errors"')
    x = numpy.array([ 2.0/200, 2.0/400, 2.0/800, 2./1600])
    y = numpy.array(totalErrors)
    print "Convergence Table"
    print str(2./x[0])+"  "+str(totalErrors[0])
    for i in range(1, len(totalErrors)):
        print str(2./x[i])+"  "+str(totalErrors[i])+"  "+str(convergence_rate(totalErrors[i-1], 2./x[i-1], totalErrors[i], 2./x[i]))
    for i in range(x.shape[0]):
        print "Step size = %f, Error value = %f" % (x[i], y[i])
    x = numpy.log2(x)
    y = numpy.log2(y)
    g.plot(Gnuplot.Data(x,y))
    first_slope = (y[1]-y[0])/(x[1]-x[0])
    second_slope = (y[2]-y[1])/(x[2]-x[1])
    third_slope = (y[2]-y[0])/(x[2]-x[0])
    print "Slope between first and second points is "+str(first_slope)
    print "Slope between second and third points is "+str(second_slope)
    print "Slope between first and third points is "+str(third_slope)
    raw_input("Press a key to continue")
    print 'Done'



    
