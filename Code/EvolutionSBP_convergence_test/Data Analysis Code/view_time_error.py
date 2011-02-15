import sys
sys.path.append("../bin")
import numpy
import h5py_array
import Gnuplot

#sys.argv[1] = 'exp_-20xx.hdf'
#sys.argv[2] = '2.0'
#sys.argv[3] = 'gif'
#sys.argv[4] = 'test.gif'

with h5py_array.H5pyArray(sys.argv[1]) as file:
    print "Producing graph..."
    data = file.require_group("Comparison Data/"+sys.argv[2])
    g = Gnuplot.Gnuplot()
    if sys.argv[2] is not None and sys.argv[4] is not None:
        g('set terminal '+sys.argv[3])
        g('set out \"'+sys.argv[4]+"\"")
    g('set style data lines')
    g('set title "Errors from %s at time %.3f" enhanced' % (sys.argv[1],  data.attrs['time']))
    g('set xlabel "r"')
    g('set ylabel "Log base 2 of absolute value of difference"')
    
    plot_data = []
    for i in range(len(data)):
        diff = data[data.keys()[i]]
        domain = diff.attrs['domain']
        v = diff.value[0, :]
        name = diff.attrs['minuend'].split('/')[-1]
        plot_data += [Gnuplot.Data(domain,numpy.log2(v), title = name+", i = "+str(i))]
    g.plot(*plot_data)
    if sys.argv[2] is None and sys.argv[4] is None:
        raw_input("Press key to finish programme")
    print '...Done.-'



    
