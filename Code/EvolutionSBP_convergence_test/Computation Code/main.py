import sys
sys.path.append("../bin")
import advection_convergence_test
import ibvp
import h5py_array as h5py
import time
import system
import diffop
import actions
import solvers

try:
    tstart = float(sys.argv[1])
    tstop = float(sys.argv[2])
except:
    tstart = 0.0
    tstop = 10.0

rintervals = [ibvp.Grid((201,)), ibvp.Grid((401,)), ibvp.Grid((801,)), ibvp.Grid((1601,))]
rk4 = solvers.RungeKutta4()
CFLs = [2]
systems = [advection_convergence_test.advection_eqn(D = diffop.D43_CNG(),\
        CFL = 2, tau = 2.5)]
year = str(time.localtime()[0])
month = str(time.localtime()[1])
day = str(time.localtime()[2])
hdf_file = h5py.H5pyArray("advection_"+day+"-"+month+"-"+year+".hdf")
gnu_plot_settings = ['set yrange [-1.2:1.2]', 'set style data linespoints',
    'set title "Advection equation" enhanced', 'set xlabel "r"']
print "Starting computation."
for system in systems:
    for rinterval in rintervals:
        problem = ibvp.IBVP(rk4, system, rinterval, action = (\
                #actions.Plotter(frequency = 1, xlim = (1,2), ylim = (-5,5),\
                #    findex = (0,1), delay = 0.0),\
                actions.GNUplotter(*gnu_plot_settings, frequency = 1, delay = 0.01), \
                actions.HDFOutput(hdf_file,rk4, system, rinterval, name = "Grid = "+rinterval.name)))
        problem.run(tstart, tstop)
print "Done."
