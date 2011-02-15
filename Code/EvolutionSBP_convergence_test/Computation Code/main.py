import sys
import time

sys.path.append("../bin")
import system
import advection_smooth_initial_data
import advection_convergence_test
import one_d_wave_equation
import ibvp
import h5py_array as h5py
import diffop
import actions
import solvers

try:
    tstart = float(sys.argv[1])
    tstop = float(sys.argv[2])
except:
    tstart = 0.0
    tstop = 10.0

rintervals = [ibvp.Periodic_Grid((50, )), ibvp.Periodic_Grid((100,)),\
    ibvp.Periodic_Grid((200,)), ibvp.Periodic_Grid((400,))]
rk4 = solvers.RungeKutta4()
CFLs = [1.]
systems = []
for cfl in CFLs:
    systems += [one_d_wave_equation.wave_eqn(CFL = cfl)] #advection_eqn(D = D43_CNG(), CFL = 2., tau = 2.5)
year = str(time.localtime()[0])
month = str(time.localtime()[1])
day = str(time.localtime()[2])
hdf_file = h5py.H5pyArray("exp_-20xx_simple.hdf")
data = hdf_file.require_group("Raw Data")
gnu_plot_settings = ['set yrange [-1.2:1.2]', 'set style data linespoints',
    'set title "Advection equation" enhanced', 'set xlabel "r"']
print "Starting computation."
for system in systems:
    for rinterval in rintervals:
        problem = ibvp.IBVP(rk4, system, grid = rinterval, maxIteration = 100000,  action = (\
                #actions.Plotter(frequency = 1, xlim = (1,2), ylim = (-5,5),\
                #    findex = (0,1), delay = 0.0),\
                actions.GNUplotter(*gnu_plot_settings, frequency = 1, delay = 0.01), \
                actions.HDFOutput(data,rk4, system, rinterval,\
                    name = "Grid = "+rinterval.name,  cmp = rinterval.shape[0])))
        problem.run(tstart, tstop)
print "Done."
