from __future__ import division

#import python libraries
import sys
import os
import logging
import h5py
import math
import argparse

#Import standard code base
from coffee import ibvp, actions, solvers, grid
from coffee import ibvp, actions, solvers, grid
from coffee.diffop import fd, fft, sbp
from coffee.diffop import fd, fft, sbp
from coffee.io import simulation_data
from coffee.io import simulation_data

#import system to use
import OneDwave

################################################################################
# Parser settings 
################################################################################
# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This program contains the necessary code for initialization of
the EvolutionSBP code.""")

# Parse files
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced. Defaults to test.""")
args = parser.parse_args()
################################################################################
# These are the commonly altered settings
################################################################################

#output settings
store_output = False
display_output = True
if store_output and args.f is None:
    print "OneDAdvection_setup.py: error: argument -f/-file is required"
    sys.exit(1)
    
# log file settings
if store_output:
    args.logf = os.path.splitext(args.f)[0]+".log"

# Set up logger
file_log_level = logging.INFO
if store_output and not display_output:
    logging.basicConfig(filename=args.logf,\
        filemode='w',\
        level=file_log_level,\
        format = '%(filename)s:%(lineno)d - %(levelname)s:%(message)s')
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log.addHandler(console)
elif store_output and display_output:
    logging.basicConfig(filename=args.logf,\
        filemode='w', \
        format = '%(filename)s:%(lineno)d - %(levelname)s:%(message)s',\
        level=file_log_level)
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log.addHandler(console)
else:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("main")
    

log.info("Starting configuration.")

# How many systems?
num_of_grids = 1

# How many grid points?
N = 200

# What grid to use?
xstart = 0
xstop = 4

# Times to run between
tstart = 0.0
tstop = 5

# Penalty boundary parameter
tau = 10

# Configuration of System
CFLs = [0.4 for i in range(num_of_grids)] 
#raxis_2D_diffop = fd.FD22()
raxis_2D_diffop = sbp.D43_2_CNG()
#raxis_2D_diffop = fft.RFFT(2,xstop-xstart)
#raxis_2D_diffop = fft.FFT_diff_scipy(2,xstop-xstart)
#raxis_2D_diffop = fft.FFT(2,xstop-xstart)
#raxis_2D_diffop = fft.FFT_scipy(2,xstop-xstart)
#raxis_2D_diffop = fft.FFTW(2,xstop-xstart)

# Configuration of IBVP
solver = solvers.RungeKutta4()
maxIteration = 1000000

################################################################################
# Grid construction
################################################################################

# Grid point data      
raxis_gdp = [N*2**i for i in range(num_of_grids)]

# Calcualte number of ghost points. I assume that number required on the right
# and left are the same.
#ghp = 2#raxis_2D_diffop.ghost_points()
#ghost_points = ghp    

# Build grids
grids = [grid.Interval_1D(raxis_gdp[i], [[xstart,xstop]],\
    comparison = raxis_gdp[i]) for i in range(num_of_grids)]

################################################################################
# Print logging information
################################################################################
if log.isEnabledFor(logging.DEBUG):
    log.debug("HDF file = %s"%args.f)
    log.debug("Start time = %f"%tstart)
    log.debug("Stop time = %f"%tstop)
    log.debug("CFLs are = %s"%repr(CFLs))
    log.debug("grids = %s"%repr(grids))

################################################################################
# Initialise systems
################################################################################
systems = []
if log.isEnabledFor(logging.DEBUG):
    log.debug("Initialising systems.")
for i in range(num_of_grids):
    systems += [OneDwave.OneDwave(\
        raxis_2D_diffop,\
        CFL = CFLs[i],\
        tau = tau,\
        log_parent=log\
        )]
if log.isEnabledFor(logging.DEBUG):
    log.debug("Initialisation of systems complete.")

################################################################################
# Set up hdf file to store output
################################################################################
if store_output:
    hdf_file = h5py.File(args.f)


################################################################################
# Set up action types for data storage in hdf file
################################################################################
output_actions = [\
    actions.SimOutput.Data(),\
    actions.SimOutput.Times(),\
    actions.SimOutput.System(),\
    actions.SimOutput.Domains()
    ]

################################################################################
# Set up gnu plot settings
################################################################################
gnu_plot_settings = [\
    'set yrange [-1:1]',\
    #'set xrange [0:1]',\
    #'set style data lines',\
    #'set zrange [-1:1]',\
    #'set title "2D wave" enhanced',\
    #'set mapping cylindrical'
    #'set xlabel "r"',\
    #'set key left',\
    #'set tics out',\
    'set style data lines'\
    ]

################################################################################
# Perform computation
################################################################################
log.info("Simulation configuration complete.")
for i,system in enumerate(systems):
        #Construct Actions
        actionList = []
        if display_output:
            actionList += [actions.GNUPlotter1D(\
                *gnu_plot_settings,frequency = 1,\
                system = system,\
                delay = 0.\
                )]
        if store_output:
            actionList += [actions.SimOutput(\
                hdf_file,\
                solver, \
                system, \
                grids[i], \
                output_actions,\
                overwrite = True,\
                name = grids[i].name,\
                cmp_ = grids[i].comparison\
                )]
        log.info("Starting simulation %i with system %s"%(i,repr(system)))
        problem = ibvp.IBVP(solver, system, grid = grids[i],\
                maxIteration = 1000000, action = actionList)
        problem.run(tstart, tstop)
        log.info("Simulation complete")
log.info("Calculations complete")
