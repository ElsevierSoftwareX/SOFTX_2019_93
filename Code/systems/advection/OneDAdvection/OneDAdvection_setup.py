from __future__ import division

#import python libraries
import sys
import os
import numpy as np
import logging
import h5py
import math
import argparse

#Import standard code base
from coffee import ibvp, actions, solvers, grid
from coffee.actions import gp_plotter
from coffee.diffop import fd, fft
from coffee.diffop.sbp import sbp

#import system to use
import OneDAdvection

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
N = 100

# What grid to use?
xstart = 0
xstop = 2

# Times to run between
tstart = 0.0
tstop = 10.0

# Configuration of System
CFLs = [0.5 for i in range(num_of_grids)]
tau = 1

# Select diffop
#raxis_1D_diffop = sbp.D43_Strand()
#raxis_1D_diffop = fd.FD12()
#raxis_1D_diffop = fd.FD14()
raxis_1D_diffop = sbp.D42()
#raxis_1D_diffop = sbp.D43_Tiglioetal()
#raxis_1D_diffop = sbp.D43_CNG()
#raxis_1D_diffop = fft.FFT_diff_scipy(1,xstop-xstart)
#raxis_1D_diffop = fft.FFTW(1,xstop-xstart)
#raxis_1D_diffop = fft.FFTW_real(1,xstop-xstart)
#raxis_1D_diffop = fft.FFTW_convolve(1,xstop-xstart)
#raxis_1D_diffop = fft.FFT(1,xstop-xstart)
#raxis_1D_diffop = fft.RFFT(1)
#raxis_1D_diffop = fft.FFT_scipy(1,xstop-xstart)
#raxis_1D_diffop = fft.RFFT_scipy(1)
#raxis_1D_diffop = fft.FFT_lagrange1(N,xstop-xstart)


################################################################################
# Grid construction
################################################################################

# Grid point data      
raxis_gdp = [N*2**i for i in range(num_of_grids)]

# Calcualte number of ghost points. I assume that number required on the right
# and left are the same.
ghp = 2 #raxis_1D_diffop.ghost_points()
ghost_points = ghp    

# Build grids
grids = [grid.UniformCart((raxis_gdp[i],), [(xstart, xstop)],\
    comparison = raxis_gdp[i]) for i in range(num_of_grids)]

################################################################################
# Print logging information
################################################################################
if log.isEnabledFor(logging.DEBUG):
    log.debug("HDF file location = %s"%file_location)
    log.debug("HDF file root name = %s"%file_name)
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
    systems += [OneDAdvection.OneDAdvection(\
        raxis_1D_diffop,
        CFL = CFLs[i], tau = tau
        )]
if log.isEnabledFor(logging.DEBUG):
    log.debug("Initialisation of systems complete.")

# Configuration of IBVP
solvers = [solvers.RungeKutta4(system) for system in systems]
maxIteration = 1000000

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
    actions.SimOutput.Domains()\
    ]

################################################################################
# Set up gnu plot settings
################################################################################
gnu_plot_settings = [\
    #'set yrange [-30:0]',\
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
for i, system in enumerate(systems):
        #Construct Actions
        actionList = []
        if display_output:
            actionList += [gp_plotter.Plotter1D(
                system,
                *gnu_plot_settings,
                frequency = 1,
                delay = 0.
                )]
        if store_output:
            actionList += [actions.SimOutput(\
                hdf_file,\
                solvers[i], \
                system, \
                grids[i], \
                output_actions,\
                overwrite = True,\
                name = grids[i].name,\
                cmp_ = grids[i].comparison\
                )]
        log.info("Starting simulation %i with system %s"%(i,repr(system)))
        problem = ibvp.IBVP(solvers[i], system, grid = grids[i],\
                maxIteration = 1000000, action = actionList)
        problem.run(tstart, tstop)
        log.info("Simulation complete")
log.info("Calculations complete")
