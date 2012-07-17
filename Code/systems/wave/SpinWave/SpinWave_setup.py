from __future__ import division

#import python libraries
import os
import sys
import time
import numpy as np
import logging
import h5py
import math
from mpi4py import MPI
import argparse

#Import standard code base
from coffee import ibvp, actions, solvers, grid
from coffee.diffop import fd, fft, sbp

#import system to use
import SpinWave

################################################################################
# Argument Parsing and File Nameing
################################################################################
# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This script contains the settings for running SpinWave simulations.""")

# Parse files
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced.""")
parser.add_argument('-d','-display', default=False, 
    action='store_true', help=\
"""A flag to indicate if visual display is required.""")
parser.add_argument('-i','-info', default=False, 
    action='store_true', help=\
"""A flag to indicate if information about progress of simulation is required.""")
args = parser.parse_args()
################################################################################  
# These are the commonly altered settings
################################################################################

#output settings
store_output = args.f is not None
display_output = args.d
    
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
num_of_grids = 3

# How many grid points?
Nr = 50
Ntheta = 24
Nphi = 12

# What grid to use?
rstart = 1
rstop = 3
phistart = 0
phistop = math.pi
thetastart = 0
thetastop = 2*math.pi

# Times to run between
tstart = 0.0
tstop = 3

# Configuration of System
CFLs = [0.5 for i in range(num_of_grids)]
tau = 1.0
iv_routine = "exp_bump"
lmax = 5

# 1st derivative r
diffop = sbp.D43_Strand()

# Configuration of IBVP
solver = solvers.RungeKutta4()
maxIteration = 1000000

################################################################################
# Grid construction
################################################################################
# Grid point data      
r_gdp = [Nr*2**i for i in range(num_of_grids)]
theta_gdp = [Ntheta*2**i for i in range(num_of_grids)]
phi_gdp = [Nphi*2**i for i in range(num_of_grids)]

# Build grids
grids = [grid.UniformCart(
    (r_gdp[i], theta_gdp[i], phi_gdp[i]),
    [[rstart, rstop], [thetastart, thetastop], [phistart, phistop]],
    comparison = r_gdp[i] * theta_gdp[i] * phi_gdp[i]
    ) for i in range(num_of_grids)]

################################################################################
# Print logging information
################################################################################
if __debug__:
    log.debug("HDF file = %s"%args.f)
    log.debug("Start time = %f"%tstart)
    log.debug("Stop time = %f"%tstop)
    log.debug("CFLs are = %s"%repr(CFLs))

################################################################################
# Initialise systems
################################################################################
systems = []
if __debug__:
    log.debug("Initialising systems.")
for i in range(num_of_grids):
    systems += [SpinWave.SpinWave(
        diffop,
        CFL = CFLs[i],
        tau = tau,
        iv_routine=iv_routine,
        lmax=lmax
        )]
if __debug__:
    log.debug("Initialisation of systems complete.")

################################################################################
# Set up hdf file to store output
################################################################################
if store_output:
    hdf_file = h5py.File(args.f)


################################################################################
# Set up action types for data storage in hdf file
################################################################################
output_actions = [
    actions.SimOutput.Data(),
    actions.SimOutput.Times(),
    actions.SimOutput.System(),
    actions.SimOutput.Domains()
    ]

################################################################################
# Set up gnu plot settings
################################################################################
gnu_plot_settings = [\
    #'set yrange [-30:0]',\
    'set yrange [-2:2]',\
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
    if args.i:
        actionList += [actions.Info()]
    if display_output:
        actionList += [actions.GNUPlotter2D(
            *gnu_plot_settings,
            frequency = 1,
            system = system,
            delay = 0.
            )]
    if store_output:
        actionList += [actions.SimOutput(
            hdf_file,
            solver, 
            system,
            grids[i], 
            output_actions,
            overwrite = True,
            name = grids[i].name,
            cmp_ = grids[i].comparison
            )]
    log.info("Starting simulation %i with system %s"%(i,repr(system)))
    problem = ibvp.IBVP(solver, system, grid = grids[i],\
            maxIteration = 1000000, action = actionList)
    problem.run(tstart, tstop)
    log.info("Simulation complete")
log.info("Calculations complete")
