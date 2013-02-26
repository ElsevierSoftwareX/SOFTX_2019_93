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
from coffee.actions import gp_plotter
from coffee.diffop import fd, fft
from coffee.diffop.sbp import sbp

#import system to use
import TwoDAdvection

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
parser.add_argument('-i','-info', default=False, action='store_true', 
    help="""A flag to indicate if information about progress of simulation is
required.""")
parser.add_argument('-tstop', type=float, default=1.0, help="""The time that the
        simulation will run to. Defaults to 1.0.""")
parser.add_argument('-tstart', type=float, default=0.0, help="""The initial time
        of the simulation. Defaults to 0.0.""")
parser.add_argument('-nsim', type=int, default=1, help="""The number of
simulations that you want to run. Defaults to 1.""")

# Actually parse the arguments
args = parser.parse_args()

###############################################################################
# File and logging settings
###############################################################################
#output settings
store_output = args.f is not None
display_output = args.d
    
# file settings
if store_output:
    args.f = os.path.abspath(args.f)
    try:
        if not os.path.exists(os.path.split(args.f)[0]):
            os.makedirs(os.path.split(args.f)[0])
    except OSError, oserror:
        if oserror.errno is not errno.EEXIST:
            raise oserror
    args.logf = os.path.splitext(args.f)[0]+"%d"%MPI.COMM_WORLD.rank+".log"

# Set up logger
file_log_level = logging.DEBUG
console_log_level = logging.INFO
if store_output:
    logging.basicConfig(
        filename=args.logf,
        filemode='w',
        level=file_log_level,
        format = '%(filename)s:%(lineno)d - %(levelname)s:%(message)s'
        )
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(level=console_log_level)
    log.addHandler(console)
else:
    logging.basicConfig(level=console_log_level)
    log = logging.getLogger("main")
log.info("Starting configuration.")

###############################################################################
# MPI set up
###############################################################################
log.info("Initialising mpi.cart_comm")
dims_list = [0,0]
dims = MPI.Compute_dims(MPI.COMM_WORLD.size, dims_list)
periods = [0,0]
reorder = True
mpi_comm = MPI.COMM_WORLD.Create_cart(dims, periods=periods, reorder=reorder)
log.info("Initialisation of mpi.cart_comm complete")

################################################################################
# System, IBVP and Grid settings
################################################################################
log.info("Starting configuration.")

# How many systems?
num_of_grids = args.nsim

# How many grid points?
Nx = 50
Ny = 50

# What grid to use?
xstart = 1
xstop = 3
ystart = 0
ystop = 2

# Times to run between
tstart = 0.0
tstop = 3

# Configuration of System
CFLs = [0.5 for i in range(num_of_grids)]
tau = 0.5


# 1st derivative x
xaxis_1D_diffop = sbp.D43_Strand()
#raxis_1D_diffop = fd.FD12()
#raxis_1D_diffop = fd.FD14()
#raxis_1D_diffop = sbp.D42(log)
#raxis_1D_diffop = fft.FFT_diff_scipy(1,rstop-rstart)
#raxis_1D_diffop = fft.FFT(1,xstop-xstart)
#raxis_1D_diffop = fft.RFFT(1)
#raxis_1D_diffop = fft.FFT_scipy(1,xstop-xstart)
#raxis_1D_diffop = fft.FFT_lagrange1(N,xstop-xstart)

# 1nd derivative y
yaxis_1D_diffop = sbp.D43_Strand()

# Configuration of IBVP
maxIteration = 1000000

################################################################################
# Grid construction
################################################################################

# Grid point data      
axes = [(Nx*2**i,Ny*2**i) for i in range(num_of_grids)]

# Build grids
grids = [
    grid.UniformCart(
        axes[i], 
    [(xstart,xstop), (ystart, ystop)],
    comparison = i,
    mpi_comm = mpi_comm
    ) for i in range(num_of_grids)]

################################################################################
# Print logging information
################################################################################
if __debug__:
    log.debug("HDF file = %s"%args.d)
    log.debug("Start time = %f"%tstart)
    log.debug("Stop time = %f"%tstop)
    log.debug("CFLs are = %s"%repr(CFLs))
    log.debug("grids = %s"%repr(grids))

################################################################################
# Initialise systems
################################################################################
systems = []
if __debug__:
    log.debug("Initialising systems.")
for i in range(num_of_grids):
    systems += [TwoDAdvection.TwoDadvection(\
        -1,-1,\
        xaxis_1D_diffop,\
        yaxis_1D_diffop,\
        CFL = CFLs[i],\
        log_parent=log,\
        equation_coords = 'Cartesian',\
        tau = tau,\
        )]
if __debug__:
    log.debug("Initialisation of systems complete.")

################################################################################
# Initialise Solvers
################################################################################
RKsolvers = [solvers.RungeKutta4(system) for system in systems]

################################################################################
# Set up hdf file to store output
################################################################################
if store_output and mpi_comm.Get_rank() == 0:
    hdf_file = h5py.File(args.f)

################################################################################
# Set up action types for data storage in hdf file
################################################################################
if store_output and mpi_comm.Get_rank() == 0:
    output_actions = [
        actions.SimOutput.Data(),
        actions.SimOutput.Times(),
        actions.SimOutput.System(),
        actions.SimOutput.Domains()
        ]

################################################################################
# Set up gnu plot settings
################################################################################
if display_output and mpi_comm.rank == 0:
    gnu_plot_settings = [\
        #'set yrange [-1:1]',\
        #'set xrange [0:1]',\
        #'set style data lines',\
        'set zrange [-1:1]',\
        'set title "2D wave" enhanced',\
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
log.info("Starting simulation.")
for i,system in enumerate(systems):
        log.info("Doing run %i"%i)
        #Construct Actions
        actionList = []
        if display_output and mpi_comm.rank == 0:
            actionList += [gp_plotter.Plotter2D(\
                *gnu_plot_settings,frequency = 1,\
                system = system\
                )]
        if store_output and mpi_comm.Get_rank() == 0:
            actionList += [actions.SimOutput(\
                hdf_file,\
                RKsolvers[i], \
                system, \
                grids[i], \
                output_actions,\
                overwrite = True,\
                name = grids[i].name,\
                cmp_ = grids[i].comparison\
                )]
        problem = ibvp.IBVP(RKsolvers[i], system, grid = grids[i],\
                maxIteration = maxIteration, action = actionList)
        problem.run(tstart, tstop)
        log.info("Run %i complete"%i)
log.info("Simulation complete")
