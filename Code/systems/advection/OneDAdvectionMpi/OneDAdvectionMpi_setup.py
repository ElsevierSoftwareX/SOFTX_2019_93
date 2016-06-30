from __future__ import division

#import python libraries
import sys
import os
import numpy as np
import logging
import h5py
import math
import argparse
from mpi4py import MPI

#Import standard code base
from coffee import ibvp, solvers, grid, actions
from coffee.diffop import fd, fft
from coffee.diffop.sbp import sbp
from coffee.actions import gp_plotter

#import system to use
import OneDAdvectionMpi

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
parser.add_argument('-d','-display', default=False, 
    action='store_true', help=\
"""A flag to indicate if visual display is required.""")
parser.add_argument('-s', '-speed', default=-1, type=float, help="""The
characteristic speed of the system, defaults to -1 (i.e. the wave travels to
the right at a speed of 1.""")
parser.add_argument('-npoints', default=100, type=int, help="""The number of
grid points in the total simulation, defaults to 100.""")
args = parser.parse_args()
################################################################################
# These are the commonly altered settings
################################################################################

#output settings
store_output = args.f is not None
display_output = args.d
    
# file settings
if store_output:
    args.f = os.path.abspath(args.f)
    if not os.path.exists(os.path.split(args.f)[0]):
        os.makedirs(os.path.split(args.f)[0])
    args.logf = os.path.splitext(args.f)[0]+"%d"%MPI.COMM_WORLD.rank+".log"

# Set up logger
file_log_level = logging.DEBUG
file_logging_format = '%(filename)s:%(lineno)d - %(levelname)s: %(message)s'
console_logging_format = repr(MPI.COMM_WORLD.rank) + ":" + logging_format
if store_output and not display_output:
    logging.basicConfig(
        filename=args.logf,
        filemode='w',
        level=file_log_level,
        format=logging_format,
    )
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(console_logging_format))
    log.addHandler(console)
elif store_output and display_output:
    logging.basicConfig(
        filename=args.logf,
        filemode='w', 
        format=logging_format,
        level=file_log_level
    )
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(console_logging_format))
    log.addHandler(console)
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format=console_logging_format
    )
    log = logging.getLogger("main")

log.info("Starting configuration.")

# How many systems?
num_of_grids = 1

# How many grid points?
N = args.npoints

# What grid to use?
xstart = 0
xstop = 2

# Times to run between
tstart = 0.0
tstop = 0.10

# speed of system
speed = args.s

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

# Configuration of IBVP
maxIteration = 1000000

###############################################################################
# MPI set up
###############################################################################
log.info("Initialising mpi.cart_comm")
dims = MPI.Compute_dims(MPI.COMM_WORLD.size, [0]) 
periods = [0]
reorder = True
mpi_comm = MPI.COMM_WORLD.Create_cart(dims, periods=periods, reorder=reorder)

log.info("Initialisation complete")

################################################################################
# Grid construction
################################################################################

# Grid point data      
raxis_gdp = [N*2**i for i in range(num_of_grids)]

# Calcualte number of ghost points. I assume that number required on the right
# and left are the same.
ghost_points = (raxis_1D_diffop.ghost_points(),)

# Build grids
grids = [
    grid.UniformCart(
        (raxis_gdp[i],), 
        [[xstart, xstop]], 
        comparison=raxis_gdp[i],
        mpi_comm=mpi_comm,
        ghost_points=ghost_points
        ) 
    for i in range(num_of_grids)
    ]  

################################################################################
# Initialise systems
################################################################################
systems = []
if __debug__:
    log.debug("Initialising systems.")
for i in range(num_of_grids):
    systems += [OneDAdvectionMpi.OneDAdvectionMpi(\
        raxis_1D_diffop, speed,
        CFL = CFLs[i], tau = tau
        )]
if __debug__:
    log.debug("Initialisation of systems complete.")

###############################################################################
# Construction of solver
###############################################################################
solvers = [solvers.RungeKutta4(sys) for sys in systems]

################################################################################
# Set up hdf file to store output
################################################################################
if store_output and mpi_comm.Get_rank() == 0:
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
# Print logging information
################################################################################
if __debug__:
    if store_output:
        log.debug("HDF file = %s"%args.f)
        log.debug("Logging file = %s"%args.logf)
    log.debug("Start time = %f"%tstart)
    log.debug("Stop time = %f"%tstop)
    log.debug("CFLs are = %s"%repr(CFLs))
    log.debug("grids = %s"%repr(grids))
    log.debug("speed = %s"%repr(speed))

################################################################################
# Perform computation
################################################################################
log.info("Simulation configuration complete.")
for i,system in enumerate(systems):
        #Construct Actions
        actionList = []
        if display_output and mpi_comm.rank == 0:
            actionList += [gp_plotter.Plotter1D(
                system,
                *gnu_plot_settings,frequency = 1,
                delay = 0.
                )]
        if store_output and mpi_comm.rank == 0:
            actionList += [actions.SimOutput(\
                hdf_file,
                solvers[i],
                system,
                grids[i],
                output_actions,
                overwrite = True,
                name = grids[i].name,
                cmp_ = grids[i].comparison
                )]
        log.info("Starting simulation %i with system %s"%(i,repr(system)))
        problem = ibvp.IBVP(solvers[i], system, grid = grids[i],
                maxIteration = 1000000, action = actionList)
        problem.run(tstart, tstop)
        log.info("Simulation complete")
log.info("Calculations complete")
