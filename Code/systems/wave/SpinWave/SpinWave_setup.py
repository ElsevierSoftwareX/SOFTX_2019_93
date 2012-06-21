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
import TwoDAdvection_polar

################################################################################
# Argument Parsing and File Nameing
################################################################################
# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This script contains the settings for running SpinWave simulations.""")

# Parse files
parser.add_argument('-o','-output_file', help=\
"""The name and location of the hdf file to be produced.""")

#for file naming
month = str(time.localtime()[1])
day = str(time.localtime()[2])

# Default file settings
working_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
default_file_location = working_directory+"/"+"../../../Output/"
default_file_name = "SpinWave_%i-%s-%s.hdf"%\
    (MPI.COMM_WORLD.rank,month,day)

# Collect args and set up defaults
args = parser.parse_args()
if args.o is None:
    args.o = default_file_location+default_file_name
hdf_file_name = os.path.abspath(args.o)
log_file_name = os.path.abspath(os.path.splitext(args.o)[0]+".log")

################################################################################  
# These are the commonly altered settings
################################################################################

#output settings
store_output = False
display_output = True

# Set up logger
file_log_level = logging.DEBUG
if store_output and not display_output:
    logging.basicConfig(filename =log_file_name,\
        filemode='w',\
        level=file_log_level,\
        format = '%(filename)s:%(lineno)d - %(levelname)s:%(message)s')
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log.addHandler(console)
elif store_output and display_output:
    logging.basicConfig(filename =log_file_name,\
        filemode='w',format = '%(filename)s:%(lineno)d - %(levelname)s:%(message)s',\
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
Nr = 50
Ntheta = 50
Nphi = 50

# What grid to use?
rstart = 0
rstop = 2
phistart = 0
phistop = math.pi
thetastart = 0
thetastop = 2*math.pi

# Times to run between
tstart = 0.0
tstop = 3

# Configuration of System
CFLs = [0.5 for i in range(num_of_grids)]
tau = 0.5


# 1st derivative r
raxis_1D_diffop = sbp.D43_Strand()
#raxis_1D_diffop = fd.FD12()
#raxis_1D_diffop = fd.FD14()
#raxis_1D_diffop = sbp.D42(log)
#raxis_1D_diffop = fft.FFT_diff_scipy(1,rstop-rstart)
#raxis_1D_diffop = fft.FFT(1,xstop-xstart)
#raxis_1D_diffop = fft.RFFT(1)
#raxis_1D_diffop = fft.FFT_scipy(1,xstop-xstart)
#raxis_1D_diffop = fft.FFT_lagrange1(N,xstop-xstart)

# 1nd derivative phi
phiaxis_diffop = fft.FFT_diff_scipy(1,phistop-phistart)
#raxis_1D_diffop = fft.FFT(1,phistop-phistart)
#raxis_1D_diffop = fft.RFFT(1)
#raxis_1D_diffop = fft.FFT_scipy(1,phistop-phistart)

# Configuration of IBVP
solver = solvers.RungeKutta4()
maxIteration = 1000000

################################################################################
# Grid construction
################################################################################

# Grid point data      
raxis_gdp = [(Nr*2**i,Nphi*2**i) for i in range(num_of_grids)]

# Calcualte number of ghost points. I assume that number required on the right
# and left are the same.
try:
    ghp_1D = raxis_1D_diffop.ghost_points()
    ghp_2D = raxis_2D_diffop.ghost_points()
except:
    ghp_1D = [2,2]
    ghp_2D = [2,2]
l_ghp = max(ghp_1D[0],ghp_2D[0])
r_ghp = max(ghp_1D[1],ghp_2D[1])
if l_ghp != r_ghp:
    raise Exception("ghost_points miss match")
ghost_points = l_ghp    

# Build grids
#grids = [grid.Interval_2D_polar_mpi_PT(raxis_gdp[i], ([rstart,rstop],[phistart,phistop]),\
#    i) for i in range(num_of_grids)]
grids = [grid.Interval_2D(raxis_gdp[i], ([rstart,rstop],[phistart,phistop]),\
    i) for i in range(num_of_grids)]

################################################################################
# Print logging information
################################################################################
if __debug__:
    log.debug("HDF file = %s"%args.o)
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
    systems += [TwoDAdvection_polar.TwoDadvection(\
        -1,-1,\
        raxis_1D_diffop,\
        phiaxis_diffop,\
        CFL = CFLs[i],\
        log_parent=log,\
        equation_coords = 'Cartesian',\
        tau = tau,\
        )]
if __debug__:
    log.debug("Initialisation of systems complete.")

################################################################################
# Set up hdf file to store output
################################################################################
hdf_file = h5py.File(hdf_file_name)

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
        if display_output:
            actionList += [actions.GNUPlotter2D(\
                *gnu_plot_settings,frequency = 1,\
                system = system\
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
        problem = ibvp.IBVP(solver, system, grid = grids[i],\
                maxIteration = maxIteration, action = actionList)
        problem.run(tstart, tstop)
        log.info("Run %i complete"%i)
log.info("Simulation complete")
print args.o
