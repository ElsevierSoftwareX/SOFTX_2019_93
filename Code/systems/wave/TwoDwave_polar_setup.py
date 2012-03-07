from __future__ import division

#import python libraries
import sys
import time
import numpy as np
import logging
import h5py
import math

#Import standard code base
from skyline import ibvp, actions, solvers, grid
from skyline.diffop import fd, fft, sbp
#from skyline.io import simulation_data

#import system to use
import TwoDwave_polar

#for file naming
year = str(time.localtime()[0])
month = str(time.localtime()[1])
day = str(time.localtime()[2])

################################################################################  
# These are the commonly altered settings
################################################################################

#file settings
#file_location = "/localhome/bwhale/"
file_location = "../../../Output/"
file_name = "no_mpi_test"

#output settings
store_output = True
display_output = True

# Set up logger
file_log_level = logging.INFO
if store_output and not display_output:
    logging.basicConfig(filename =\
        file_location+file_name+"-%s-%s-%s.log"%(year,month,day),\
        filemode='w',\
        level=file_log_level,\
        format = '%(filename)s:%(lineno)d - %(levelname)s:%(message)s')
    log = logging.getLogger("main")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log.addHandler(console)
elif store_output and display_output:
    logging.basicConfig(filename =\
        file_location+file_name+"-%s-%s-%s.log"%(year,month,day),\
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
Nphi = 50

# What grid to use?
rstart = 1
rstop = 3
phistart = 0
phistop = 2

# Times to run between
tstart = 0.0
tstop = 2.0

# Configuration of System
CFLs = [0.5 for i in range(num_of_grids)]


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


# 2nd derivative r
#raxis_2D_diffop = fd.FD22()
raxis_2D_diffop = sbp.D43_2_CNG()
#raxis_2D_diffop = fft.FFT_diff_scipy(2,rstop-rstart)
#raxis_2D_diffop = fft.FFT(2,rstop-rstart)
#raxis_2D_diffop = fft.RFFT(2)
#raxis_2D_diffop = fft.FFT_scipy(2,rstop-rstart)


# 2nd derivative phi
phiaxis_diffop = fft.FFT_diff_scipy(2,phistop-phistart)
#raxis_1D_diffop = fft.FFT(2,phistop-phistart)
#raxis_1D_diffop = fft.RFFT(2)
#raxis_1D_diffop = fft.FFT_scipy(2,phistop-phistart)

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
grids = [grid.Interval_2D_polar_mpi(raxis_gdp[i], ([rstart,rstop],[phistart,phistop]),\
    ghost_points, i) for i in range(num_of_grids)]

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
    systems += [TwoDwave_polar.TwoDwave(\
        raxis_1D_diffop,\
        raxis_2D_diffop,\
        phiaxis_diffop,\
        CFL = CFLs[i],\
        log_parent=log\
        )]
if log.isEnabledFor(logging.DEBUG):
    log.debug("Initialisation of systems complete.")

################################################################################
# Set up hdf file to store output
################################################################################
#hdf_file = h5py.H5pyArray(file_location + file_name +"-%s-%s-%s.hdf"\
#  %(year,month,day))
full_file_name = file_location + file_name +"-%s-%s-%s.hdf"%(year,month,day)
hdf_file = h5py.File(full_file_name)


################################################################################
# Set up action types for data storage in hdf file
################################################################################
output_actions = [\
    actions.SimOutput.Data(),\
    actions.SimOutput.Times()\
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
                maxIteration = 1000000, action = actionList)
        problem.run(tstart, tstop)
log.info("Simulation complete")
print full_file_name
