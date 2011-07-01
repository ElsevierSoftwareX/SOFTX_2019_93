#! /usr/bin/env python
import argparse
import subprocess

# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This program runs a complete simulation: the numerical calculations, the error caculations and the visulations.""")

# Parse files
#parser.add_argument('file', help=\
#"""The name of the hdf file to be produced.""")

# Parse times
parser.add_argument('-terr','-times-error',nargs='+',help=\
"""A list of all times for which error calculations should be performed. This list will also be used for visualisation of error data.""")
parser.add_argument('-tani','-times-start-animation',nargs='+',help=\
"""A list of time ranges in the format "[t0,t1]" to be used for the start and stop times of animations.""")
parser.add_argument('-tplo','-times-plot',nargs='+', help=\
"""A list of all times for which plots of the specified data types should be made.""")

# Parse data types
parser.add_argument('-dg','-dgTypes',nargs='+',default='raw',help=\
"""A list of the data group types for error calculation and visulisation. Currently only implemented for "raw".""")

args = parser.parse_args()
#print args

#args.dg = ['raw']
if args.tani is not None:
    args.tani = [eval(tani) for tani in args.tani]

def bash_string(sarray):
    rs = ''
    for s in sarray:
        rs = rs + "'%s' "%s
    return rs.strip()

args.dg = bash_string(args.dg)
args.terr_bash = bash_string(args.terr)

main_run = "python -O ../Computation/main.py"
print "MAIN CALCULATION: "+main_run
args.file =  subprocess.Popen(main_run,\
    shell=True,stdout=subprocess.PIPE).communicate()[0]

errorNum_run = []
for time in args.terr:
    errorNum_run += ["./errorNumerical -t %s -dg %s %s"%\
        (time,args.dg,args.file)]
        
hdfvis_error = []
hdfvis_plot = []
for time in args.terr:
    hdfvis_error += ['./hdfvis -dg %s err -t %s %s'%\
        (args.dg,time,args.file)]
    hdfvis_plot  += ['./hdfvis -dg %s plot -t %s %s'%\
        (args.dg,time,args.file)]
        
hdfvis_ani = []
for tani in args.tani:
        hdfvis_ani  += ['./hdfvis -dg %s ani -t0 %s -t1 %s %s'%\
            (args.dg,tani[0],tani[1],args.file)]


for s in errorNum_run:
    print "ERROR CALCULATION: "+s
    subprocess.call(s,shell=True)
for s in hdfvis_error: 
    print "ERROR VISULISATION: "+s    
    subprocess.call(s,shell=True)
for s in hdfvis_plot: 
    print "GRAPHING: "+s    
    subprocess.call(s,shell=True)
for s in hdfvis_ani: 
    print "ANIMATION: "+s    
    subprocess.call(s,shell=True)
