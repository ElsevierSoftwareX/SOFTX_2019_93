#! /usr/bin/env python
import argparse
import subprocess

# Initialise parser
parser = argparse.ArgumentParser(description=\
"""This program runs a complete simulation: the numerical calculations, the error caculations and the visulations.""")

# Parse files
parser.add_argument('-f','-file', help=\
"""The name of the hdf file to be produced. If not given the file default from main.py will be used.""")

# Parse times
parser.add_argument('-terr','-times-error',nargs='+',help=\
"""A list of all times for which error calculations should be performed. This list will also be used for visualisation of error data.""")
parser.add_argument('-tani','-times-start-animation',nargs='+',help=\
"""A list of time ranges in the format "[t0,t1]" to be used for the start and stop times of animations.""")
parser.add_argument('-tplo','-times-plot',nargs='+', help=\
"""A list of all times for which plots of the specified data types should be made.""")

# Parse data types
parser.add_argument('-dg','-dgTypes',nargs='+',help=\
"""A list of the data group types for error calculation and visulisation. Currently only implemented for "raw".""")

# Collect args and set up defaults
args = parser.parse_args()
if args.dg is None:
    args.dg = ['raw']

if args.tani is not None:
    args.tani = [eval(tani) for tani in args.tani]

def bash_string(sarray,prefix):
    rs = ''
    for s in sarray:
        rs = rs + prefix+" '%s' "%s
    return rs.strip()

args.dg = bash_string(args.dg,'-dg')
args.terr_bash = bash_string(args.terr,'-t')

# Run main.py. Need to do this to get file name if it wasn't specified
main_run = "python -O ../Computation/main.py"
print "MAIN CALCULATION: "+main_run
#args.mfile =  subprocess.Popen(main_run,\
#    shell=True,stdout=subprocess.PIPE).communicate()[0]

if args.f is None:
    args.f = args.mfile

errorNum_run = []
errorNum_run += ["python ./errorNumerical %s %s %s"%\
        (args.terr_bash,args.dg,args.f)]
#for time in args.terr:
#    errorNum_run += ["python ./errorNumerical -t %s -dg %s %s"%\
#        (time,args.dg,args.f)]
        
hdfvis_error = []
for time in args.terr:
    hdfvis_error += ['python ./hdfvis %s err -t %s %s'%\
        (args.dg,time,args.f)]

hdfvis_plot = []
for time in args.tplo:
    hdfvis_plot  += ['python ./hdfvis %s plot -t %s %s'%\
        (args.dg,time,args.f)]
        
hdfvis_ani = []
for tani in args.tani:
        hdfvis_ani  += ['python ./hdfvis %s ani -t0 %s -t1 %s %s'%\
            (args.dg,tani[0],tani[1],args.f)]


for s in errorNum_run:
    print "ERROR CALCULATION: "+s
#    subprocess.call(s,shell=True)
for s in hdfvis_error: 
    print "ERROR VISULISATION: "+s    
#    subprocess.call(s,shell=True)
for s in hdfvis_plot: 
    print "GRAPHING: "+s    
#    subprocess.call(s,shell=True)
for s in hdfvis_ani: 
    print "ANIMATION: "+s    
#    subprocess.call(s,shell=True)
