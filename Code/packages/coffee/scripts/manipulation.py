#! /usr/bin/env python

import sys
import h5py
import argparse

sys.path.append("../../EvolutionSBP/")
import simulation_data as sd

def strip_data(simHDF,outHDF,datapoints,component):
    sims = simHDF.getSims()
    group = outHDF.create_group(sims[-1].name)
    group.create_group(sd.dgTypes['raw'])
    group.create_group(sd.dgTypes['time'])
    group.create_group(sd.dgTypes['domain'])
    step = len(sims[-1].raw)/500
    indices = []
    for i in range(0,len(sims[-1].raw),step):
        group[sd.dgTypes['raw']].create_dataset(str(i),data=sims[-1].raw[i][component,:])
        group[sd.dgTypes['time']].create_dataset(str(i),data=sims[-1].time[i])
        group[sd.dgTypes['domain']].create_dataset(str(i),data=sims[-1].domain[i])
        indices += [str(i)]
    group.create_dataset("Index_Order",data = indices)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=\
    """This program outputs the keys and contents of the first two levels of an hdf file. If the file is produced from simulation_data.py 
    then the "Numerical Error" and "System Data" groups further explored. The numerical values of data groups are not returned. 
    It does not currently provide information on the attributes of the HDF groups.

This script is also designed to be loaded as a module. If done three functions, exploreKeys(), loadHDF() and printHDF() are made available.
Remember to appropriately close the hdf file, otherwise corruption can result.""")
    parser.add_argument('file',help =\
    """The hdf file whose contents is to be printed to stdout.""")

    parser.add_argument('-o',"-output",help =\
    """The hdf file whose contents is the needed data.""")

    args = parser.parse_args()

    with sd.SimulationHDF(args.file) as file:
        with h5py.File(args.o) as output:
            strip_data(file,output,500,0)


