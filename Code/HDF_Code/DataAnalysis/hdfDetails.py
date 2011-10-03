#! /usr/bin/env python

import sys
import h5py
import argparse

sys.path.append("../../EvolutionSBP/")
import simulation_data as sd

def _validate(group):
    """If group is an instance of sd.SimulationHDF then group.file is return. Otherwise group is returned. This method validates the arguments 
    of exploreKeys() and printHDF() to ensure that they are acting on HDf groups."""
    if isinstance(group,sd.SimulationHDF):
        return group.file
    return group

def exploreKeys(group):
    """Takes an HDF group and prints the group name followed by the group keys()"""
    group = _validate(group)
    print group
    print group.keys()

def printHDF(group):
    """Prints the first two levels of the passed HDF group. If the group is a simulation_data.py HDF file then the "System Data" 
    and "Numerical Error" groups are further explored.
    
group = hdf group to be printed"""
    group = _validate(group)
    print "==========================="
    exploreKeys(group)
    for key1 in group.keys():
        print "==========================="
        exploreKeys(group[key1])
        for key2 in group[key1].keys():
            if key1 == sd.dgTypes["errorNum"]:
                print "==========================="
                exploreKeys(group[key1][key2])
            elif key1 == sd.systemD:
                print "==========================="
                exploreKeys(group[key1][key2])
                for key3 in group[key1][key2]:
                    print "%s: %s"%(\
                        group[key1][key2][key3],\
                        group[key1][key2][key3].value\
                        )
            else:
                print group[key1][key2]
      
def loadHDF(file):
    """Return a simulationHDF() object representing the HDF file. The original HDF file can be accessed via <returned object>.file. 
    SimulationHDF() objects have a number of useful methods, including simplified access to individual runs via the .getSims() method.
    
file = string giving location of the hdf file."""
    return sd.SimulationHDF(file)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=\
    """This program outputs the keys and contents of the first two levels of an hdf file. If the file is produced from simulation_data.py 
    then the "Numerical Error" and "System Data" groups further explored. The numerical values of data groups are not returned. 
    It does not currently provide information on the attributes of the HDF groups.

This script is also designed to be loaded as a module. If done three functions, exploreKeys(), loadHDF() and printHDF() are made available.
Remember to appropriately close the hdf file, otherwise corruption can result.""")
    parser.add_argument('file',help =\
    """The hdf file whose contents is to be printed to stdout.""")

    args = parser.parse_args()

    with sd.SimulationHDF(args.file) as file:
        printHDF(file.file)


