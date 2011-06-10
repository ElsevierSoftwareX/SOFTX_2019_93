import sys
import h5py

sys.path.append("../../EvolutionSBP/")
import simulation_data as sd

#sys.argv[1] = 'exp_-20xx.hdf' - hdf data file

def exploreKeys(group):
    print "==========================="
    print group
    print group.keys()
        
with sd.SimulationHDF(sys.argv[1]) as file:
    exploreKeys(file.file)
        
