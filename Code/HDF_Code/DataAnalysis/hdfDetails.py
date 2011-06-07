import sys
import h5py

sys.path.append("../../EvolutionSBP/")
import simulation_data as sd

#sys.argv[1] = 'exp_-20xx.hdf' - hdf data file

def exploreKeys(group):
    print group
    #print "group keys()"
    #print group.keys()
    for key in group.keys():
        #print group[key]
        try:
            if isinstance(group[key],h5py.highlevel.Dataset):
                return
            exploreKeys(group[key])
        except:
            pass
        
with sd.SimulationHDF(sys.argv[1]) as file:
    exploreKeys(file.file)
        
