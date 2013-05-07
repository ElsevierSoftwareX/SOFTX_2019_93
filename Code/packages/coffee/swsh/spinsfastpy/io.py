#Standard Library imports
import numpy as np

#Coffee imports
from coffee.actions.hdf_output import SimOutput
from coffee.io import simulation_data
from coffee.swsh.spinsfastpy import salm

simulation_data.dgTypes["sralm"] = "sralm"

class Sralm_Out(SimOutput.SimOutputType):

    groupname = simulation_data.dgTypes["sralm"]

    def __call__(self, it, u):
        #import pdb;pdb.set_trace()
        dg = self.data_group
        _write_salm(dg, it, u.data)

        #dg[it] = np.array([
            #np.asarray(d) for d in u.data
            #], dtype = 'complex')

        #dg[it].attrs["spins"]= np.array([
            #d.spins for d in u.data
            #], dtype = 'int')

        #lmax = u.data[0].lmax
        #for i in range(1, u.data.shape[0]):
            #if lmax is not u.data[0].lmax:
                #raise ValueError("Unable to store salm objects with different\
                #lmax in the same datagroup.")
        #dg[it].attrs["lmax"] = lmax


        #cg = u.data[0].cg
        #for i in range(1, u.data.shape[0]):
            #if cg is not u.data[0].cg:
                #raise ValueError("Unable to store salm objects with different\
                #cg in the same datagroup.")
        #dg[it].attrs["cg_module"] = cg.__module__ 
        #dg[it].attrs["cg_class"] = cg.__class__.__name__


        #bl_mult = u.data[0].bl_mult
        #for i in range(1, u.data.shape[0]):
            #if bl_mult is not u.data[0].bl_mult:
                #raise ValueError("Unable to store salm objects with different\
                #multiplication bandlimits in the same datagroup.")
        #dg[it].attrs["bl_mult"] = bl_mult
        super(Sralm_Out, self).__call__(it, u)

class SralmDataGroup(simulation_data.DataGroup):

    def __init__(self, *args, **kwds):
        super(SralmDataGroup, self).__init__(*args, **kwds)

    def __setitem__(self, i, sralm):
        #import pdb; pdb.set_trace()
        if len(sralm.shape) == 1:

            #value = np.array([
                #np.asarry(d) for d in u.data
                #], dytpe='complex')
            
            #dataset = self.group.require_dataset(
                #str(i), 
                #value.shape,  
                #value.dtype
                #)

            _write_salm(self.group, i, sralm)

    def __getitem__(self, i):
        if self.rV:
            cg_mod = __import__(
                self.group[str(i)].attrs["cg_module"],
                fromlist=[self.group[str(i)].attrs["cg_class"]]
                )
            cg_class = getattr(
                cg_mod,
                self.group[str(i)].attrs["cg_class"]
                )
            return salm.sfpy_sralm(
                self.group[str(i)].value,
                self.group[str(i)].attrs["spins"],
                int(self.group[str(i)].attrs["lmax"]),
                cg = cg_class(),
                bandlimit_multiplication = self.group[str(i)].attrs["bl_mult"]
                )
        return self.group[str(i)]
        
    def __repr__(self):
        return r"<SralmDataGroup %s (%d)>"% (self.name, len(self))       

def _write_salm(datagroup, index, salm):

    datagroup[str(index)] = np.array([
        np.asarray(d) for d in salm
        ], dtype = 'complex')

    datagroup[str(index)].attrs["spins"]= np.array([
        d.spins for d in salm
        ], dtype = 'int')

    lmax = salm[0].lmax
    for i in range(1, salm.shape[0]):
        if lmax is not salm[0].lmax:
            raise ValueError("Unable to store salm objects with different\
            lmax in the same datagroup.")
    datagroup[str(index)].attrs["lmax"] = lmax

    cg = salm[0].cg
    for i in range(1, salm.shape[0]):
        if cg is not salm[0].cg:
            raise ValueError("Unable to store salm objects with different\
            cg in the same datagroup.")
    datagroup[str(index)].attrs["cg_module"] = cg.__module__ 
    datagroup[str(index)].attrs["cg_class"] = cg.__class__.__name__

    bl_mult = salm[0].bl_mult
    for i in range(1, salm.shape[0]):
        if bl_mult is not salm[0].bl_mult:
            raise ValueError("Unable to store salm objects with different\
            multiplication bandlimits in the same datagroup.")
    datagroup[str(index)].attrs["bl_mult"] = bl_mult

simulation_data.dgTypes_DataGroups["sralm"] = (
  "coffee.swsh.spinsfastpy.io", "SralmDataGroup"
  )
