import logging
import numpy as np

from .actions import Prototype
from ..io.simulation_data import dgTypes, systemD, DataGroup, sysDTypes

# A utility class to write to SimulationHDF via the
# actions.py framework.
class SimOutput(Prototype):
    """An action to handle output of timeslices to hdf.
    
    Each piece of data to be output is passed, as an object, to this class in 
    the array actionTypes.
    
    These objects are subclasses of SimOutputType and class which is accessible
    as an attribute of SimOutput.
    
    The existing subclasses of SimOutputType are:
    Data -- write tslice.fields to a datagroup
    Exact -- calls system.exactValue(tslice.time,tslice.domain) and outputs the 
             fields of the resulting tslice to a datagroup
    Times -- writes tslice.time to a datagroup
    TimeStep -- writes the time step associated to tslice to a datagroup. This 
                is currently implemented by calling system.timestep(u)
    Domains -- writes tslice.domain to a datagroup
    Constraints -- calls system.constraint_violation(u) and writes the output to
                   a datagroup
    DerivedData -- calls a user defined function, passing the tslice, as input
                   and stores the result to a datagroup
    System -- stores information relating to the system, solver and grid. Also
              stores the cmp parameter which is used to order the simulations.
              
    For information about how to access this data please see the simulation_data
    module.
    """

    def __init__(self, hdf_file, solver, theSystem, theInterval, 
            actionTypes, 
            frequency = 1, 
            cmp_ = None, 
            overwrite = True, 
            name = None, 
            start = -float('infinity'), 
            stop = float('infinity')):
        self.log = logging.getLogger("SimOutput")
#        if __debug__:
#            self.log.debug("Setting up HDF output...")
        super(SimOutput,self).__init__(frequency)
        if name == None:
            hour = str(time.localtime()[3])
            minute = str(time.localtime()[4])
            second = str(time.localtime()[5])
            name = hour+":"+minute+":"+second
        self.name = name
        self.hdf = hdf_file
        self.solver = solver
        self.system = theSystem
        self.grid = theInterval
        self.actions = actionTypes
        self.overwrite = overwrite
        self.cmp_ = cmp_
        for action in actionTypes:
            action.setup(self)
#        if __debug__:
#            self.log.debug("HDF output setup completed.")
        
    def _doit(self,it,u):
        for action in self.actions:
#            if __debug__:
#                self.log.debug("Outputting %s"%action.groupname)
            action(it,u)

    class SimOutputType(object):

        def __init__(self, derivedAttrs = None):
            if derivedAttrs is None:
                self.derivedAttrs = {}
            else:
                self.derivedAttrs = derivedAttrs
        
        def setup(self,parent):
            if parent.overwrite:
                self.data_group = DataGroup(parent.hdf.require_group(\
                    self.groupname\
                    ).require_group(parent.name))
            else:
                self.data_group = DataGroup(parent.hdf.create_group(\
                    self.groupname\
                    ).create_group(parent.name))
            self.parent = parent
            self.data_group.attrs['cmp'] = parent.cmp_
            self.log = logging.getLogger(self.groupname)
        
        def __call__(self,it,u):
            for key,value in self.derivedAttrs.items():
                v = value(it,u,self.parent.system)
#                if __debug__:
#                    self.log.debug("Derived Attrs = %s"%str(v))
                self.data_group[it].attrs[key] = v
         
    class Data(SimOutputType):

        groupname = dgTypes["raw"]

        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = u.fields
            super(SimOutput.Data,self).__call__(it,u)

    class Exact(SimOutputType):

        groupname = dgTypes["exact"]

        def __call__(self,it,u):
            dg = self.data_group
            parent = self.parent
            dg[it] = parent.system.exactValue(u.time,u.x).fields
            super(SimOutput.Exact, self).__call__(it, u)

    class Times(SimOutputType):
        
        groupname = dgTypes["time"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = np.array([u.time])
            super(SimOutput.Times,self).__call__(it,u)
         
    class TimeStep(SimOutputType):
        
        groupname = dgTypes["dt"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = np.array([self.parent.system.timestep(u)])
            super(SimOutput.TimeStep,self).__call__(it,u)   
      
    class Domains(SimOutputType):

        groupname = dgTypes["domain"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = np.asarray(u.grid.meshes)
            dg[it].attrs["shape"] = u.grid.shape
            dg[it].attrs["bounds"] = u.grid.bounds
            dg[it].attrs["comparison"] = u.grid.comparison
            super(SimOutput.Domains,self).__call__(it,u)

    class Constraints(SimOutputType):

        groupname = dgTypes["constraint"]
        
        def __call__(self,it,u):
            dg = self.data_group
            parent = self.parent
            dg[it] = parent.system.constraint_violation(u)
            super(SimOutput.Constraints,self).__call__(it,u)

    class DerivedData(SimOutputType):
        
        def __init__(self, name, function, derivedAttrs = None):
            self.func = function
            self.groupname = name
            super(SimOutput.DerivedData,self).__init__(derivedAttrs)
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = self.func(it,u)
            super(SimOutput.DerivedData,self).__call__(it,u)

    class System(SimOutputType):
        
        groupname = systemD
        
        def setup(self,parent):
            super(SimOutput.System,self).setup(parent)
            g = self.data_group.group
            psystem = np.asarray(repr(parent.system))
            pgrid = np.asarray(repr(parent.grid))
            psolver = np.asarray(repr(parent.solver))
            pcmp = np.asarray(repr(parent.cmp_))
            g.require_dataset(sysDTypes['system'],\
                psystem.shape, psystem.dtype, \
                data = psystem)
            g.require_dataset(sysDTypes['grid'], \
                pgrid.shape, pgrid.dtype, data=pgrid)
            g.require_dataset(sysDTypes['solver'], \
                psolver.shape, psolver.dtype, \
                data = psolver)
            g.require_dataset(sysDTypes['cmp'], \
                pcmp.shape, pcmp.dtype, data=pcmp)
        
        def __call__(self,it,u):
            pass
