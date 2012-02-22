import functools
import h5py
import cPickle
import numpy as np
import Gnuplot
import time
import sys
import logging 
import math 

sys.path.append("../../EvolutionSBP/")
import system
import ibvp
import diffop
import actions
import solvers
import simulation_data

# Do not change the keys!
#
# Each dgTypes (DataGroup Types) describes the group name for a data group
# structure in the hdf file. Each data group stores hdf datasets named by
# an index. The index across different dgtypes gives data for the corresponding
# iteration.
#
# The keys represent the names of the different dgtypes and are hard coded
# throughout the code. Therefore if you wish to rename a dgtype, just change
# the item, not the key.
#
# These represent the types of data that a simulation knows about.
dgTypes = {\
    "raw":"Raw_Data",\
    "exact":"Exact_Data",\
    "errorNum":"Error_Numeric",\
    "errorExa":"Error_Exact",\
    "IJ":"Weyl_Constants_IJ",\
    "domain":"Domain",\
    "time":"Time",\
    "dt": "Time_Step",\
    "scrif":"Scri+",\
    "constraint": "Constraint",\
    "mu":"mu",\
    "mup":"mup"\
    }

dgTypesInv = dict(zip(dgTypes.values(),dgTypes.keys()))

# SystemDataTypes stores a list of all the subgroups in the system groups.

systemD = "System"

sysDTypes = {\
    "system":systemD,\
    "solver": "Solver",\
    "grid": "Grid",\
    "cmp": "cmp"\
    }



# An interface to ease interaction with the simulationHDF class when
# only a specific simulation is wanted. I expect this class to be used the
# most.
@functools.total_ordering
class Sim(object):
    
    def __init__(self,simName,simHDF):
        self.simHDF = simHDF
        self.name = simName
        existing_items = self.simHDF[systemD+"/"+self.name].keys()
        for key, item in sysDTypes.items():
            if item in existing_items:
                setattr(self,key,\
                    self.simHDF[systemD+"/"+self.name][item].value\
                    )
        self.cmp = float(self.cmp)
        existing_items = self.simHDF.file.keys()
        for key, item in dgTypes.items():
            if item in existing_items:
                if self.name in self.simHDF[item].keys():
                    setattr(self,key,\
                      DataGroup(self.simHDF[item+"/"+self.name],\
                        returnValue=True\
                        )\
                      )

    
    def tslice(self,i):
        return self.simHDF.tslice(i,self.name,dgType = dgType["raw"])
    
    def indexOfTime(self,t):
        return self.simHDF.indexOfTime(t,self.name)
        
    def indexOfDomain(self,d,time_index):
        return self.simHDF.indexOfDomain(d,time_index,self.name)
    
    def __eq__(self,other):
        return self.cmp == other.cmp
        
    def __lt__(self,other):
        return self.cmp < other.cmp
    
    def __str__(self):
        return self.name
    
    def write(self,dgType,it,data,name = None,derivedAttrs = None):
        self.simHDF.write(dgType,self.name,it,data,name,derivedAttrs)
    
    def getDgType(self,dgType):
        return self.simHDF.getDgType(dgType,self.name)
        
    def getDgTypeAttr(self,dgType,attr,i):
        return self.simHDF.getDgTypeAttr(dgType,attr,i,self.name)    
    
    def animate(self,dgType="raw",gnuCommands=None,\
        gnuInitialisationCommands = {'debug':0,'persist':1},\
        tstart = -float('Infinity'),tstop = float('Infinity'),\
        animationLength = 2,\
        framesPerSec = 60\
        ):
        self.GNUplot(self.getDgType(dgType),\
            gnuCommands=gnuCommands,\
            gnuInitialisationCommands=gnuInitialisationCommands,\
            tstart = tstart, tstop = tstop,\
            animationLength = animationLength,\
            framesPerSec = framesPerSec
            )
      
    def plot(self,time,dgType="raw",gnuCommands=None,\
        gnuInitialisationCommands = {'debug':0,'persist':1}\
        ):
        self.animate(dgType=dgType,gnuCommands=gnuCommands,\
        gnuInitialisationCommands=gnuInitialisationCommands,\
        tstart=time,tstop=time\
        )
    
    def GNUplot(self,group,  gnuCommands=None,\
        gnuInitialisationCommands=None,\
        tstart = -float('Infinity'),tstop = float('Infinity'),
        animationLength = 2,\
        framesPerSec = 60,\
        ):
        """A utility function which plots a given group.
        """
        if gnuCommands is None:
            gnuCommands = []
        if gnuInitialisationCommands is None:
            gnuInitialisationCommands = []
        
        # Get x values
        domains = self.getDgType('domain')
        
        # Get all times
        times = self.getDgType('time')
        times.rV = True
        numOfFrames = len(times)
        frameSkip = int(numOfFrames/(animationLength*framesPerSec))

        # Get data for scri            
        scrif = self.getDgType('scrif')
        scrif.rV = True

        #Initialize gnuplot
        gnu = Gnuplot.Gnuplot(**gnuInitialisationCommands)
        gnu.reset()
        for command in gnuCommands:
          gnu(command)
        
        # Iterate across group
        # Get starting and stoping index
        nextFrame_index = self.indexOfTime(tstart)
        stop_index = self.indexOfTime(tstop)
        
        # While there is a next frame...
        while nextFrame_index<numOfFrames:
            # plot data
            i = nextFrame_index
            y = group[i]
            gnu.title('Simulation %s at time %f'%(self.name,times[i])) 
            plotItems = []
            for j,row in enumerate(np.atleast_2d(y.value)):
                plotItems +=[Gnuplot.Data(domains[i],\
                    row, \
                    title = 'Component '+str(j))]
            plotItems += [Gnuplot.Data(domains[i],scrif[i],\
                title = 'Scri+')]
            gnu.plot(*plotItems)
            # if there are not enough frames left set the frameSkip to 0
            if nextFrame_index+frameSkip >= numOfFrames:
                frameSkip = 0
            nextFrame_index += 1+frameSkip
            # if the next frame is larger than the stop_index then stop
            # plotting
            if nextFrame_index > stop_index:
                break
        gnu.close()


    class dsReturnValue(object):
    
        def __init__(self,dataset):
            self.ds = dataset
            
        def __getitem__(self,key):
            return self.ds[key].value

# Allows for interaction with the hdf file without specific reference
# to a particular simulation. I expect that this class will only be used
# for easy access to the sim objects. 
class SimulationHDF(object):

    def __init__(self,fileName,**kwds):
        self.file = h5py.File(fileName)
    
    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
        self.file.close()
    
    def sim(self,name):
        return Sim(name,self)
    
    def name(self):
      return self.file.name
    
    def getSims(self):
        simArray = [self.sim(name) for name in \
            self.file[systemD].keys()]
        return sorted(simArray)
    
    def __getitem__(self,key):
        return self.file[key]
    
    def getSimData(self,sim):
        sg = self.file[dgTypes["system"]+sim]
        rl = {"name":sim}
        for key,item in systemData:
            rl[key] = sg[item].value
        return rl
            
    def tslice(self,i,sim,dgType="raw"):
        rdata = self.file[dgTypes[dgtype]+sim][str(i)]
        time = self.file[dgTypes["times"]+sim][str(i)]
        domain = self.file[dgTypes["domains"]+sim][str(i)]
        return tslice(data,domain,time)
        
    def getDgType(self,dgType,sim):
        return DataGroup(self.file[dgTypes[dgType]+"/"+sim])   
    
    def getDgTypeAttr(dgType,attr,i,sim):
        return DataGroup(self.file[dgTypes[dgType]+"/"+sim]).attr[attr]
    
    #def indexOfTime(self,t,sim):
    #    times_dg = DataGroup(self.file[dgTypes["time"]+"/"+sim])
    #    times_dg.rV = True
    #    uIndex = simulation_data.binarysearch(times_dg,0,len(times_dg)-1,t)
    #    uTime = times_dg[uIndex]
    #    lIndex = uIndex-1
    #    lTime = times_dg[lIndex]
    #    dts = DataGroup(self.file[dgTypes["dt"]+"/"+sim])
    #    dt = dts[lIndex].value
    #    if t<lTime+dt/2:
    #        return lIndex
    #    else:
    #        return uIndex
    
    def indexOfTime(self,t,sim):
        times_dg = DataGroup(self.file[dgTypes["time"]+"/"+sim])
        min_time = times_dg[0][0]
        max_time = times_dg[len(times_dg)-1][0]
        if t <= min_time:
            return 0
        elif t>= max_time:
            return len(times_dg)-1
        else:
            for i,time_dg in enumerate(times_dg):
                dt = times_dg[i+1].value-time_dg.value
                if t-dt/2<time_dg.value<t+dt/2:
                    return i
            return None
        
#    def indexOfDomain(self,d,time_index,sim):
#        domain_dg = DataGroup(self.file[dgTypes["domain"]+"/"+sim])[time_index]
#        dr = domain_dg[1] - domain_dg[0]
#        for i,pos in enumerate(domain_dg):
#            if d-dr/2<=pos<d+dr/2:
#                return i
#        return -1    
    
    def write(self,dgType,sim,it,data,name = None,derivedAttrs = None,
        overwrite = True):
        """This method allows for writing to SimulationHDF objects. Note that if dgType is an error type data group then name must be given. We recommend that its value be taken as the data group from which the error data was generated."""        
        # Create empy derivedAttrs if no argument is passed
        if derivedAttrs is None:
            self.derivedAttrs = {}
        else:
            self.derivedAttrs = derivedAttrs
        
        #If dgType is an error type then name must be set.
        #if dgType == sd.dgTypes["errorNum"] or\
        #    dgType == sd.dgTypes["errorExa"]:
        #    if name is None:
        #        raise Exception("""If SimulationHDF.write() is based an error dgType the name keyword must be set. Suggested usage is that name = the dgType of the data from which the error data was calculated.""")
        
        # get name if not none
        if name is not None:
            dg_name = dgType+"/"+sim+"/"+name
        else:
            dg_name = dgType+"/"+sim
        
        # get dg
        if overwrite:
            if name is not None:
                dg = DataGroup(self.file.require_group(dgType)\
                    .require_group(sim)\
                    .require_group(name))
            else:
                dg = DataGroup(self.file.require_group(dgType)\
                    .require_group(sim))
        else:
            if name is not None:
                dg = DataGroup(self.file.create_group(dgType)\
                    .create_group(sim)\
                    .create_group(name))
            else:
                dg = DataGroup(self.file.create_group(dgType)\
                    .create_group(sim))
        
        # add data and derived attrs
        dg[it] = data
        for key,value in self.derivedAttrs.items():
            dg[it].attrs[key] = value

# A utility class to write to SimulationHDF via the
# actions.py framework.
class SimOutput(actions.UserAction):
    def __init__(self, hdf_file,solver,theSystem,theInterval, 
        actionTypes,frequency = 1,name = None,cmp = None,overwrite = True,\
        debug_parent = "main.IBVP"):
        self.log = logging.getLogger(debug_parent+".SimOutput")
        if __debug__:
            self.log.debug("Setting up HDF output...")
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
        self.cmp = cmp
        for action in actionTypes:
            action.setup(self)
        if __debug__:
            self.log.debug("HDF output setup completed.")
        
    def _doit(self,it,u):
        for action in self.actions:
            if __debug__:
                self.log.debug("Outputting %s"%action.groupname)
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
            self.data_group.attrs['cmp'] = parent.cmp
            self.log = logging.getLogger(self.groupname)
        
        def __call__(self,it,u):
            for key,value in self.derivedAttrs.items():
                v = value(it,u,self.parent.system)
                if __debug__:
                    self.log.debug("Derived Attrs = %s"%str(v))
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
            dg[it] = parent.system.exactValue(u)
            super(SimOutput.Exact, self).__call__(it, u)

    class Scrif(SimOutputType):
        
        groupname = dgTypes["scrif"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = (1./self.parent.system.mu(u.x))-u.time
            super(SimOutput.Scrif,self).__call__(it,u)
            
    class Mu(SimOutputType):
        
        groupname = dgTypes["mu"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = self.parent.system.mu(u.x)
            super(SimOutput.Mu,self).__call__(it,u)

    class Mup(SimOutputType):
        
        groupname = dgTypes["mup"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = self.parent.system.mup(u.x)
            super(SimOutput.Mup,self).__call__(it,u)

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
            dg[it] = u.x
            super(SimOutput.Domains,self).__call__(it,u)

    class Constraints(SimOutputType):

        groupname = dgTypes["constraint"]
        
        def __call__(self,it,u):
            dg = self.data_group
            parent = self.parent
            dg[it] = parent.system.constraint_violation(u)
            super(SimOutput.Constraints,self).__call__(it,u)

    class WeylConstants(SimOutputType):

        groupname = dgTypes["IJ"]
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = self.parent.system.weyl_constantsIJ(it,u)
            super(SimOutput.WeylConstants,self).__call__(it,u)

    class DerivedData(SimOutputType):
        
        def __init__(self,name, function,derivedAttrs = None):
            self.func = function
            self.groupname = name
            super(SimOutput.DerivedData,self).__init__(derivedAttrs)
        
        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = self.func(it,u)
            super(SimOutput.DerivedData,self).__call__(it,u)

    class Exact(SimOutputType):

        groupname = dgTypes["exact"]

        def __call__(self,it,u):
            dg = self.data_group
            parent = self.parent
            dg[it] = parent.system.exactValue(u.time,u.x).fields
            super(SimOutput.Exact, self).__call__(it, u)

    class System(SimOutputType):
        
        groupname = systemD
        
        def setup(self,parent):
            super(SimOutput.System,self).setup(parent)
            g =self.data_group.group
#            psystem = np.asarray(cPickle.dumps(parent.system,-1))
#            pgrid = np.asarray(cPickle.dumps(parent.grid,-1))
#            psolver = np.asarray(cPickle.dumps(parent.solver))
#            pcmp = np.asarray(cPickle.dumps(parent.cmp))
            psystem = np.asarray(repr(parent.system))
            pgrid = np.asarray(repr(parent.grid))
            psolver = np.asarray(repr(parent.solver))
            pcmp = np.asarray(repr(parent.cmp))
            g.require_dataset(sysDTypes['system'],\
                psystem.shape,psystem.dtype,\
                data=psystem)
            g.require_dataset(sysDTypes['grid'],\
                pgrid.shape,pgrid.dtype,data=pgrid)
            g.require_dataset(sysDTypes['solver'],\
                psolver.shape,psolver.dtype,\
                data = psolver)
            g.require_dataset(sysDTypes['cmp'],\
                pcmp.shape,pcmp.dtype,data=pcmp)
        
        def __call__(self,it,u):
            pass


# A wrapper class that helps ease iteration over datasets with 
# str(int) indices going from 0,1,... upwards.
class DataGroup():
    """The DataGroup class wraps a h5py group so that the setter, getter
    and iter methods do sensible array like things.
    """
    global group
    
    @property
    def attrs(self):
        return self.group.attrs
    
    @property
    def name(self):
        return self.group.name
    
    def attrs_list(self, kwd):
        list = []
        for data_set in self:
            list += data_set.attrs[kwd]
        return list
    
    def index_of_attr(self, attr, value, start_index = 0, \
        value_comparor = lambda x:x==value):
        """Returns the index of the dataset in self whose attribute attr
        has value value. The function value_comparor allows for fudging a 
        little."""
        index = -1
        for i in range(start_index, len(self)):
            if value_comparor(self[i].attrs[attr]):
                index = self[i].attrs['index']
                break
        return index
    
#    def index_of(self,value):
#        min_v = self[0]
#        max_v = self[-1]
#        if value <= min_v:
#            return 0
#        elif value>= max_v:
#            return len(times_dg)-1
#        else:
#            for i,v in enumerate(self):
#                dt = times_dg[i+1].value-time_dg.value
#                if t-dt/2<time_dg.value<t+dt/2:
#                    return i
#            return None
    
    def __init__(self, grp,returnValue=False):
        """The group to behave like an array. It is assumed that
        the group has/will have a number of datasets with the labels
        '0','1', etc... 
        """
        self.group = grp
        self.rV = returnValue

    def __iter__(self):
        """Iterates in increasing numerical order 0,1,2,3,...
        across the datasets of the group. Returns the dataset
        at position 0,1,2,3...
        """
        i = 0
        while True:
            try:
                yield  self[i]
                i+=1
            except:
                return

    def __len__(self):
        return len(self.group)
    
    def __setitem__(self, i, value):
        value = np.array(value)
        dataset = self.group.require_dataset(str(i), value.shape,  value.dtype)
        dataset[:] = value
        dataset.attrs['index'] = i
    
    def __getitem__(self, i):
        if self.rV:
          return self.group[str(i)].value
        return self.group[str(i)]
        
    def __repr__(self):
        return r"<H5pyArray datagroup %s (%d)>"% (self.name, len(self))       
        
def binarysearch(a,low,high,value):
    if high<low:
        return -1
    mid = int(low +(high-low)/2.)
    if mid ==0:
        return 0
    if value<=a[mid-1]:
        return binarysearch(a,low,mid,value)
    elif value>a[mid]:
        return binarysearch(a,mid+1,high,value)
    else:
        return mid  
        
# array_value_index_mapping takes two arrays and returns a list of pairs
# of indices (index1,index2) so that correct[index1] = comparison[index2]
# this is very useful when performing error calculations
def array_value_index_mapping(correct,comparison,\
    compare_function= lambda x, y:x==y,\
    compare_on_axes = 1):
    index_mapping = []
    cor_dims = len(correct.shape)
    com_dims = len(comparison.shape)
    if compare_on_axes == 0:
        cor_ind = [0 for i in range(cor_dims)]
        com_ind = [0 for i in range(com_dims)]
    else:
        cor_ind = [0 for i in range(cor_dims-1)]
        com_ind = [0 for i in range(com_dims-1)]
    return _array_value_index_mapping_recursive(correct,cor_ind,\
        comparison,com_ind,index_mapping,compare_on_axes,0)

def _array_value_index_mapping_recursive(correct,cor_ind,\
    comparison,com_ind,\
    index_mapping,compare_on_axes,depth):
    while cor_ind[depth] < correct.shape[depth] and\
         com_ind[depth] < comparison.shape[depth]:
         if compare_on_axes == 0:
             com = comparison[tuple(com_ind)]
             cor = correct[tuple(cor_ind)]
         else:
             com = comparison[tuple(com_ind)][depth]
             cor = correct[tuple(cor_ind)][depth]
         if com < cor:
            com_ind[depth] = com_ind[depth]+1
         elif com > cor:
            cor_ind[depth] = cor_ind[depth]+1
         elif com == cor:
            if depth == len(correct.shape)-1-compare_on_axes:
                index_mapping += [(tuple(cor_ind),tuple(com_ind))]
                com_ind[depth] = com_ind[depth]+1
                cor_ind[depth] = cor_ind[depth]+1
            else:
                index_mapping = _array_value_index_mapping_recursive(\
                    correct,cor_ind,\
                    comparison,com_ind,index_mapping,compare_on_axes,depth+1)
                com_ind[depth] = com_ind[depth]+1
                cor_ind[depth] = cor_ind[depth]+1
                com_ind[depth+1] = 0
                cor_ind[depth+1] = 0
         else:
            raise Exception("Unable to compare %s and %s"%\
                                    (com,cor))
    return index_mapping


#def array_value_index_mapping(\
#    correct,\
#    comparison,\
#    compare_function= lambda x, y:x==y,\
#    compare_last_axes = 0,\
#    ):
#    index_mapping = []
#    _list_comparor_recursive([],  correct, [], comparison, index_mapping,\
#        compare_function,compare_last_axes = compare_last_axes)
#    return index_mapping
#
#def _list_comparor_recursive(correct_index,  correct_axes, comparison_index,\
#        comparison_axes, reduced_comparison_axes, compare_function,\
#        compare_last_axes = 0):
#    if len(correct_axes.shape) == compare_last_axes and\
#        len(comparison_axes.shape) == compare_last_axes:
#        if compare_last_axes == 0 and compare_function(correct_axes, comparison_axes)\
#            or compare_last_axes !=0 and all(compare_function(correct_axes, comparison_axes)):
#            if len(correct_index) == 1:
#                reduced_comparison_axes.append((correct_index[0], \
#                    comparison_index[0]))
#            else:
#                reduced_comparison_axes.append((tuple(correct_index), \
#                    tuple(comparison_index)))
#    elif not len(correct_axes.shape) == compare_last_axes:
#        for i, row in enumerate(correct_axes):
#            correct_index.append(i)
#            _list_comparor_recursive(correct_index, row, comparison_index, \
#                comparison_axes,  reduced_comparison_axes, compare_function, \
#                compare_last_axes = compare_last_axes)
#            correct_index.pop()
#    elif not len(comparison_axes.shape) == compare_last_axes:
#        for i, row in enumerate(comparison_axes):
#            comparison_index.append(i)
#            _list_comparor_recursive(correct_index,  correct_axes, \
#                comparison_index,\
#                row, reduced_comparison_axes, compare_function,\
#                compare_last_axes = compare_last_axes)
#            comparison_index.pop()
