import functools
import h5py
import cPickle
import numpy as np
import sys

sys.path.append("../../EvolutionSBP/")
import system
import ibvp
import diffop
import actions
import solvers
import hdf_output
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
    "raw":"Raw",\
    "error":"Error",\
    "weyl":"Weyl",\
    "domain":"Domain",\
    "time":"Time",\
    "dt": "TimeStep",\
    "constraint": "Constraint"\
    }

dgTypesInv = dict(zip(dgTypes.values(),dgTypes.keys()))

# SystemDataTypes stores a list of all the subgroups in the system groups.

systemD = "System"

sysDTypes = {\
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
                setattr(self,key,cPickle.loads(\
                    self.simHDF[systemD+"/"+self.name][key].value\
                    ))
        self.cmp = float(self.cmp)
        existing_items = self.simHDF.file.keys()
        for key, item in dgTypes.items():
            if item in existing_items:
                if self.name in self.simHDF[item].keys():
                    setattr(self,key,\
                        Sim.dsReturnValue(\
                            DataGroup(self.simHDF[item+"/"+self.name])
                        ))
    
    def tslice(self,i):
        return self.simHDF.tslice(i,self.name,dgType = dgType["raw"])
    
    def indexOfTime(self,t):
        return self.simHDF.indexOfTime(t,self.name)
    
    def __eq__(self,other):
        return self.cmp == other.cmp
        
    def __lt__(self,other):
        return self.cmp < other.cmp
    
    def __str__(self):
        return self.name
    
    def write(self,dgType,it,data,name = None,derivedAttrs = None):
        self.simHDF.write(dgType,self.name,it,data,name,derivedAttrs)
    
    def plot(self,*args,**kwds):
        dgType = args[0]
        self.GNUplot(dgType,**kwds)
    
    def getDgType(self,dgType):
        return self.simHDF.getDgType(dgType,self.name)
        
    def getDgTypeAttr(self,dgType,attr,i):
        return self.simHDF.getDgTypeAttr(dgType,attr,i,self.name)
        
    def GNUplot(group,  xlabel=None, ylabel=None,\
        yrange=None,  graphPause=0.01, pauseAtEnd=True):
        """A utility function which plots a given group.
        If group = None then the current data_group is used.
        """
        #Ensure that group is a datagroup
        group = self.dgType
        domains = self.domains
        times = self.times

        #Initialize gnuplot
        gnu = Gnuplot.Gnuplot(debug=0)
        gnu('set style data lines') 
        gnu.xlabel(xlabel)
        gnu.ylabel(ylabel)
        if yrange != None:
            gnu('set yrange '+yrange)
        
        #Iterate across group
        for i,y in enumerate(group):
            gnu.title('Time {0}'.format(times[i])) 
            plotItems = []
            for j,row in enumerate(numpy.atleast_2d(dataset.value)):
                plotItems +=[Gnuplot.Data(domains[i],row, \
                    title = 'Component '+str(j))]
            gnu.plot(*plotItems)
            time.sleep(graphPause)
        
        if pauseAtEnd:
            raw_input('Press key to continue')


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
    
    def indexOfTime(self,t,sim):
        times_dg = DataGroup(self.file[dgTypes["time"]+"/"+sim])
        for i,time_dg in enumerate(times_dg):
            dt = times_dg[i+1].value-time_dg.value
            if t-dt/2<time_dg.value<t+dt/2:
                return i
        return -1
    
    def write(self,dgType,sim,it,data,name = None,derivedAttrs = None,
        overwrite = True):
        # Create empy derivedAttrs if no argument is passed
        if derivedAttrs is None:
            self.derivedAttrs = {}
        else:
            self.derivedAttrs = derivedAttrs
        
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
            v = value(it,u,self.parent.system)
            self.data_group[it].attrs[key] = v

# A utility class to write to SimulationHDF via the
# actions.py framework.
class SimOutput(actions.UserAction):
    def __init__(self, hdf_file,solver,theSystem,theInterval, 
        actionTypes,frequency = 1,name = None,cmp = None,overwrite = True):
        print "Setting up HDF output...",
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
        print "Done.-"
        
    def _doit(self,it,u):
        for action in self.actions:
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
        
        def __call__(self,it,u):
            for key,value in self.derivedAttrs.items():
                v = value(it,u,self.parent.system)
                if __debug__:
                    print "%s"%str(v)
                self.data_group[it].attrs[key] = v
         
    class Data(SimOutputType):

        groupname = dgTypes["raw"]

        def __call__(self,it,u):
            dg = self.data_group
            dg[it] = u.fields
            super(SimOutput.Data,self).__call__(it,u)

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

        groupname = dgTypes["weyl"]
        
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

    class System(SimOutputType):
        
        groupname = systemD
        
        def setup(self,parent):
            super(SimOutput.System,self).setup(parent)
            g =self.data_group.group
            psystem = np.asarray(cPickle.dumps(parent.system,-1))
            pgrid = np.asarray(cPickle.dumps(parent.grid,-1))
            psolver = np.asarray(cPickle.dumps(parent.solver))
            pcmp = np.asarray(cPickle.dumps(parent.cmp))
            g.require_dataset('system',psystem.shape,psystem.dtype,\
                data=psystem)
            g.require_dataset('grid',pgrid.shape,pgrid.dtype,data=pgrid)
            g.require_dataset('solver',psolver.shape,psolver.dtype,\
                data = psolver)
            g.require_dataset('cmp',pcmp.shape,pcmp.dtype,data=pcmp)
        
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
    
    def __init__(self, grp):
        """The group to behave like an array. It is assumed that
        the group has/will have a number of datasets with the labels
        '0','1', etc... 
        """
        self.group = grp

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
        return self.group[str(i)]
        
    def __repr__(self):
        return r"<H5pyArray datagroup %s (%d)>"% (self.name, len(self))       
        
# array_value_index_mapping takes two arrays and returns a list of pairs
# of indices (index1,index2) so that correct[index1] = comparison[index2]
# this is very useful when performing error calculations
def array_value_index_mapping(\
    correct,\
    comparison,\
    compare_function= lambda x, y:x==y\
    ):
    index_mapping = []
    _list_comparor_recursive([],  correct, [], comparison, index_mapping,\
        compare_function)
    return index_mapping

def _list_comparor_recursive(correct_index,  correct_axes, comparison_index,\
        comparison_axes, reduced_comparison_axes, compare_function):
    if correct_axes.shape==() and comparison_axes.shape==():
        if compare_function(correct_axes, comparison_axes):
            if len(correct_index) == 1:
                reduced_comparison_axes.append((correct_index[0], \
                    comparison_index[0]))
            else:
                reduced_comparison_axes.append((tuple(correct_index), \
                    tuple(comparison_index)))
    elif not correct_axes.shape == ():
        for i, row in enumerate(correct_axes):
            correct_index.append(i)
            _list_comparor_recursive(correct_index, row, comparison_index, \
                comparison_axes,  reduced_comparison_axes, compare_function)
            correct_index.pop()
    elif not comparison_axes.shape == ():
        for i, row in enumerate(comparison_axes):
            comparison_index.append(i)
            _list_comparor_recursive(correct_index,  correct_axes, \
                comparison_index,\
                row, reduced_comparison_axes, compare_function)
            comparison_index.pop()
