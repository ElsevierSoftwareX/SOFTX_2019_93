from mpi4py import MPI
import logging
import numpy as np

class OneD_even:

    def __init__(self,array_length,ghost_points,log = None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.boundary, self.communicate, self.rebuild, self.domain = \
            self._mpi_domain_slices(ghost_points,array_length)
        if log is None:
            self.log = logging.getLogger("OneD_even")
        else:
            self.log = log.getChild("OneD_even")
    
    def send(self,data):
        if self.communicate is None: return
        for comm_info in self.communicate:
            dest_offset, send_slice = comm_info[0]
            self.comm.send(data[:,send_slice],dest = self.rank+dest_offset)
            
    def recv(self,data):
        if self.communicate is None: return
        for comm_info in self.communicate:
            dest_offset, recv_slice = comm_info[1]
            data[:,recv_slice] = \
                self.comm.recv(source = self.rank+dest_offset)
    
    def collect_data(self,domain,data):
        #If there are no partitions of the domain we don't need to collect
        if self.size == 1:
            return data
        #First get the data that needs to be sent
        field = data[:,self.rebuild]
        #Second gather the data
        fields = self.mpicomm.gather(field, root = 0)
        if self.rank == 0:
            array_shape = reduce(lambda x,y:x+y,[f.shape[1] for f in fields])
            if __debug__:
                self.log.debug("Collected array length is %i"%array_shape)
            rdata = np.zeros((5,array_shape))
            start_i = 0
            for i in range(len(fields)):
                f = fields[i]
                fshape = f.shape[1]
                rdata[:,start_i:fshape+start_i] = f
                start_i = fshape+start_i
            return rdata
        else: 
            return None
    
    def _mpi_domain_indices(self,array_length):
        if __debug__:
            self.log.debug("Calculating start and end indices for subdomain")
            self.log.debug("Array_length is %i"%array_length)
        q,r = divmod(array_length, self.size)
        if __debug__:
            self.log.debug("q = %i, r = %i"%(q,r))
        s = self.rank*q + min(self.rank,r)
        e = s + q
        if self.rank < r:
            if __debug__:
                self.log.debug("Self.mpirank > r so we add one to end point")
            e = e + 1
        if __debug__:
            self.log.debug("Start index = %i, End index = %i"%(s,e))
        return s,e
        
    def _mpi_domain_slices(self,num_ghost_points,array_length):
        if self.size == 1:
            return (None,None,slice(None),slice(None))
        s,e = self._domain_indices(array_length)
        if self.mpirank == 0:
            return (-1,\
                [[(1,slice(-2*gp,-gp)),(1,slice(-gp,None))]],\
                slice(s,e),\
                slice(s,e+gp))
        elif self.mpirank == self.mpisize-1:
            return (1,\
                [[(-1,slice(gp,2*gp)),(-1,slice(None,gp))]],\
                slice(s,e),\
                slice(s - gp, e))
        else:
            return (0,\
                [[(1,slice(-2*gp,-gp)),(1,slice(-gp,None))],\
                    [(-1,slice(gp,2*gp)),(-1,slice(None,gp))]],\
                slice(s,e),\
                slice(s-gp, e+gp))

class PT_2D_1D:
    """This class implements the mpiinterfaces api for a 2D grid in which
    only one dimension, the first, has been divided over multiple cpus."""

    def __init__(self,array_length):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        if __debug__:
            self.log = logging.getLogger("PT_2D_1D")
        self.rebuild,self.domain = self._mpi_domain_slices(1,array_length)

        
    def _communication_required(self):
        return self.size != 1
            
    def get_edge(self,data):
        if self._communication_required():
            if self.rank == 0:
                v = np.empty_like(data[-1])
                self.comm.send(data[-1],dest = 1)
                v = self.comm.recv(source = 1)
                rv = [(slice(-1,None),v)]
            elif self.rank == self.size -1:
                v = np.empty_like(data[0])
                self.comm.send(data[0],dest = self.rank-1)
                v = self.comm.recv(source = self.rank-1)
                rv = [(slice(0,1),v)]
            else:
                v = np.empty_like(data[0])
                w = np.empty_like(data[-1])
                self.comm.send(data[0], dest = self.rank-1)
                self.comm.send(data[-1], dest = self.rank+1)
                v = self.comm.recv(source = self.rank-1)
                w = self.comm.recv(source = self.rank+1)
                rv = [(slice(0,1),v),(slice(-1,None),w)]
        else:
            rv = []
        return rv
        
    def collect_data(self,data):
        #If there are no partitions of the domain we don't need to collect
        if not self._communication_required():
            return data
        #First get the data that needs to be sent
        if __debug__:
            #self.log.debug("self.rank = %i"%self.rank)
            self.log.debug("self.rebuild = %s"%(repr(self.rebuild)))
            self.log.debug("data.shape = %s"%repr(data.shape))
        field = data[:,self.rebuild,:]
        if __debug__:
            self.log.debug("field.shape = %s"%repr(field.shape))
        #Second gather the data
        fields = self.comm.gather(field, root = 0)
        if self.rank == 0:
            if __debug__:
                self.log.debug("The collected fields have shapes %s"%\
                    repr([f.shape for f in fields]))
            array_shape = reduce(lambda x,y:x+y,[f.shape[1] for f in fields])
            rdata = np.zeros((data.shape[0],array_shape,data.shape[2]))
            if __debug__:
                self.log.debug("Collected array length is %i"%array_shape)
                self.log.debug("rdata.shape = %s"%(repr(rdata.shape)))
            start_i = 0
            for i in range(len(fields)):
                f = fields[i]
                fshape = f.shape[1]
                rdata[:,start_i:fshape+start_i,:] = f
                start_i = fshape+start_i
            return rdata
        else: 
            return None
        
        
    def _mpi_domain_indices(self,array_length):
        if __debug__:
            self.log.debug("Calculating start and end indices for subdomain")
            self.log.debug("Array_length is %i"%array_length)
        q,r = divmod(array_length, self.size)
        if __debug__:
            self.log.debug("q = %i, r = %i"%(q,r))
        s = self.rank*q + min(self.rank,r)
        e = s + q
        if self.rank < r:
            if __debug__:
                self.log.debug("Self.mpirank < r so we add one to end point")
            e = e + 1
        if __debug__:
            self.log.debug("Start index = %i, End index = %i"%(s,e))
        return s,e
        
    def _mpi_domain_slices(self,num_ghost_points,array_length):
        if self._communication_required():
            s,e = self._mpi_domain_indices(array_length)
            if self.rank == 0:
                return [slice(None,None),slice(s,e)]
            elif self.rank == self.size-1:
                return [slice(1,None),slice(s-1,e)]
            else:
                return [slice(1,None),slice(s-1,e)]
        else:
            return [slice(None),slice(None)]
