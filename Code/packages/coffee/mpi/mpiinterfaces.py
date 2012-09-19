import abc
from mpi4py import MPI
import logging
import numpy as np

###############################################################################
# Abstract Base Class for MPI Interfaces
###############################################################################
class MPIInterface(object):
     
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, mpi_comm, *args, **kwds):
        """Arguments:
        mpi_comm: an MPI_COMM object wrapped by mpi4py."""
        self.comm = mpi_comm
        super(MPIInterface, self).__init__(*args, **kwds)

    @abc.abstractmethod
    def subdomain(self, shape):
        """Returns a slice object that represents the subdomain of the full
        grid that each grid object associated to each process should
        managed."""

    @abc.abstractmethod
    def communicate(self, data):
        """This method takes the data corresponding to the relevant grid,
        sends it to the right place, recieves the relevant data and performs
        the correct method to update the original data."""
    
    @abc.abstractmethod
    def neighbour_slices(self, shape):
        """This method should return a interable of tuples. Each tuple
        is (source, dest, slice). Source is this process, so in almost all
        cases it will be self.mpi.rank. dest is the rank of the neighbour.
        slice is a slice object that extracts the data to be sent, or the
        portion of data into which data will be written."""

###############################################################################
# Concrete implementations
###############################################################################
class EvenCart(MPIInterface):

    def __init__(self, domain, *args, **kwds):
        super(EvenCart, self).__init__(*args, **kwds)
        self.log = logging.getLogger("EvenCart")
        self.domain = domain
        self.domain_mapping = self._make_domain_mappings(domain)

    def _make_domain_mapping(self, domain):
        if self.comm is None:
            return tuple([slice(None, None, None) for dim in domain])
        r_map = []
        for rank in range(size):
            coords = self.comm.Get_coords(rank)
            r_map += [tuple([
                _array_slice(domain[i], coord)
                for i, coord in enumerate(coords)
                ])
                ]
        return r_map                
    
    def neighbour_slices(self, shape):
        if self.comm is None:
            return []
        dims = self.comm.Get_dim()
        neighbours = []
        for d in range(dims):
            source, dest = self.comm.Shift(d, 1)
            if dest is not None:
                data_slice = _get_dataslice(shape, d, 1)  
                neigbours += [(self.comm.rank, dest, data_slice)]
            source, dest = self.comm.Shift(d, -1)
            if dest is not None:
                data_slice = _get_dataslice(shape, d, -1)  
                neigbours += [(self.comm.rank, dest, data_slice)]
        return neighbours

    def _get_dataslice(shape, dim, direction):
        data_slice = [slice(None,None,None) for d in shape]
        if direction == 1:
            data_slice[dim] = slice(-1, None, None)
        elif direction == -1:
            data_slice[dim] = slice(None, 1, None)
        return tuple(data_slice)

    def _array_indices(self, array_length, rank=self.comm.rank):
        if __debug__:
            self.log.debug("Calculating start and end indices for  subdomain")
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
        return slice(s, e, None)
          
    @property
    def subdomain(self):
        return tuple(self.domain_mapping[self.comm.rank])
            
    def communicate(self, data):
        nslices = neighbour_slices(self, data.shape)
        r_data = []
        for source, dest, dslice in nslices:
            recv_data = np.empty_like(data[dslice])
            self.comm.sendrecv(
                sendobj=data[dslice],
                dest=dest,
                recvobj=recv_data,
                source=source
                )
            r_data += [(dslice, recv_data)]
        return r_data

    def collect_data(self, data):
        #If there are no partitions of the domain we don't need to collect
        if self.comm is None:
            return data
        #Gather the data
        fields = self.comm.gather(data, root = 0)
        #If this process is root then collate the data and return
        if self.rank == 0:
            if __debug__:
                self.log.debug("The collected fields have shapes %s"%\
                    repr([f.shape for f in fields]))
            rdata = np.zeros((data.shape[0],) + self.domain)
            if __debug__:
                self.log.debug("rdata.shape = %s"%(repr(rdata.shape)))
            for rank, field in fields:
                dslice = self.domain_mapping[rank]
                rdata[(slice(None, None, None),)+ dslice] = field
            return rdata
        else: 
            return None

