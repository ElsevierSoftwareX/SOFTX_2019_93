from mpi4py import MPI
import logging
import abc
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
        self.log = logging.getLogger("MPIInterface")
        super(MPIInterface, self).__init__(*args, **kwds)

    def sendrecv_replace(self, data):
        neighbours = self._get_neighbours_dataslice(data.shape)
        for source, dest, data_slice in neighbours:
            sr_data = data[data_slice]
            self.comm.Sendrevc_replace(
                sr_data,
                dest = dest,
                source = self.comm.rank
                )

    def send(self, data):
        neighbours = self._get_neighbours_dataslice(data.shape)
        for source, dest, data_slice in neighbours:
            s_data = data[data_slice]
            self.comm.Send(
                sr_data,
                dest = dest,
                )
        
    def recv(self, data):
        neighbours = self._get_neighbours_dataslice(data.shape)
        for source, dest, data_slice in neighbours:
            r_data = data[data_slice]
            self.comm.Recv(
                r_data,
                source = source,
                )

    @abc.abstractmethod
    def _get_neighbours_dataslice(self, shape):
        """This method should return a interable of tuples. Each tuple
        is (source, dest, slice). Source is this process, so in almost all
        cases it will be self.mpi.rank. dest is the rank of the neighbour.
        slice is a slice object that extracts the data to be sent, or the
        portion of data into which data will be written."""
      
###############################################################################
# Concrete implementations
###############################################################################
class EvenCart(MPIInterface):
    """A wrapper for MPI.Cartcomm that manages the selection of the relvant
    portion of a data field for send/recv mpi commands. data.shape must match
    with MPI.Cartcomm.dim and MPI.Cartcomm.dims. This class uses
    MPI.Cartcomm.Shift to ensure that periodicity of grids is accounted for.

    It is recommended that if any dimension is periodic that the sendrecv
    method is used since this relies on MPI_SENDRECV that will avoid
    blocking."""

    def __init__(self, ghost_points, *args, **kwds):
        """Arguments:
        ghost_points: an integer that describes how many overlapping grid
        points each grid contains on each boundary.
        mpi_cart_comm: an MPI_CART object wrapped by mpi4py. This object
        describes the topology of the processors."""
        self.gh = ghost_points
        super(EvenCart, self).__init__(*args, **kwds)

    def _get_neighbours_dataslice(self, shape):
        dims = self.comm.Get_dim()
        neighbours = []
        for d in range(dims):
            source, dest = self.comm.Shift(d, 1)
            if dest is not None:
                data_slice = _get_dataslice(shape, d, 1)  
                neigbours += [(dest, data_slice)]
            source, dest = self.comm.Shift(d, -1)
            if dest is not None:
                data_slice = _get_dataslice(shape, d, -1)  
                neigbours += [(dest, data_slice)]
        return neighbours

    def _get_dataslice(shape, dim, direction):
        data_slice = [slice(None,None,None) for d in shape]
        if direction == 1:
            data_slice[dim] = slice(-self.gh, None, None)
        elif direction == -1:
            data_slice[dim] = slice(self.gh, None, None)
        return tuple(data_slice)

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
