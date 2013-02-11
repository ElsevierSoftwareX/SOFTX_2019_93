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
        """Returns a list of tuples. The first element of each tuple is a slice
        representing the portion of data which corresponds to the second
        element of the tuple. The second element of the tuple is an array of
        data which are the communicated values that correspond to the 
        positions described by the slice."""

    @abc.abstractmethod
    def boundary_slices(self, shape):
        """Returns slices which represent the portions of the domain which
        are on a external boundary."""

    def barrier(self):
        if self.comm is None:
            return
        return self.comm.barrier()

###############################################################################
# Concrete implementations
###############################################################################
class EvenCart(MPIInterface):

    def __init__(self, domain, ghost_points=1, *args, **kwds):
        super(EvenCart, self).__init__(*args, **kwds)
        self.log = logging.getLogger("EvenCart")
        self.domain = domain
        self.gp = ghost_points
        self.domain_mapping = self._make_domain_mappings(domain)

    def boundary_slices(self, shape):
        #if __debug__:
            #self.log.debug("Calculating boundary slices in mpi")
        extra_dims = len(shape) - len(self.domain)
        edims_shape = shape[:extra_dims]
        edims_slice = tuple([
            slice(None,None,None)
            for i in range(extra_dims)
            ])
        coords = self.comm.Get_coords(self.comm.rank)
        #if __debug__:
            #self.log.debug("coords is %s"%coords)
        dims = self.comm.dims
        r_slices = []
        periods = self.comm.periods
        for i, dim in enumerate(dims):
            if periods[i]:
                continue
            if coords[i] == 0 and coords[i] == dim - 1:
                r_slice = [
                    slice(None, None, None)
                    for d in self.domain
                    ]
                r_slice[i] = slice(None, 1, None)
                r_slices += [(i, -1, edims_slice + tuple(r_slice))]
                r_slice = [
                    slice(None, None, None)
                    for d in self.domain
                    ]
                r_slice[i] = slice(-1, None, None)
                r_slices += [(i, 1, edims_slice + tuple(r_slice))]
            elif coords[i] == 0:
                r_slice = [
                    slice(None, None, None)
                    for d in self.domain
                    ]
                r_slice[i] = slice(None, 1, None)
                r_slices += [(i, -1, edims_slice + tuple(r_slice))]
            elif coords[i] == dim-1:
                r_slice = [
                    slice(None, None, None)
                    for d in self.domain
                    ]
                r_slice[i] = slice(-1, None, None)
                r_slices += [(i, 1, edims_slice + tuple(r_slice))]
        return r_slices


    def _make_domain_mappings(self, domain):
        if self.comm is None:
            return tuple([slice(None, None, None) for dim in domain])
        r_map = []
        for rank in range(self.comm.size):
            coords = self.comm.Get_coords(rank)
            r_map += [tuple([
                self._array_slice(domain[i], coord, self.comm.dims[i])
                for i, coord in enumerate(coords)
                ])
                ]
        #if __debug__:
            #self.log.debug("domain_mapping is %s"%r_map)
        return r_map                
    
    def _neighbour_slices(self, shape):
        if self.comm is None:
            return []
        dims = self.comm.Get_dim()
        neighbours = []
        for d in range(dims):
            source, dest = self.comm.Shift(d, 1)
            #if __debug__:
                #self.log.debug("source=%d, dest=%d"%(source, dest))
            send_slice, recv_slice = self._get_dataslice(
                source, dest,
                shape, d, 1
                )
            neighbours += [(source, dest, send_slice, recv_slice)]
            source, dest = self.comm.Shift(d, -1)
            send_slice, recv_slice = self._get_dataslice(
                source, dest,
                shape, d, -1
                )
            neighbours += [(source, dest, send_slice, recv_slice)]
        #if __debug__:
            #self.log.debug("neighbour_slices is %s"%neighbours)
        return neighbours

    def _get_dataslice(self, source, dest, shape, dim, direction):
        if dest == -1:
            rsend_slice = None
        else:
            send_slice = [slice(None,None,None) for d in shape]
        if source == -1:
            rrecv_slice = None
        else:
            recv_slice = [slice(None,None,None) for d in shape]
        dim_index = len(shape) - self.comm.Get_dim() + dim
        if direction == 1:
            if dest is not -1:
                send_slice[dim_index] = slice(-self.gp, None, None)
                rsend_slice = tuple(send_slice)
            if source is not -1:
                recv_slice[dim_index] = slice(None, self.gp, None)
                rrecv_slice = tuple(recv_slice)
        elif direction == -1:
            if dest is not -1:
                send_slice[dim_index] = slice(None, self.gp, None)
                rsend_slice = tuple(send_slice)
            if source is not -1:
                recv_slice[dim_index] = slice(-self.gp, None, None)
                rrecv_slice = tuple(recv_slice)
        return rsend_slice, rrecv_slice

    def _array_slice(self, array_length, rank, num_ranks):
        if __debug__:
            self.log.debug("Calculating start and end indices for  subdomain")
            self.log.debug("Array_length is %i"%array_length)
        #divide domain into appropriate parts
        q,r = divmod(array_length, num_ranks)
        if __debug__:
            self.log.debug("q = %i, r = %i"%(q,r))
        #use the rank to details which part is relevant for this process
        s = rank * q + min(rank, r)
        e = s + q
        #Adjust e to account for min(rank, r) term, which spreads the remainder
        #over the appropriate number of processes
        if rank < r:
            if __debug__:
                self.log.debug("rank < r so we add one to end point")
            e = e + 1
        #add in ghost_points if we can
        #this currently works for gp = 1 for an SAT boudnary method.
        #The code will need to be checked for consistency if this changes.
        if s - self.gp > -1:
            s = s - self.gp + 1
        if e + self.gp < array_length:
            e = e + self.gp
        if __debug__:
            self.log.debug("Start index = %i, End index = %i"%(s,e))
        return slice(s, e, None)
          
    @property
    def subdomain(self):
        if self.comm is None:
            return self.domain_mapping
        return tuple(self.domain_mapping[self.comm.rank])
            
    def communicate(self, data):
        #if self.comm.size == 1:
            #return []
        nslices = self._neighbour_slices(data.shape)
        #if __debug__:
            #self.log.debug("about to perform communication")
            #self.log.debug("nslices = %s"%(repr(nslices)))
            #self.log.debug("data is %s"%repr(data))
        r_data = []
        for source, dest, send_slice, recv_slice in nslices:
            #if __debug__:
                #self.log.debug(
                    #"source=%d, dest=%d, send_slice=%s, recv_slice=%s"%
                    #(source, dest, repr(send_slice), repr(recv_slice))
                    #)
            if dest < 0:
                send_data = None
            else:
                #I really did not want to copy this view. For large data
                #sets this feels like an unnecessary delay.
                #However mpi4py did not seem to work with views and using
                #self.comm.sendrecv to send and recv arbitrary objects.
                #Plus self.comm.Sendrecv requires contiguous data
                #so only in rare cases is it possible to send.
                #Hence the np.copy statement is required.
                send_data = np.array(data[send_slice], copy=True, order='C')
            if source < 0:
                recv_data = None
            else:
                #when testing the above assertion I suggest changing this
                #line to np.ones_like, rather than np.empty_like
                recv_data = np.empty_like(data[recv_slice])
            #if __debug__:
                #self.log.debug("About to sendrecv")
                #self.log.debug("data to be sent is %s"%send_data)
            self.comm.Sendrecv(
                send_data,
                dest=dest,
                recvbuf=recv_data,
                source=source
                )
            #if __debug__:
                #self.log.debug("Received data = %s"%recv_data)
                #self.log.debug("Sendrecv completed")
            if source >= 0:
                r_data += [(recv_slice, recv_data)]
        #if __debug__:
            #self.log.debug("r_data = %s"%repr(r_data))
            #self.log.debug("communication complete")
        return r_data

    def collect_data(self, data):
        #Note that this method does not take account of ghost_points in
        #the domains. This does not cause a problem. It just means
        #that more data than necessary is passed.

        #If there are no partitions of the domain we don't need to collect
        if self.comm is None:
            return data
        #Gather the data
        #if __debug__:
            #self.log.debug("Data is %s"%repr(data))
        fields = self.comm.gather(data, root = 0)
        #if __debug__:
            #self.log.debug("Data has been gathered.")
        #If this process is root then collate the data and return
        if self.comm.rank == 0:
            #if __debug__:
                #self.log.debug("The collected fields have shapes %s"%\
                    #repr([f.shape for f in fields]))
            extra_dims = len(data.shape) - len(self.domain)
            rdata_edims_shape = data.shape[:extra_dims]
            rdata_edims_slice = tuple([
                slice(None,None,None)
                for i in range(extra_dims)
                ])
            #if __debug__:
                #self.log.debug("data.shape is %s"%str(data.shape))
                #self.log.debug(
                    #"rdata_shape_edims is %s"%str(rdata_edims_slice)
                    #)
                #self.log.debug("self.domain is %s"%str(self.domain))
            rdata = np.zeros(
                rdata_edims_shape + self.domain
                )
            #if __debug__:
                #self.log.debug("rdata.shape = %s"%(repr(rdata.shape)))
            for rank, field in enumerate(fields):
                dslice = self.domain_mapping[rank]
                #if __debug__:
                    #self.log.debug("dslice is %s"%repr(dslice))
                    #self.log.debug(
                        #"rdata_edims_slice is %s"%repr(rdata_edims_slice)
                        #)
                    #self.log.debug("rdata.shape is %s"%repr(rdata.shape))
                    #self.log.debug("field.shape is %s"%repr(field.shape))
                    #self.log.debug(
                        #"rdata[rdata_edims_slice + dslice].shape is %s"
                        #%(repr(rdata[rdata_edims_slice + dslice].shape))
                        #)
                rdata[rdata_edims_slice + dslice] = field
            return rdata
        else: 
            return None

