""" 
This module is part of spinsfastpy. Please see the package documentation. It
contains an object which represents the spherical harmonic coefficients of a 
function defined on S^2. Internally it provides python bindings for revision 90 
of the methods listed in alm.h from Huffenberger's & Wandelt's spinsfast code. 
See "Fast and exact spin-s Spherical Harmonic Transformations" in Astrophysical 
Journal Supplement Series (2010) 189:255-260. All references to sections, 
appendices or equations in the documentation below refer to equations given in 
this paper.

The object is a subclass of numpy.ndarray which contains the spin spherical 
harmonic coefficients along with some meta data.

The module also contains bindings to the methods given in the file alm.h 
contained in Huffenberger's & Wandelt's spinsfast code.

Copyright 2012
Ben Whale 
version 3 - GNU General Public License
<http://www.gnu.org/licenses/>
"""
# future imports
from __future__ import division

# Standard libraries
import ctypes
import numpy as np
import math
import os
import inspect
from abc import ABCMeta

# Package internal imports
from coffee.swsh import w3j
from coffee.swsh import clebschgordan as cg_mod
from coffee.swsh import salm

# load a shared object library version of spinsfast
sf = ctypes.CDLL(
         os.path.abspath(
             os.path.join(
                 os.path.dirname(inspect.getfile(inspect.currentframe())),
                 "..", "lib", "libspinsfast.so.1"
                 )
             )
         )

class sfpy_salm(np.ndarray):
    
    """
    Represents the spin spherical harmonic coefficients of a function defined on
    S^2.
    
    The coefficient of sYlm is accessed as sfpy_salm[s,l,m]. Basic slicing 
    is implemented.
    """
    
    cg_default = cg_mod.CGStone()
    
    def __new__(cls, array, spins, lmax, cg=None, bandlimit_multiplication=False):
        """
        Returns an instance of alm.salm.
        
        While it is possible to construct instances of this class directly, see
        the file example_mulispin.c of spinsfast, the more usual construction
        will be as the object returned by spinsfastpy.forward.forward.
        
        Arguments:
        array -- the alm array with the structure described in the doc string
                 for the class.
        spins -- an array of the spin values. It must be the case that 
                 len(spins) == len(array[:,0]). This is not checked.
                 
        Returns:
        salm -- an instance of class alm.salm
        """
        # no error checking is currently done to test if s>=l. Be warned.
        spins = np.asarray(spins)
        if len(array.shape) == 1:
            if spins.shape is () and array.shape[0] == _lmax_Nlm(lmax):
                obj = np.asarray(array).view(cls)
            else:
                raise ValueError("Mal-formed array and shape for lmax")
        elif len(array.shape) == 2:
            if spins.shape[0] == array.shape[0] \
                and array.shape[1] == _lmax_Nlm(lmax):
                obj = np.asarray(array).view(cls)
            else:
                raise ValueError("Mal-formed array and shape for lmax")
        else:
            raise ValueError("Array shape has too many dimensions")
        obj.spins = spins
        obj.lmax = lmax
        obj.bl_mult = bandlimit_multiplication
        #cg should never be None
        if cg is None:
            obj.cg = sfpy_salm.cg_default
        else:
            obj.cg = cg
        return obj
            
    def __array_finalize__(self, obj):
        if obj is None: return
        self.spins = getattr(obj, 'spins', None)
        self.lmax = getattr(obj, 'lmax', None)
        self.bl_mult = getattr(obj, 'bl_mult', None)
        self.cg = getattr(obj, "cg", sfpy_salm.cg_default)
       
    def __str__(self):
        s = "spins = %s,\n"%repr(self.spins)
        if self.spins.shape is ():
            for l in range(self.lmax + 1):
                s +="(%f, %f): %s\n"%(
                    self.spins, 
                    l, 
                    repr(self[l].view(np.ndarray))
                    )
        else:
            for spin in self.spins:
                for l in range(self.lmax + 1):
                    s +="(%f, %f): %s\n"%(
                        spin, 
                        l, 
                        repr(self[spin,l].view(np.ndarray))
                        )
        return s 
       
    def __repr__(self):
        s = "sfpy_salm object:\nlmax = %d,\nspins = %s,\nbandlimit_multiplication = %s,\ncg = %s,\ncoefficients = \n%s"%(
            self.lmax,
            repr(self.spins),
            repr(self.bl_mult),
            repr(self.cg),
            repr(self.view(np.ndarray))
            )
        return s

    @property
    def coefs(self):
        return self.view(np.ndarray)[0]

    def multiplication_bandlimit(self, bool):
        self.bl_mult = bool

    def __getslice__(self, start, stop):
        """Thiss solves a subtle bug, where __getitem__ is not called, and all
        the dimensional checking not done, when a slice of only the first
        dimension is taken, e.g. a[1:3]. From the Python docs:
        Deprecated since version 2.0: Support slice objects as parameters
        to the __getitem__() method. (However, built-in types in CPython
        currently still implement __getslice__(). Therefore, you have to
        override it in derived classes when implementing slicing.)
        """  
        return self.__getitem__(slice(start, stop))

    def __getitem__(self, key):
        alt_key = self.convert_key(key)
        if len(alt_key) == 1:
            if self.spins.shape is ():
                return self.view(np.ndarray)[alt_key]
            else:
                return sfpy_salm(
                    self.view(np.ndarray)[alt_key],
                    self.spins[alt_key],
                    self.lmax,
                    self.cg,
                    self.bl_mult
                    )
        else:
            return self.view(np.ndarray)[alt_key]
        
    def __setitem__(self, key, value):
        alt_key = self.convert_key(key)
        self.view(np.ndarray)[alt_key] = value
        
    def convert_key(self, key):
        # convert ket into a tuple of ints and slices
        key = np.index_exp[key]

        # perform the transformation from salm to indices
        if len(key) == 1:
            if self.spins.shape is ():
                return _convert_lm_key(key, self.lmax),
            else:
                return _convert_spin_key(key[0], self.spins),
        elif len(key) >= 2:
            if self.spins.shape is ():
                return _convert_lm_key(key, self.lmax),

            else:
                spin_key = _convert_spin_key(key[0], self.spins)
                order_key = _convert_lm_key(key[1:], self.lmax)
                return spin_key, order_key
        else:
            raise IndexError("Too many indices")

    def _addsub(self, other, add):
        #check object types and delegate addition if not sfpy_salm
        if not isinstance(other, sfpy_salm):
            if add:
                return sfpy_salm(
                    self.view(np.ndarray) + other, 
                    self.spins, 
                    self.lmax,
                    self.cg, 
                    self.bl_mult
                    )
            else:
                return sfpy_salm(
                    self.view(np.ndarray) - other, 
                    self.spins, 
                    self.lmax,
                    self.cg, 
                    self.bl_mult
                    )

        #Perform set up
        #Remember that we have three cases to deal with: 
        # 1) other.spins == () and self.spins == () 
        # 2) other.spins == () and self.spins is not ()
        # 3) other.spins is not () and self.spins is not ()
        # I deal with these cases by converting everything to
        # arrays and then formatting the output appropriately.
        lmax = max(self.lmax, other.lmax)
        bl_mult = self.bl_mult or other.bl_mult
        cg = self.cg
        if self.cg is None and other.cg is not None:
            cg = other.cg
        s_spins = self.spins
        s_array = np.asarray(self)
        if self.spins.shape == ():
            s_spins = np.atleast_1d(s_spins)
            s_array = np.atleast_2d(s_array)
        o_spins = other.spins
        o_array = np.asarray(other)
        if other.spins.shape is ():
            o_spins = np.atleast_1d(o_spins)
            o_array = np.atleast_2d(o_array)
        spins = np.union1d(s_spins, o_spins)
        if self.dtype <= other.dtype:
            dtype = other.dtype
        elif self.dtype > other.dtype:
            dtype = self.dtype
        else:
            raise ValueError("Unable to infer the dtype of the result of\
            addition")
        array = np.zeros(
            (spins.shape[0], _lmax_Nlm(lmax)), 
            dtype=dtype
            )
        s_len = _lmax_Nlm(self.lmax)
        o_len = _lmax_Nlm(other.lmax)

        #Do computation
        for i, spin in enumerate(spins):
            s_spins_select = s_spins==spin
            self_has_spin = any(s_spins_select)
            o_spins_select = o_spins==spin
            other_has_spin = any(o_spins_select)
            if self_has_spin:
                self_index = np.where(s_spins_select)[0][0]
                array[i][:s_len] = s_array[self_index]
                if other_has_spin:
                    other_index = np.where(o_spins_select)[0][0]
                    if add:
                        array[i][:o_len] += o_array[other_index]
                    else:
                        array[i][:o_len] -= o_array[other_index]
            elif other_has_spin:
                other_index = np.where(o_spins_select)[0][0]
                if add:
                    array[i][:o_len] += o_array[other_index]
                else:
                    array[i][:o_len] -= o_array[other_index]
            else:
                raise Exception("A spin has been encountered that is not in \
                either summand.")

        # Now we need to appropriately format the output
        if self.spins.shape is () and other.spins.shape is () \
            and self.spins == other.spins:
            spins = spins[0]
            array = array[0]    
        return sfpy_salm(array, spins, lmax, cg, bl_mult)


    def __add__(self,other):
        return self._addsub(other, True)

    def __sub__(self,other):
        return self._addsub(other, False)             

    def __rmul__(self, other):
        return sfpy_salm(
            other * self.view(np.ndarray),
            self.spins, 
            self.lmax,
            self.cg, 
            self.bl_mult
            )

    def __mul__(self, other):
        # Do we need to treat this multiplication as between salm objects?
        if not isinstance(other, sfpy_salm):
            a = sfpy_salm(
                self.view(np.ndarray) * other, 
                self.spins, 
                self.lmax,
                self.cg, 
                self.bl_mult
                )
            return a
        # Set up work
        # Transform everything to arrays
        # Then format back when returning the result
        bl_mult = self.bl_mult or other.bl_mult
        cg = self.cg
        if self.cg is None and other.cg is not None:
            cg = other.cg
        if self.cg is None:
            raise Exception("alm.cg must be set to a valid clebschgordan object before multiplication can be done.")
        if bl_mult:
            lmax = min(self.lmax, other.lmax)
        else:
            lmax = self.lmax + other.lmax
        s_spins = self.spins
        s_array = np.asarray(self)
        if self.spins.shape == ():
            s_spins = np.atleast_1d(s_spins)
            s_array = np.atleast_2d(s_array)
        self_sorted_spins = sorted(s_spins)
        o_spins = other.spins
        o_array = np.asarray(other)
        if other.spins.shape is ():
            o_spins = np.atleast_1d(o_spins)
            o_array = np.atleast_2d(o_array)
        other_sorted_spins = sorted(o_spins)
        min_spin = self_sorted_spins[0] + other_sorted_spins[0]
        max_spin = self_sorted_spins[-1] + other_sorted_spins[-1]
        spins = np.arange(min_spin, max_spin+1, 1)
        if self.dtype <= other.dtype:
            dtype = other.dtype
        elif self.dtype > other.dtype:
            dtype = self.dtype
        else:
            raise ValueError("Unable to infer the dtype of the result of\
            multiplication")
        array = np.zeros( 
            (spins.shape[0], _lmax_Nlm(lmax)), 
            dtype=dtype 
            )
        # Do the calculation already!
        for self_s_index, self_s in enumerate(s_spins): 
            for other_s_index, other_s in enumerate(o_spins):
                s = self_s + other_s
                s_index = np.where(spins==s)[0][0]
                for k, self_salm in enumerate(s_array[self_s_index]):
                    self_j, self_m = _ind_lm(k)
                    if self_j < abs(self_s):
                        continue
                    for l, other_salm in enumerate(o_array[other_s_index]):
                        other_j, other_m = _ind_lm(l)
                        if other_j < abs(other_s):
                            continue
                        m = self_m + other_m
                        jmin = max(
                            abs(self_j - other_j), 
                            abs(self_s + other_s),
                            abs(self_m + other_m)
                            )
                        jmax = min(self_j + other_j,lmax)
                        js = np.arange(jmin, jmax+1, 1)
                        first_f = 0.5 * math.sqrt(
                            (2 * self_j + 1) * (2 * other_j + 1) / math.pi
                            )
                        for j in js:
                            second_f = 1 / math.sqrt(2 * j +1)
                            cg_fact = self.cg(
                                self_j, self_m, other_j, other_m, j, m
                                ) \
                                * \
                                self.cg(
                                    self_j, -self_s, other_j, -other_s, j, -s
                                )
                            jm_index = _lm_ind(j, m)
#                            print "(j,m) = (%d, %d)"%(j,m)
#                            print "(s_index, jm_index) = (%d, %d)"%(s_index,jm_index)
#                            print "value is %f"%(first_f * second_f \
#                                * cg_fact * self_salm * other_salm)
                            # Change this calculation to avoid this sum...
                            array[s_index, jm_index] = array[s_index, jm_index] +\
                                first_f * second_f \
                                * cg_fact * self_salm * other_salm
#                            print "array is now %s"%repr(array)
        if self.spins.shape is () and other.spins.shape is ():
            array = array[0]
            spins = spins[0]
        return sfpy_salm(
            array, 
            spins, 
            lmax, 
            cg=self.cg, 
            bandlimit_multiplication=bl_mult
            )

salm.Salm.register(sfpy_salm)

def _convert_lm_key(key, lmax):
    if isinstance(key, int):
        l_ind = _lm_ind(key[0],-key[0])
        return slice(l_ind, l_ind + 2*key[0] +1)
    elif isinstance(key, slice):
        raise IndexError("The order index cannot be a slice")
    else:
        if isinstance(key[0], slice):
            raise IndexError("The order index cannot be a slice")
        elif key[0] <= lmax and key[0] >= 0:
            if len(key) == 1:
                l_ind = _lm_ind(key[0],-key[0])
                return slice(l_ind, l_ind + 2*key[0] +1)
            elif len(key) == 2:
                if isinstance(key[1], slice):
                    raise IndexError("The degree index cannot be a slice")
                elif abs(key[1]) <= key[0]:
                    return _lm_ind(key[0],key[1])
                else:
                    raise IndexError("degree out of bounds")
        else:
            raise IndexError("order out of bounds")

def _convert_spin_key(key, spins):
    if isinstance(key, int):
        try:
            spin_key = np.where(spins == key)[0][0]
        except IndexError:
            raise IndexError("Spins %d not found in spins"%key)
    elif isinstance(key, slice):
        if key.start is None:
            start = 0
        else:
            start = np.where(spins == key.start)[0][0]
        if key.stop is None:
            stop = spins.size
        else: 
            stop = np.where(spins == key.stop)[0][0]
        spin_key = slice(start, stop, key.step)
    else:
        raise IndexError("spin index may only be an integer or slice")
    return spin_key

class sfpy_sralm(np.ndarray):

    """
    Represents the spin spherical harmonic coefficients of a function defined on
    [a,b] x S^2. It is assume that this array will be stored in an array
    corresponding to spins. Thus, while this object knows it's own spin
    it is not possible for this object to store multiple spins.
    
    The coefficient of sYlm is accessed as sfpy_salm[:,l,m]. Basic slicing 
    is implemented.
    """
    
    cg_default = None
    
    def __new__(cls, array, spins, lmax, cg=None, bandlimit_multiplication=False):
        """
        Returns an instance of alm.sralm.

        The array must have the following structure: len(array.shape)=2 or 3,
        If .... balch
        array.shape[0] = number of spins
        array.shape[1] = number of grid points in the interval
        array.shape[2] = salm.__lmax_Nlm(lmax)
        
        Arguments:
        array -- the alm array with the structure described in the doc string
                 for the class.
        spinis -- an array of ints giving the spin values.
                 
        Returns:
        sfpy_sralm -- an instance of class sfpy_sralm
        """
        spins = np.asarray(spins)
        if len(array.shape) == 2:
            if spins.shape is () and array.shape[1] == _lmax_Nlm(lmax):
                obj = np.asarray(array).view(cls)
            else:
                raise ValueError("Mal-formed array and shape for lmax")
        elif len(array.shape) == 3:
            if spins.shape[0] == array.shape[0] and \
                array.shape[2] == _lmax_Nlm(lmax):
                obj = np.asarray(array).view(cls)
            else:
                raise ValueError("Mal-formed array and shape for lmax")
        else:
            raise ValueError("Array shape has too many or too few dimensions")
        obj.spins = spins
        obj.lmax = lmax
        obj.bl_mult = bandlimit_multiplication
        # cg should never be none
        if cg is None:
            obj.cg = sfpy_salm.cg_default
        return obj
            
    def __array_finalize__(self, obj):
        if obj is None: return
        self.spins = getattr(obj, 'spins', None)
        self.lmax = getattr(obj, 'lmax', None)
        self.bl_mult = getattr(obj, 'bl_mult', None)
        self.cg = getattr(obj, "cg", sfpy_salm.cg_default)
       
    def __str__(self):
        s = "spins = %s,\n"%repr(self.spins)
        s += "lmax = %d,\n"%self.lmax
        if self.spins.shape is ():
            #for i in range(self.shape[0]):
            for i in [0,1,2,self.shape[0]-3, self.shape[0]-2, self.shape[0]-1]:
                for l in range(self.lmax + 1):
                    s +="(%.1f, %.1f) at %d r value: %s\n"%(
                        int(self.spins), 
                        l, 
                        i,
                        repr(self[i,l])
                        )
        else:
            for j in self.spins:
                for i in range(self.shape[1]):
                    for l in range(self.lmax + 1):
                        s +="(%.1f, %.1f) at %d r value: %s\n"%(
                            j, 
                            l, 
                            i,
                            repr(self[j,i,l])
                            )
        return s 
       
    def __repr__(self):
        s = "<sfpy_sralm lmax = %d, spins = %s, bandlimit_multiplication =\
        %s, cg = %s, coefficients = %s>"%(
            self.lmax,
            repr(self.spins),
            repr(self.bl_mult),
            repr(self.cg),
            repr(self.view(np.ndarray))
            )
        return s

    def __getslice__(self, start, stop):
        """Thiss solves a subtle bug, where __getitem__ is not called, and all
        the dimensional checking not done, when a slice of only the first
        dimension is taken, e.g. a[1:3]. From the Python docs:
        Deprecated since version 2.0: Support slice objects as parameters
        to the __getitem__() method. (However, built-in types in CPython
        currently still implement __getslice__(). Therefore, you have to
        override it in derived classes when implementing slicing.)
        """  
        return self.__getitem__(slice(start, stop))

    def __getitem__(self, key):
        #import pdb; pdb.set_trace()
        key = self.convert_key(key)
        if self.spins.shape is ():
            return self._getitem_no_spin(key)
        else:
            return self._getitem_spin(key)

    def _getitem_no_spin(self, key):
        rv = np.asarray(self)[key]
        if len(key) == 1:
            if len(rv.shape) == 2:
                if rv.shape[0] == 1:
                    return sfpy_salm(
                        rv[0],
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
                else:
                    return sfpy_sralm(
                        rv,
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
            else:
                return sfpy_salm(
                    rv,
                    self.spins,
                    self.lmax,
                    self.cg,
                    self.bl_mult
                    )
        elif len(key) == 2:
            if not isinstance(rv, np.ndarray) or len(rv.shape) < 2:
                return rv
            else:
                if rv.shape[0] == 1:
                    return sfpy_salm(
                        rv[0],
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
                else:
                    return sfpy_sralm(
                        rv,
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
        raise IndexError("Unable to process selection")

    def _getitem_spin(self, key):
        rv = np.asarray(self)[key]
        rs = self.spins[key[0]]
        if len(key) < 2:
            return sfpy_sralm(
                rv,
                rs,
                self.lmax,
                self.cg,
                self.bl_mult
                )
        elif len(key) == 3:
            return rv  
        elif len(key) == 2:
            # This is the awkward case. Did we get rid of s or r dependence?
            # note that r dependence can be removed in two ways, either the
            # rv array has reduced shape length or length of the dimension
            # representing r is 1.
            if rs.shape is ():
                #spins has disapeared we know that len(rv.shape) <=2
                if len(rv.shape) < 2:
                    #since the key is selecting for only s and r we know that
                    # rv must contain all l values for the selected s and r
                    # this is an salm object
                    return sfpy_salm(
                        rv,
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
                else:
                    #we need to test to see if r dependence has been removed
                    if rv.shape[0] == 1:
                        return sfpy_salm(
                            rv[0],
                            self.spins,
                            self.lmax,
                            self.cg,
                            self.bl_mult
                            )
                    else:
                        return sfpy_sralm(
                            rv,
                            rs,
                            self.lmax,
                            self.cg,
                            self.bl_mult
                            )
            else:
                #spin dependence still exists
                if len(rv.shape) < 2:
                    #the r dependence has been removed
                    return sfpy_salm(
                        rv,
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
                elif rv.shape[1] == 1:
                    return sfpy_salm(
                        rv[:,0,:],
                        self.spins,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
                else:
                    return sfpy_sralm(
                        rv,
                        rs,
                        self.lmax,
                        self.cg,
                        self.bl_mult
                        )
        # if nothing was selected return an error
        raise IndexError("Unable to process selection")

    def convert_key(self, key):
        key = np.index_exp[key]
        if len(key) == 1:
            if self.spins.shape is ():
                return key
            return _convert_spin_key(key[0], self.spins),
        elif len(key) == 2:
            if self.spins.shape is ():
                return key[0], _convert_lm_key(key[1:], self.lmax)
            return _convert_spin_key(key[0], self.spins), key[1] 
        elif len(key) >= 3:
            if self.spins.shape is ():
                return key[0], _convert_lm_key(key[1:], self.lmax)
            else:
                return (
                    _convert_spin_key(key[0], self.spins), 
                    key[1],
                    _convert_lm_key(key[2:], self.lmax)
                    )
        else:
            return IndexError("too many indices")

    def __add__(self, other):
        if isinstance(other, sfpy_sralm):
            if self.spins.shape == ():
                rv_temp = self[0] + other[0]
                rv = np.empty(
                    (self.shape[0],) + (rv_temp.shape[0],),
                    dtype = self.dtype
                    )
                rv[0] = rv_temp
                for i in range(1, rv.shape[0]):
                    rv[i] = self[i] + other[i]
            else:
                rv_temp = self[:,0] + other[:, 0]
                rv = np.empty(
                    (rv_temp.shape[0],) + (self.shape[1],) + (rv_temp.shape[1],),
                    dtype = self.dtype
                    )
                rv[:, 0] = rv_temp
                for i in range(1, rv.shape[0]):
                    rv[:, i] = self[:, i] + other[:, i]
            return sfpy_sralm(
                rv,
                rv_temp.spins,
                rv_temp.lmax,
                cg=rv_temp.cg,
                bandlimit_multiplication=rv_temp.bl_mult
                )
        else:
            return sfpy_sralm(
                self.view(np.ndarray) + other,
                self.spins,
                self.lmax,
                cg=self.cg,
                bandlimit_multiplication=self.bl_mult
                )

    def __radd__(self, other):
        """Note that no checking for the correct spin is performed."""
        return sfpy_sralm(
                other + np.asarray(self), 
                self.spins,
                self.lmax,
                cg=self.cg,
                bandlimit_multiplication=self.bl_mult
                )

    def __sub__(self, other):
        if isinstance(other, sfpy_sralm):
            if self.spins.shape == ():
                rv_temp = self[0] - other[0]
                rv = np.empty(
                    (self.shape[0],) + (rv_temp.shape[0],),
                    dtype = self.dtype
                    )
                rv[0] = rv_temp
                for i in range(1, rv.shape[0]):
                    rv[i] = self[i] - other[i]
            else:
                rv_temp = self[:,0] - other[:, 0]
                rv = np.empty(
                    (rv_temp.shape[0],) + (self.shape[1],) + (rv_temp.shape[1],),
                    dtype = self.dtype
                    )
                rv[:, 0] = rv_temp
                for i in range(1, rv.shape[0]):
                    rv[:, i] = self[:, i] - other[:, i]
            return sfpy_sralm(
                rv,
                rv_temp.spins,
                rv_temp.lmax,
                cg=rv_temp.cg,
                bandlimit_multiplication=rv_temp.bl_mult
                )
        else:
            return sfpy_sralm(
                self.view(np.ndarray) - other,
                self.spins,
                self.lmax,
                cg=self.cg,
                bandlimit_multiplication=self.bl_mult
                )

    def __rsub__(self, other):
        """Note that no checking for the correct spin is performed."""
        return sfpy_sralm(
                other - np.asarray(self), 
                self.spins,
                self.lmax,
                cg=self.cg,
                bandlimit_multiplication=self.bl_mult
                )

    def __mul__(self, other):
        if isinstance(other, sfpy_sralm):
            rv_temp = self[:,0] * other[:, 0]
            rv = np.empty(
                (rv_temp.shape[0],) + (self.shape[1],) + (rv_temp.shape[1],),
                dtype = self.dtype
                )
            rv[:, 0] = rv_temp
            for i in range(1, rv.shape[0]):
                rv[:, i] = self[:, i] * other[:, i]
            return sfpy_sralm(
                np.ndarray(rv),
                rv_temp.spins,
                rv_temp.lmax,
                cg=rv_temp.cg,
                bandlimit_multiplication=rv_temp.bl_mult
                )
        else:
            return sfpy_sralm(
                self.view(np.ndarray) * other,
                self.spins,
                self.lmax,
                cg=self.cg,
                bandlimit_multiplication=self.bl_mult
                )

    def __rmul__(self, other):
        return sfpy_sralm(
            other * self.view(np.ndarray),
            self.spins,
            self.lmax,
            cg=self.cg,
            bandlimit_multiplication=self.bl_mult
            )


salm.Salm.register(sfpy_sralm)

def _Nlm_lmax(Nlm):
    """
    Returns the maximum l value for which any (l,m) value can be stored in the
    array.
    
    Some care is required with this as the method makes no guarantee that
    all |m|<=l values can be stored in the array, only that one (in this case
    m=-l), can be stored in the array.
    
    Arguments:
    Nlm -- the length of the array
    
    Returns:
    lmax -- the maximum l for which at least one m value can be stored in the
            array
    """
    lmax, m = _ind_lm(Nlm-1)
    return lmax

def _lm_ind(l,m):
    """
    Returns an array index given l and m.
    
    Binds to the method lm_ind of alm.h.
    
    Arguments:
    l -- an int giving the l value of the needed coefficient
    m -- an int giving the m value of the needed coefficient
    
    Returns:
    index -- an int giving the index of the array which contains the 
             coefficient for l and m
    """
    return sf.lm_ind(int(l),m,0)

def _ind_lm(i):
    """
    Returns l and m given an array index.
    
    Binds to the method ind_lm of alm.h.
    
    Arguments:
    index -- an int giving the index of the array which contains the 
             coefficient for l and m
    
    Returns:
    (l, m) -- a tuple of ints giving the l and m values for the given index
    """
    a = ctypes.c_int()
    b = ctypes.c_int()
    sf.ind_lm(i, ctypes.byref(a), ctypes.byref(b), 0)
    return a.value, b.value
   
def _lmax_Nlm(lmax):
    """
    Returns the maximum array length given the maximum l.
    
    Binds to the method N_lm of alm.h.
    
    Arguments:
    lmax -- an int giving the maximum l value
    
    Returns:
    Nlm -- an int giving the array length for alm
    """
    return sf.N_lm(lmax)
    
if __name__ == "__main__":
    lmax = 2
    a = np.ones((2,_lmax_Nlm(lmax)))
    spins = 2
    p = sfpy_sralm(a, spins, lmax)
    p[-1]
    print "\n\n"
    p[-1:]
    print "\n\n"
    p[slice(-1,None,None)]
    #from coffee.swsh import clebschgordan as cg
    #cg_object = cg.CGStone()

    #lmax = 3
    #spins = -1
    #Nlmax = _lmax_Nlm(lmax)
    #a = np.arange(Nlmax)
    #a[:] =1 
    ##a = np.array([a])
    #a_salm = sfpy_salm(a, spins, lmax, cg=cg_object)
    #print "a info"
    #print a_salm
    #print a_salm.spins
    #print a_salm.shape

    #lmax = 3
    #spins = -1
    #Nlmax = _lmax_Nlm(lmax)
    #b = np.arange(Nlmax)
    #b = np.array(b)
    #b_salm = sfpy_salm(b, spins, lmax, cg=cg_object)
    #print "\n\nb info"
    #print b_salm
    #print b_salm.spins
    #print b_salm.shape
 
    #c_salm = a_salm * b_salm
    #print "\n\nc info"
    #print c_salm
    #print c_salm.spins
    #print c_salm.shape

    #print "\n\n"
    #print c_salm[5,-2]
