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
    
    cg_default = None
    
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
        obj = np.asarray(array).view(cls)
        obj.spins = np.atleast_1d(spins)
        obj.lmax = lmax
        obj.bl_mult = bandlimit_multiplication
        if cg is None:
            obj.cg = sfpy_salm.cg_default
        return obj
            
    def __array_finalize__(self, obj):
        if obj is None: return
        self.spins = getattr(obj, 'spins', None)
        self.lmax = getattr(obj, 'lmax', None)
        self.bl_mult = getattr(obj, 'bl_mult', None)
        self.cg = getattr(obj, "cg", None)
       
    def __str__(self):
        s = "spins = %s,\n"%repr(self.spins)
        for spin in self.spins:
            for l in range(self.lmax + 1):
                s +="(%f, %f): %s\n"%(spin, l, repr(self[spin,l].view(np.ndarray)))
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
        if self.shape[0] == 1:
            return self.view(np.ndarray)[0]
        else:
            return self.view(np.ndarray)[0]

    def multiplication_bandlimit(self, bool):
        self.bl_mult = bool
        
    def __getitem__(self, key):
        key = np.index_exp[key]
        alt_key = self.convert_key(key)
        if len(key) == 1:
            return sfpy_salm(
                np.atleast_2d(self.view(np.ndarray)[alt_key]),
                self.spins[alt_key[0]],
                self.lmax,
                self.cg,
                self.bl_mult
                )
        else:
            return self.view(np.ndarray)[alt_key]
        
    def __setitem__(self, key, value):
        key = np.index_exp[key]
        alt_key = self.convert_key(key)
        self.view(np.ndarray)[alt_key] = value
        
    def convert_key(self, key):
        spin_key = self._convert_spin_key(key[0])
        if len(key) == 1:
            r_key = spin_key,
        elif len(key) == 2:
            if key[1] > self.lmax or key[1] < 0:
                raise IndexError("order out of bounds")
            l_ind = _lm_ind(key[1],-key[1])
            order_key = slice(l_ind, l_ind + 2*key[1] +1)
            r_key = spin_key, order_key
        elif len(key) == 3:
            if abs(key[2]) > self.lmax:
                raise IndexError("degree out of bounds")
            order_key = _lm_ind(key[1],key[2])
            r_key = spin_key, order_key
        else:
            return IndexError("too many indices")
        return r_key
        
    def _convert_spin_key(self, key):
        if isinstance(key, int):
            spin_key = np.where(self.spins == key)[0][0]
        elif isinstance(key, slice):
            if key.start is None:
                start = 0
            else:
                start = np.where(self.spins == key.start)[0][0]
            if key.stop is None:
                stop = self.spins.size
            else: 
                stop = np.where(self.spins == key.stop)[0][0]
            spin_key = slice(start, stop, key.step)
        else:
            raise IndexError("spin index may only be an integer or slice")
        return spin_key
        
#    def check_lm(l,m):
#        if l > self.lmax:
#            raise ValueError('l must be less than or equal to lmax')
#        if math.abs(m)> l:
#            raise ValueError('|m| must be less than or equal to l')

    def _addsub(self, other, add):
        if not isinstance(other, sfpy_salm):
            return sfpy_salm(self.view(np.ndarray)*other, self.spins, self.lmax,
              self.cg, self.bl_mult)
        lmax = max(self.lmax, other.lmax)
        bl_mult = self.bl_mult or other.bl_mult
        cg = self.cg
        if self.cg is None and other.cg is not None:
            cg = other.cg
        spins = np.union1d(self.spins, other.spins)
        array = np.zeros((spins.shape[0], _lmax_Nlm(lmax)), dtype=np.typeDict["complex"])
        s_len = self.shape[1]
        o_len = other.shape[1]
        for i, spin in enumerate(spins):
            self_spins = self.spins==spin
            self_has_spin = any(self_spins)
            other_spins = other.spins==spin
            other_has_spin = any(other_spins)
            if self_has_spin:
                self_index = np.where(self_spins)[0][0]
                array[i][:s_len] = self.view(np.ndarray)[self_index]
                if other_has_spin:
                    other_index = np.where(other_spins)[0][0]
                    if add:
                        array[i][:o_len] += other.view(np.ndarray)[other_index]
                    else:
                        array[i][:o_len] -= other.view(np.ndarray)[other_index]
            elif other_has_spin:
                other_index = np.where(other_spins)[0][0]
                if add:
                    array[i][:o_len] += other.view(np.ndarray)[other_index]
                else:
                    array[i][:o_len] -= other.view(np.ndarray)[other_index]
            else:
                raise Exception("A spin has been encountered that is not in \
                either summand.")
        return sfpy_salm(array, spins, lmax, cg, bl_mult)


    def __add__(self,other):
        return self._addsub(other, True)

    def __sub__(self,other):
        return self._addsub(other, False)             

    def __mul__(self, other):
        if not isinstance(other, sfpy_salm):
            a = sfpy_salm(self.view(np.ndarray)*other, self.spins, self.lmax,
              self.cg, self.bl_mult)
            return a
        if self.cg is None:
            raise Exception("alm.cg must be set to a valid clebschgordan object before multiplication can be done.")
        if self.bandlimit_mult or other.bandlimit_mult:
            lmax = min(self.lmax, other.lmax)
        else:
            lmax = self.lmax + other.lmax
        self_sorted_spins = sorted(self.spins)
        other_sorted_spins = sorted(other.spins)
        min_spin = self_sorted_spins[0] + other_sorted_spins[0]
        max_spin = self_sorted_spins[-1] + other_sorted_spins[-1]
        spins = np.arange(min_spin, max_spin+1, 1)
        array = np.zeros( (spins.shape[0], _lmax_Nlm(lmax)), dtype=np.typeDict['complex'] )
        for self_s_index, self_s in enumerate(self.spins): 
            for other_s_index, other_s in enumerate(other.spins):
                s = self_s + other_s
                s_index = np.where(spins==s)[0][0]
                for k, self_salm in enumerate(self[self_s_index]):
                    for l, other_salm in enumerate(other[other_s_index]):
                        self_j, self_m = _ind_lm(k)
                        other_j, other_m = _ind_lm(l)
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
        return sfpy_salm(array, spins, lmax)

salm.Salm.register(sfpy_salm)


#def _lm_getitem(array, key):
#    if isinstance(key, int):
#        i = _lm_ind(key,-key)
#        return array[slice(i, i + 2 * key + 1)]
#    elif isinstance(key, slice):
#        r_array = []
#        start, stop, stride = _get_slice_indices(key[0])
#        for i in range(start, stop, stride):
#            r_array += _lm_getitem_(array, i)
#        return np.concatenate(r_array)
#    elif isinstance(key, tuple):
#        if len(key) == 1:
#            return _lm_getitem(array, key[0])
#        if isinstance(key[0], int):
#            if isinstance(key[1], int):
#                i = _lm_ind(*key)
#                return array[i]
#            elif isinstance(key[1], slice):
#                r_array = []
#                start, stop, stride = _get_slice_indices(key[1])
#                for i in range(start, stop, stride):
#                    r_array += array[_lm_ind(key[0], i)]
#                return np.concatenate(r_array)
#        elif isinstance(key[0], slice):
#            r_array = []
#            l_start, l_stop, l_stride = _get_slice_indices(key[0])
#            for l in range(l_start, l_stop, l_stride):
#                if isinstance(key[1], int):
#                    r_array += array[_lm_index(l, key[1])]
#                elif isinstance(key[1], slice):
#                    rm_array = []
#                    m_start, m_stop, m_stride = _get_slice_indices(key[1])
#                    for m in range(m_start, m_stop, m_stride):
#                        rm_array += array[_lm_ind(l, m)]
#                    r_array += [np.concatenate(rm_array)]
#            return np.concatenate(r_array)
#    raise ValueError("Invalid key")

#def _get_slice_indices(slice):
#    start, stop, stride = slice.indices
#    if start is None:
#        start = 0
#    if stride is None:
#        stride = 1
#    return start, stop, stride

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
    lmax = 3
    spins = np.array([-1,0,1])
    Nlmax = _lmax_Nlm(lmax)
    a = np.arange(Nlmax)
    a = np.array([a-15, a, a+15])
    salm = sfpy_salm(a, spins, lmax)
    salm[:, 1] = 2
    print salm


