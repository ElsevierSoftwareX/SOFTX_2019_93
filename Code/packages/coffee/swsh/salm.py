""" 
Copyright 2012
Ben Whale 
version 3 - GNU General Public License
<http://www.gnu.org/licenses/>
"""
# Standard package imports
from abc import ABCMeta, abstractmethod


class Salm(object):
    """The Abstract Base Class for representations of the spin weighted
    spherical decomposition of a function with bandwidth limit."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __repr__(self):
        """A string representation of the concrete class."""
    
    
    @abstractmethod
    def multiplication_bandlimit(self, bool):
        """Set the bandlimit behaviour for multiplication of salm objects.
        
        Arguments:
        bool - True = the minimum bandlimit of both salm objects is used.
               False = the bandlimit is the sum of the bandlimits of both
                       salm objects"""
        self.bl_mult = bool

    @abstractmethod
    def __getitem__(self, key):
        """Retrieve the coefficients of the decomposition of the function.
        
        All salm objects should be accesible as salm[s,l,m]. We leave it to
        the developers discretion if slicing, and what type of slicing, is 
        implemented.
        
        Arguments:
        (s,l,m) - integer or half-integer
        
        Returns:
        salm_coef - the coefficient of the sYlm component of the function"""
        
    def check_lm(l,m):
        """Determines in the l and m values are valid.
        
        Arguments:
        l - Integer or half-integer
        m - Integer or half-integer
        
        Returns:
        bool - If l and m are valid then return True, else return False"""
        if l > self.lmax:
            raise ValueError('l must be less than or equal to lmax')
        if math.abs(m)> l:
            raise ValueError('|m| must be less than or equal to l')

    @abstractmethod
    def __getitem__(self, key):
        pass
        
    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __add__(self,other):
        pass
 
    @abstractmethod
    def __sub__(self,other):
        pass
        
    @abstractmethod
    def __mul__(self, other):
        pass
