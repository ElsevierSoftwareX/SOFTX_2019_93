""" 
This module wraps the Huffenberger and Wandelt spinsfastpy python module. It
does this to provide an additional level of indirection to make it easier to
update the C spinsfast code and associated python module if needed.
It currently is based on revision 104
of the forward transformations
implemented in Huffenberger's & Wandelt's spinsfast code. See "Fast and exact 
spin-s Spherical Harmonic Transformations" in Astrophysical Journal Supplement 
Series (2010) 189:255-260. All references to sections, appendices or equations 
in the documentation below refer to equations given in this paper.
The spinsfast code website is
http://astrophysics.physics.fsu.edu/~huffenbe/research/spinsfast/index.html

Methods:

forward -- Returns the spin spherical harmonic coefficients, equation (6),
    using either Imm or Jmm which can be specified via keyword arguments

Copyright 2018
Ben Whale 
version 4 - GNU General Public License
<http://www.gnu.org/licenses/>
"""

import sys
import os
sys.path.insert(0, os.path.join(".", "huffenberger_wandelt", "lib"))
import spinsfast

def lm_ind(l, m, lmax=0):
    """
    Returns an array index given l and m.
    
    Arguments:
    l -- an int giving the l value of the needed coefficient
    m -- an int giving the m value of the needed coefficient
    lmax -- an int giving the maximum allowed l value
    
    Returns:
    index -- an int giving the index of the array which contains the 
             coefficient for l and m
    """
    return spinsfast.lm_ind(l, m, lmax)


def ind_lm(i, lmax=0):
    """
    Returns l and m given an array index.
    
    Binds to the method ind_lm of alm.h.
    
    Arguments:
    index -- an int giving the index of the array which contains the 
             coefficient for l and m
    lmax -- an int giving the maximum allowed l value
    
    Returns:
    [l, m] -- a list of ints giving the l and m values for the given index
    """
    return spinsfast.ind_lm(i, lmax)

def Nlm_lmax(Nlm):
    """
    Returns the maximum l value for which any (l, m) value can be stored in the
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
    lmax, _ = ind_lm(Nlm-1)
    return lmax

def lmax_Nlm(lmax):
    """
    Returns the maximum array length given the maximum l.
    
    Arguments:
    lmax -- an int giving the maximum l value
    
    Returns:
    Nlm -- an int giving the array length for alm
    """
    return spinsfast.N_lm(lmax)
