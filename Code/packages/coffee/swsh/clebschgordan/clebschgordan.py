from __future__ import division
import ctypes
import logging
import math
import numpy as np
import os
import inspect
import subprocess
from numpy import ctypeslib, typeDict

from coffee.swsh import w3j

NUMERICAL_ERROR_TOLERANCE = 1e-15

def valid(j1, j2, j3, m1, m2, m3):
    return w3j.valid(j1, j2, j3, m1, m2, m3)

class CGW3j(object):
    """A class that produces Clebsch Gordan coefficients.
    
    Given an object that calculates Wigner 3j symbols this class calculates
    Clebsch Gordan coefficients. We assume that the 3j symbol values can be
    accessed as w3j(j1, j2, j3, m1, m2, m3)"""
  
    def __init__(self, w3j):
        """The construction for cg_w3j objects.
        
        Arguments:
        w3j - an object which produces wigner 3j objects. Please see the 
        documentation of the __getitem__ method for an assumption about the
        behaviour of the w3j object.
        """
        self.w3j = w3j
        
    def __call__(self, j1, m1, j2, m2, j3, m3):
        """Returns Clebsch Gordan coefficients.
        
        We perform only one consistancy check, that m3 = m1 + m2. If this
        check fails we return the value 0.0.
        
        We assume that the 3j symbols can be accessed via the code
        self.w3j(j1, j2, j3, m1, m2, m3)/
        
        Arguements:
        j1, j2, j3, m1, m2, m3 - Integer or half integer values
        
        Returns:
        Returns the value of the Clebsch Gordan coefficient 
        < j1, m1 ; j2, m2 | j3, m3 >"""
        
        w3j = self.w3j(j1, j2, j3, m1, m2, -m3)
        if w3j is 0.0:
            return w3j
        fact = math.sqrt(2*j3+1)
        if abs((j1 - j2 + m3) % 2 - 1) < NUMERICAL_ERROR_TOLERANCE: 
            fact = -fact
        rv = fact * w3j
        return rv
        
class CGStone(object):
    """Provides python access to Anthony Stone's RRF code (v 3.2), via the
    program rrfcalc. For some reason wrapping his fortran code directly with
    f2py ends with errors that are difficult to diagnose.
    """
    
    def __call__(self, j1, m1, j2, m2, j3, m3):
        """Returns the double precision value of the exact calculation of the
        Clebsch Gordan coefficient ( j1, m1; j2, m2 | j3, m3 )
                  
        Arguments:
        j1, j2, j3 - Integer or half-integer values.
        m1, m2, m3 - Integer or half-integer values so that
                     mi is one of -ji, -ji+1, ..., 0, ..., ji-1, ji
                     where i is 1, 2 or 3.
        
        Return:
        Returns the value of the Clebsch Gordan coefficient 
        ( j1, m1; j2, m2 | j3, m3 )"""
        if not w3j.valid(j1, j2, j3, m1, m2, -m3): 
            raise ValueError("j1, j2, j3 must be integers or half integers the \
            mi's must be so that mi is one of -ji, -ji+1, ..., 0, ..., ji-1, ji \
            where i is 1, 2 or 3.")
        if w3j.trivial_zero(j1, j2, j3, m1, m2, -m3):
            return 0.0
        rrfcalc = os.path.abspath(
            os.path.join(
                os.path.dirname(inspect.getfile(inspect.currentframe())),
                "..", "lib", "rrfcalc"
                )
            )
        exact = subprocess.Popen(
            [rrfcalc], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE
            )
        argument = "CG %d/2 %d/2 %d/2 %d/2 %d/2 %d/2" \
            % (2*j1, 2*j2, 2*j3, 2*m1, 2*m2, 2*m3)
        stdout, stderr = exact.communicate(argument)
        rv = eval(stdout[147:-4].replace("sqrt","math.sqrt"))
        return rv

    def exact(self, j1, m1, j2, m2, j3, m3):
        """Returns the exact value of the
        Clebsch Gordan coefficient ( j1, m1; j2, m2 | j3, m3 )
                  
        Arguments:
        j1, j2, j3 - Integer or half-integer values.
        m1, m2, m3 - Integer or half-integer values so that
                     mi is one of -ji, -ji+1, ..., 0, ..., ji-1, ji
                     where i is 1, 2 or 3.
        
        Return:
        Returns the value of the Clebsch Gordan coefficient 
        ( j1, m1; j2, m2 | j3, m3 ) The format of the string is 
        <+/-><integer>/<integer>*math.sqrt(<integer>/<integer>). The double
        precison value represented by this string can be retrived by using the
        eval() builtin."""
        if not w3j.valid(j1, j2, j3, m1, m2, m3): 
            raise ValueError("j1, j2, j3 must be integers or half integers the \
            mi's must be so that mi is one of -ji, -ji+1, ..., 0, ..., ji-1, ji \
            where i is 1, 2 or 3.")
        if w3j.trivial_zero(j1, j2, j3, m1, m2, m3):
            return 0.0
        rrfcalc = os.path.abspath(
            os.path.join(
                os.path.dirname(inspect.getfile(inspect.currentframe())),
                "..", "lib", "rrfcalc"
                )
            )
        exact = subprocess.Popen(
            [rrfcalc], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE
            )
        argument = "CG %d/2 %d/2 %d/2 %d/2 %d/2 %d/2" \
            %(2*j1, 2*j2, 2*j3, 2*m1, 2*m2, 2*m3)
        stdout = exact.communicate(argument)[0][147:-4].replace(
            "sqrt","math.sqrt"
            )
        return stdout        
        
class CGBoris(object):
    """A class that produces Clebsch Gordan coefficients.
    
    This class assumes that the object that calculates w3j symbols is of
    type wj3_boris (see the w3j module). With this information Boris's C code
    can be exploited to calculate Clebsch Gordan coefficients."""
    
    def __init__(self, w3j):
        """The constructor for cg_boris.
        
        Arguments:
        w3j - an object of class w3j_boris
        """
        self._cg = ctypes.CDLL(
        os.path.abspath(
            os.path.join(
                os.path.dirname(inspect.getfile(inspect.currentframe())),
                "..", "lib", "libboris.so.1.0.1"
                )
            )
        )
        self._cg.clebschGordanLookup.restype = ctypes.c_double
        self._cg.clebschGordanLookup.argtypes = [
            ctypeslib.ndpointer(dtype=typeDict['double'], ndim=1, \
                flags='CONTIGUOUS, WRITEABLE, ALIGNED'),
            ctypeslib.ndpointer(dtype=typeDict['double'], ndim=1, \
                flags='CONTIGUOUS, WRITEABLE, ALIGNED'),
            ctypes.c_void_p
            ]
        
    
        self._cplCw3jStruct_ptr = w3j._cplCw3jStruct_ptr
        
    def __call__(self, j1, m1, j2, m2, j3, m3):
        """Returns Clebsch Gordan coefficients.
        
        We perform only one consistancy check, that m3 = m1 + m2. If this
        check fails we return the value 0.0.
               
        Arguements:
        j1, j2, j3, m1, m2, m3 - Integer or half integer values
        
        Returns:
        Returns the value of the Clebsch Gordan coefficient 
        < j1, m1 ; j2, m2 | j3, m3 >"""
        if not w3j.valid(j1, j2, j3, m1, m2, -m3): 
            raise ValueError("j1, j2, j3 must be integers or half integers the \
            mi's must be so that mi is one of -ji, -ji+1, ..., 0, ..., ji-1, ji\
            where i is 1, 2 or 3.")
        if w3j.trivial_zero(j1, j2, j3, m1, m2, -m3):
            return 0.0
        js = np.array([j1,j2,j3], dtype = typeDict['double'])
        ms = np.array([m1,m2,m3], dtype = typeDict['double'])
        rv = self._cg.clebschGordanLookup(js, ms, self._cplCw3jStruct_ptr)
        return rv
        
if __name__=="__main__":
    import sys
#    cgStone = CGStone()
#    j1 = 3/2
#    j2 = 0 
#    j3 = 3/2
#    m1 = -1/2
#    m2 = 0
#    m3 = -1/2
#    print cgStone(j1, m1, j2, m2, j3, m3)
#    wBen = w3j.W3jBen("delete.me")
#    cgBenw3j = CGW3j(wBen)
#    print cgBenw3j(j1, m1, j2, m2, j3, m3)
#    sys.exit(0)
    from swsh import w3j
    x = 2
    NUMERICAL_ERROR_TOLERANCE = 1e-15
    wBen = w3j.W3jBen("delete.me")
    wBoris = w3j.W3jBoris(10,0.5)
    wBoris.calculate()
    cgBenw3j = CGW3j(wBen)
    cgStone = CGStone()
    cgBorisw3j = CGW3j(wBoris)
    cgBoris = CGBoris(wBoris)
    ms = np.array([(u,v,w) 
        for u in np.arange(-x,x,0.5) 
        for v in np.arange(-x,x,0.5) 
        for w in np.arange(-x,x,0.5)])
    js = np.array([(u,v) 
        for u in np.arange(0,x,0.5) 
        for v in np.arange(0,x,0.5)])
    print "Objects created"
    print "Test commensing"
    count_range = range(0, 16 * x + 1, 2 * x)
    for j1 in np.arange(0,x,0.5):
        print "\nj1 is %f"%j1
        for j2, j3 in js:
            for m1, m2, m3 in ms:
                sys.stdout.write('.')
                if w3j.valid(j1, j2, j3, m1, m2, -m3):
                    #print "collecting 3j values"
                    a = cgStone(j1, m1, j2, m2, j3, m3)
                    b = cgBenw3j(j1, m1, j2, m2, j3, m3)
                    c = cgBorisw3j(j1, m1, j2, m2, j3, m3)
                    d = cgBoris(j1, m1, j2, m2, j3, m3)
                    if (abs(a-b)< NUMERICAL_ERROR_TOLERANCE 
                        and abs(b-c) < NUMERICAL_ERROR_TOLERANCE 
                        and abs(a-c) < NUMERICAL_ERROR_TOLERANCE
                        and abs(a-d) < NUMERICAL_ERROR_TOLERANCE
                        and abs(b-d) < NUMERICAL_ERROR_TOLERANCE
                        and abs(c-d) < NUMERICAL_ERROR_TOLERANCE):
                        continue
                    else:
                        print "\nFail: %f, %f, %f, %f, %f, %f" \
                            %(j1, j2, j3, m1, m2, m3)
                        print "abs(cgStonew3j - cgBenw3j) = %.15f" \
                            %(abs(a-b))
                        print "abs(cgStonew3j - cgBorisw3j) = %.15f" \
                            %(abs(a-c))
                        print "abs(cgBenw3j - cgBorisw3j) = %.15f" \
                            %(abs(b-c))
                        print "abs(cgBoris - cgBorisw3j) = %.15f" \
                            %(abs(d-c))
                        print "abs(cgBoris - cgBenw3j) = %.15f" \
                            %(abs(d-b))
                        print "abs(cgBoris - cgStonew3j) = %.15f" \
                            %(abs(d-a))
                sys.stdout.flush()
    print "\nTest completed"
