from __future__ import division
import sys
import numpy

"""
Command line inputs

sys.argv[#]
#
1 = order of derivative
2 = length of stencil
3 = location of derivative
4 = list of the value of the coordinate at
sencil points

The notation below is based on the notation used in the paper
"Generation of finite difference formulas on arbitrarily spaced grids"
by Bengt Fornberg
"""

M = int(sys.argv[1])
N = int(sys.argv[2])
x0 = float(sys.argv[3])
alphas = map(float, sys.argv[4][1:-1].split(','))
delta = numpy.zeros((M+1,N+1,N+1))

delta[0,0,0]=1
c1 = 1
for n in range(1,N+1):
    c2 = 1
    for v in range(n):
        c3 = alphas[n]-alphas[v]
        c2 = c2*c3
        delta[0,n,v] = ((alphas[n]-x0)*delta[0,n-1,v])/c3
        for m in range(1,min(n,M)+1):
            delta[m,n,v] = ((alphas[n]-x0)*delta[m,n-1,v]-m*delta[m-1,n-1,v])/c3
    delta[0,n,n] = (c1/c2)*(-(alphas[n-1]-x0)*delta[0,n-1,n-1])
    for m in range(1,min(n,M)+1):
        delta[m,n,n] = (c1/c2)*(m*delta[m-1,n-1,n-1]-\
            (alphas[n-1]-x0)*delta[m,n-1,n-1])
    c1 = c2

print "Calculation complete"
print "x0 = %f, alphas = %s"%(x0,str(alphas))
print "Coefficients"
print "------------------"
for m in range(M+1):
    for n in range(N+1):
        if not any(delta[m,n,:]):
            continue
        print "Derivative (M) = %i, Stencil Length (N) = %i, Accuracy estimate = %i"\
            %(m,n,n-m+1)
        z = zip(alphas,delta[m,n,:])
        z.sort(lambda x,y:cmp(x[0],y[0]))
        for val in z:
            if val[1] == 0:
                continue
            print "(% .2f, % .9f)"%(val)
        print "------------------"

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
