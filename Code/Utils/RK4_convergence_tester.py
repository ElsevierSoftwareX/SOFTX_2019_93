from __future__ import division
import numpy
import numpy as np
import Gnuplot

def sin(x,t):
    return np.sin(x+t)
    
def dsin(x,t):
    return np.cos(x+t)

def line(x,t):
    return 2*(x+t)

def dline(x,t):
    return 2

def poly_bump(x,t):
    val = 2-(x+t)**5
    return val

def dpoly_bump(x,t):
    val = -5*(x+t)**4
    return val

def exp_bump(x,t):
    return numpy.exp(-20*(x+t)**2)

def dexp_bump(x,t):
    return -40*(x+t)*exp_bump(x,t)
    
def smooth_non_analytic(x):
    def f(y): 
        if y<=0.0:
            return 0.0
        else:
            return np.exp(-1/y)
    return np.vectorize(f)(x)

def dsmooth_non_analytic(x):
    def f(y): 
        if y<=0.0:
            return 0.0
        else:
            return np.exp(-1/y)/(y**2)
    return np.vectorize(f)(x)

def dsmooth_step(x, start, stop):
    y = (x-start)/(stop-start)
    first_term = (dsmooth_non_analytic(y)) /\
        (smooth_non_analytic(y)+smooth_non_analytic(1-y))
    second_term = -smooth_non_analytic(y)*\
        (dsmooth_non_analytic(y)-dsmooth_non_analytic(1-y))/\
        (smooth_non_analytic(y)+smooth_non_analytic(1-y))**2
    return (first_term+second_term)/(stop-start)
        
def smooth_step(x, start, stop):
    y = (x-start)/(stop-start)
    return (smooth_non_analytic(y))/ \
        (smooth_non_analytic(y)+\
                smooth_non_analytic(1-y))

def smooth_bump(x, start_up, stop_up,start_down, stop_down):
    return smooth_step(x, start_up, stop_up)*\
        (1-smooth_step(x, start_down, stop_down))

def dsmooth_bump(x, start_up, stop_up,start_down, stop_down):
    return dsmooth_step(x, start_up, stop_up)*\
        (1-smooth_step(x, start_down, stop_down))-\
        smooth_step(x, start_up, stop_up)*\
        dsmooth_step(x, start_down, stop_down)

def RK4(domain,dfunc,val,t,dt):
    k1 = dfunc(t,domain)
    k2 = dfunc(t+0.5*dt,domain+0.5*dt*k1)
    k3 = dfunc(t+0.5*dt,domain+0.5*dt*k2)
    k4 = dfunc(t+dt,domain+dt*k3)
    return val+(1/6)*dt*(k1+2*k2+2*k3+k4)    

dts = []
derivative_error = []
for i in range (4):
    domain = np.linspace(-1,1,101)
    dt = 0.01/(2**i)
    dts += [dt]
    numerical = poly_bump(domain,0)
    for x in range(int(1/dt)+1):
        numerical = RK4(domain,dpoly_bump,numerical,x*dt,dt)
    analytic = poly_bump(domain,int(1/dt)*dt)
    derivative_error += [(numpy.abs(numerical-analytic),domain)]

g = Gnuplot.Gnuplot()
g('set style data lines')
plots = []
scale = 0
for i,data in enumerate(derivative_error):
    plots += [Gnuplot.Data(data[1],numpy.log2(data[0])+scale*i,\
        title = "dt = %f"%dts[i])]
g.title("Error in spatial derivative with +%i*i scaling"%scale)
g('set terminal gif')
g('set out "test.gif"')
g('set ylabel "Log2 of absolute value of error with +%i*i scaling"'%scale)
#g('set yrange [-10:-0]')
#g('set xrange [-0.91:-0.09]')
g.plot(*plots)
raw_input("press key to finish")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
