import sys
sys.path.append("../bin")
import numpy
import h5py_array
import datagroup_utils

sys.argv[1] = 'advection_periodic_boundary31-1-2011.hdf'
sys.argv[2] = '2.0'

def comparor(x, y):
    if round(x, 12)== round(y, 12):
        return True
    return False

def smooth_non_analytic(x):
    def f(y): 
        if y<=0.0:
            return 0.0
        else:
            return numpy.exp(-1/y)
    return numpy.vectorize(f)(x)

def smooth_step(x, start, stop):
    y = (x-start)/(stop-start)
    return (smooth_non_analytic(y))/ \
        (smooth_non_analytic(y)+\
                smooth_non_analytic(1-y))

def smooth_bump(x, start_up, stop_up,start_down, stop_down):
    return smooth_step(x, start_up, stop_up)*\
        (1-smooth_step(x, start_down, stop_down))

def exact_solution(domain):
    return smooth_bump(domain,-0.25,-0.1 ,0.1, 0.25)

with h5py_array.H5pyArray(sys.argv[1]) as file:
    data = file.require_group("Comparison Data")        
    compData = data.require_datagroup("Exact "+sys.argv[2])
    compData.attrs['time'] = float(sys.argv[2])
    sub_domain = file[file.keys()[1]].attrs['Interval'].getGrid(-1, 1)
    subtrahead = numpy.array([exact_solution(sub_domain)])
    datas = [h5py_array.DataGroup(file[file.keys()[i]]) \
        for i in range(2, 5)]
    datas += [h5py_array.DataGroup(file[file.keys()[1]])]
    for i in range(len(datas)):
        diff, domain = datagroup_utils.data_group_difference_at_attr( \
            datas[i], (subtrahead, sub_domain), 'time', float(sys.argv[2]), \
            attr_comparor = comparor)#, domain_comparor = comparor)
        compData[i] = numpy.absolute(diff)
        compData[i].attrs['original'] = datas[i].name
        compData[i].attrs['domain'] = domain
    print 'Done'




