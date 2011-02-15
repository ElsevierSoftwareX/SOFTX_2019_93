import sys
sys.path.append("../bin")
import numpy
import h5py_array
import h5py
import datagroup_utils
import tslices

#sys.argv[1] = 'exp_-20xx.hdf'
#sys.argv[2] = '2.0'

def time_comparor(x, y):
    """y is the value to be comparied against."""
    return y-time_comparor.dt/2<x<y+time_comparor.dt/2

with h5py_array.H5pyArray(sys.argv[1]) as file:
    rawData = file.require_group("/Raw Data")
    compData = file.require_group("/Comparison Data").require_group(sys.argv[2])
    compData.attrs['time'] = float(sys.argv[2])
    datalist = map(lambda x:h5py_array.DataGroup(x), rawData.sortted_by_attr())
    time = float(sys.argv[2])
    try:
        if sys.argv[3] == "Exact":
            exact = True
        else:
            exact = False
    except:
        exact = False
    if exact:
        minuends = datalist
        domain = datalist[-1].attrs['grid'].getGrid(-1, 1)
        subtrahend = datalist[-1].attrs['system'].initialValues(time, x)
    else:
        subtrahend = datalist[-1]
        minuends = datalist[0:-1]
    print "Starting computation..."
    for minuend in minuends:
        if not exact:
            print"Calculating %s - %s" %(minuend.name, subtrahend.name)
            
            #Finding index for correct time
            minuend_index = -1
            for i in range(len(minuend)):
                dt = minuend[i].attrs['dt']
                if time-dt/2<minuend[i].attrs['time']<time+dt/2:
                    minuend_index = i
                    break
            subtrahend_index = -1
            for i in range(len(subtrahend)):
                dt = subtrahend[i].attrs['dt']
                if time-dt/2<subtrahend[i].attrs['time']<time+dt/2:
                    subtrahend_index = i
                    break
            
            #Comaring domains
            minuend_domain = minuend[minuend_index].attrs['domain']
            subtrahend_domain = subtrahend[subtrahend_index].attrs['domain']
            mapping = datagroup_utils.array_value_index_mapping(minuend_domain,subtrahend_domain)
            
            #Calculating difference in values of common domains
            diff = numpy.ones_like(minuend[minuend_index].value)
            for map in mapping:
                diff[:, map[0]] = minuend[minuend_index][:, map[0]]- \
                    subtrahend[0][:, map[1]]
        else:
            print "Calculating %s - exact solution" %(d.name)
        
        #Storing data
        diffset = compData.require_dataset(minuend.name.split('/')[-1],\
            diff.shape, \
            diff.dtype, data = numpy.absolute(diff))
        diffset.attrs['minuend'] = minuend.name
        diffset.attrs['subtrahend'] = subtrahend.name
        diffset.attrs['domain'] = minuend_domain
        diffset.attrs['time'] = time
    print '...computation complete.-'



    
