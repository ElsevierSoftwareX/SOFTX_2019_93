Getting started
===============

Installation instructions
-------------------------

These instructions describe how to get COFFEE's needed python modules installed.
Some of these modules require c libraries. We do not claim to provide complete
documentation for installation of these secondary libraries. Please refer to
third-party installation instructions where needed. The instructions below
cover installation on Ubuntu 18.04.

1. Clone the coffee repository locally.

2. Include the coffee/packages library in your `PYTHONPATH`. 
``cd <path to coffee/package directory>``
``export PYTHONPATH=$(pwd):$PYTHONPATH``

3. Coffee uses `python2`. We prefer ``virualenv`` but any method to avoid repository
contamination can be used.
``virtualenv -p <python2 executable> --system-site-packages``
``source venv/bin/activate``

4. Install ``gnuplot-py`` if you wish to use ``gnuplot`` for visualisation 
(http://www.gnuplot.info/). The other alternative is ``matplotlib``. 
Examples scripts all use ``gnuplot`` and do not detect its absence. 
Errors will be generated if they display output to the screen.
``cd gnuplot-py``
``tar xvfz gnuplot-py-1.8.tar.gz``
``cd gnugnuplot-py-1.8``
``python setup.py install``
``cd ../..``

5. Install the following python modules. They are likely to rely on additional
``C`` libraries. Please see module specific documentation to resolve any issues.
The modules are: ``mpi4py``, ``h5py``, ``scipy``, ``PyFFTW3``.

6. Compile the spin weighted spherical harmonic routines.
``cd coffee/swsh/spinsfastpy/huffenberger_wandelt/``
``tar xvfz spinsfast_rev104.tar.gz``
``cd spinsfast_rev104``
You will now need to follow the installation instructions listed in ``README``.
You should be able to install ``spinsfast`` by executing the following:
``export build=build/config.mk``
``make``
If you encounter missing headers first attempt installation of the following
``C``-libraries: ``libfftw3-dev`` and ``python-numpy``.

7. Go to the systems directory and attempt to run them.

Performing your first simulation and getting to know COFFEE
-----------------------------------------------------------

COFFEE contains a handful of examples which solve simple differtial equations
using its capabilities. After installation you should make sure that each 
example runs without error and can display to your screen. 

The first example you try should be the `OndDAdvection` system located in the
`Code/systems/advection/OneDAdvection` folder. To run this system navigate to 
this folder and type `python OneDAdvection_setup.py`. This file instantiates an 
`IBVP` object and calls the objects `run` method. This simulation will commense 
and, assuming that everything is setup correctly, an animation of the simulation
will be displayed.

You should take a look at the file `python OneDAdvection_setup.py`. It contains
a large number of options and constructs a handful of objects. These options and
objects control how the simulation proceeds and what happens during the
simulation. 

This file and other example "setup files" contain more code than is necessary 
(for example all the stuff to do with `argparse`). Nevertheless they give a
detailed and concrete demonstration of what COFFEE can do.

Please enjoy looking at the other examples located in the `Code/systems` folder.

