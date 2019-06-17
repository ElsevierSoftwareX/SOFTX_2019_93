Overview
========

Coffee is the name for a code library and collection of scripts written in python.
It is  designed to solve
hyperbolic PDE's via the method of lines. The design philosophy follows that of
python. Coffee does not fail gracefully, you are expected to read code and
can dynamically alter essentially everything about the program.

Installation instructions
=======

These instructions describe how to get coffee's needed python modules installed.
Some of these modules require c libraries. We do not claim to provide complete
documentation for installation of these secondary libraries. Please refer to
third-party installation instructions where needed. The instructions below
cover installation on Ubuntu 18.04.

1. Clone the coffee repository locally.
1. Include the coffee/packages library in your `PYTHONPATH`. 
``cd <path to coffee/package directory>``
``export PYTHONPATH=$(pwd):$PYTHONPATH``
1. Coffee uses `python2`. We prefer ``virualenv`` but any method to avoid repository
contamination can be used.
``virtualenv -p <python2 executable> --system-site-packages``
``source venv/bin/activate``
1. Install ``gnuplot-py`` if you wish to use ``gnuplot`` for visualisation (http://www.gnuplot.info/). The 
other alternative is ``matplotlib``. Examples scripts all use ``gnuplot`` and do
not detect its absence. Errors will be generated if they display output to the
screen.
``cd gnuplot-py``
``tar xvfz gnuplot-py-1.8.tar.gz``
``cd gnugnuplot-py-1.8``
``python setup.py install``
``cd ../..``
1. Install the following python modules. They are likely to rely on additional
``C`` libraries. Please see module specific documentation to resolve any issues.
The modules are: ``mpi4py``, ``h5py``, ``scipy``, ``PyFFTW3``.
1. Compile the spin weighted spherical harmonic routines.
``cd coffee/swsh/spinsfastpy/huffenberger_wandelt/``
``tar xvfz spinsfast_rev104.tar.gz``
``cd spinsfast_rev104``
You will now need to follow the installation instructions listed in ``README``.
You should be able to install ``spinsfast`` by executing the following:
``export build=build/config.mk``
``make``
If you encounter missing headers first attempt installation of the following
``C``-libraries: ``libfftw3-dev`` and ``python-numpy``.
1. Go to the systems directory and attempt to run them.



