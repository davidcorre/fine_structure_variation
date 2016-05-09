Installation 
============
(tested only on Ubuntu 14.04 LTS)

Install Anaconda
++++++++++++++++
Very practical to build a python environment (Either Python 2.7 or 3.5)
In my opinion it is better to install anaconda for 3.5 version as you can 
specify which version of python you want to use.
https://www.continuum.io/downloads

Create an environment to use python 2.7 and its dependencies

::

    conda create -n py27 python=2.7 anaconda

So now every time you want to use these libraries just write in a terminal
source activate py27
# To use your normal path again just write 
source deactivate

#Install emcee (MCMC sampler)
conda install -n py27 --channel https://conda.anaconda.org/OpenAstronomy emcee

#Install corner plot
pip install corner




Make some beautiful corner plots.

Corner plot /ˈkôrnər plät/ (noun):
    An illustrative representation of different projections of samples in
    high dimensional spaces. It is awesome. I promise.

Built by `Dan Foreman-Mackey <http://dan.iel.fm>`_ and collaborators (see
``corner.__contributors__`` for the most up to date list). Licensed under
the 2-clause BSD license (see ``LICENSE``).


Installation
------------

Just run

::

    pip install corner

to get the most recent stable version.


Usage
-----

The main entry point is the ``corner.corner`` function. You'll just use it
like this:

::

    import numpy as np
    import corner

    ndim, nsamples = 5, 10000
    samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
    figure = corner.corner(samples)
    figure.savefig("corner.png")

With some other tweaks (see `demo.py
<https://github.com/dfm/corner.py/blob/master/demo.py>`_) you can get
something that looks awesome like:

.. image:: https://raw.github.com/dfm/corner.py/master/corner.png

By default, data points are shown as grayscale points with contours.
Contours are shown at 0.5, 1, 1.5, and 2 sigma.

For more usage examples, take a look at `tests.py
<https://github.com/dfm/corner.py/blob/master/tests.py>`_.


Documentation
-------------

All the options are documented in the docstrings for the ``corner`` and
``hist2d`` functions. These can be viewed in a Python shell using:

::

    import corner
    print(corner.corner.__doc__)

or, in IPython using:

::

    import corner
    corner.corner?


A note about "sigmas"
+++++++++++++++++++++
See the documentation here: https://github.com/dfm/corner.py
