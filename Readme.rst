Installation 
============
(tested only on Ubuntu 14.04 LTS)

- Install Anaconda

Very practical to build a python environment (either Python 2 or 3)
Install the version for python 3 as you can specify which version of python you want to use.
Download it here: https://www.continuum.io/downloads

Create an environment to use python 2.7 and its dependencies

::

    conda create -n py27 python=2.7 anaconda

So now every time you want to use these libraries just write in a terminal
::

    source activate py27

To exit the environment and use your standard path again just write 
::

    source deactivate

- Install emcee (MCMC sampler)

locally in this environnement only:
::
 
    conda install -n py27 --channel https://conda.anaconda.org/OpenAstronomy emcee

or if you want to install on your standard path 
::
    
    pip install emcee

For more infos about it see: https://github.com/dfm/emcee

- Install corner plot
just type in a terminal
::

    pip install corner

A note about "sigmas"
+++++++++++++++++++++
See the documentation here: https://github.com/dfm/corner.py
