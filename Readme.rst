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

About the "sigmas" used in this cornet plot: see the documentation here: https://github.com/dfm/corner.py

- Scalar field
The scalar fields are computed using the ScalPy code: https://github.com/sum33it/scalpy

Here you just find the scalar.py from these code as it does not need more and it is modified to include alpha variations. 


Usage
======

Once you have loaded the python 2.7 environment by (see above) just write in a terminal
::

    python run_emcee.py

if it works fine everything should be installed correctly

Now, there are two scalar fields that can be used:
    - exponential potential
    - power law potential
    
In order to load a model just write:
::

    scalarexp(gamma_i,Ophi_i,lambda_i): scalar field with exponential potential
    scalarpow(gamma_i,Ophi_i,lambda_i,n): scalar field with power law potential
with the following arguments:

::

    gamma_i: initial value 
    ophi_i: initial value (at a = 0.001 or z \approax 1000) of density parameter for scalar field
    lambda_i: initial value for lambda parameter (see "Sen and Scherrer")
    n: order of power law potential V(phi)=phi^n

In the ScalPy the gamma_i is set to 1e-4 and is not able to vary. Here I allow it to vayr


The code is pretty well described so you should be able to uderstand the role of each part.

Briefly, in first part of run_emcee.py you define where the data are saved and where you want to store your results.
Then you select the potential you want to use, the number of walkers and the number of steps each walker has to achieve. The initial values of parameters to be sampled are drawned from a gaussian with the mean being "pos0" and a deviation given by "pos0_dispersion". Each walker will start with different initial values.

You can also chose either to correct the resulting samples and to plot them or not.

The parameters 
