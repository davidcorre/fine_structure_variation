# -*- coding:utf-8 -*-

import numpy as np
import emcee
from emcee.utils import MPIPool
from scalar import *
from sample_correction import correct_sample
from plots import corner_plot, plot_steps

#Set up the different paths and data filenames
rootname='../../data'
filename_SN='Union_21_SN.txt'
filename_BAO='BAO_farook.txt'
filename_alpha='alpha_ana_marta.txt'

#Set results directory
results_dir='results'

#Input files 
SN_data = np.genfromtxt('%s/%s' % (rootname,filename_SN), dtype=float, delimiter='\t',names=True,skip_header=5)
BAO_data = np.genfromtxt('%s/%s' % (rootname,filename_BAO), dtype=float, delimiter=' ',names=True)
alpha_data = np.genfromtxt('%s/%s' % (rootname,filename_alpha), dtype=float, delimiter=' ',names=True)

#Select model
model_type=2    # 1: power law    2: exponential

#Select parameters

#For power law
if model_type ==1:
    params2sample=['gammai','omega_phii','slopei','n','zeta']

    pos0 = [0.1,1.3e-9,1.,-2.,0.]      # Initial values of the parameters  [Gamma, Omega_phi,slope,n,zeta]
    pos0_dispersion = [0.05,5e-10,0.2,0.3,5e-6]    #Add a dispersion on these initial values
    results_dir=results_dir+'/scalarpow'

#For exponential
elif model_type == 2:
    params2sample=['gammai','omega_phii','slopei','zeta']

    pos0 = [0.1,1.3e-9,1.,0.]      # Initial values of the parameters  [Gamma, Omega_phi,slope,zeta]
    pos0_dispersion = [0.01,5e-10,0.2,5e-6]    #Add a dispersion on these initial values 
    results_dir=results_dir+'/scalarexp'

ndim = len(pos0)                # number of parameters to fit

# Set up the sampler parameters
nwalkers = 10           # number of walkers
steps = 50              # Number of steps per walker
num_threads = 4         # number of threads to use
scale_parameter = 2     # if acceptance fractio too low, increase it. 

#Whether the samples have to be corrected
samples_correction=False

#Whether to display the plots
disp_plot=True

#The first step from which values are plotted
burnin=10



#####################################################
# SN union2.1 data
#####################################################
z_sn = SN_data['Redshift']*1.
#magnitude
mu_obs = SN_data['Distance_modulus']*1.
#uncertainty on magnitude
mu_error = SN_data['error']*1.
#Luminosity distance from measuremnts
DL_obs = 10**((mu_obs-25.)/5.)
#uncertainty on luminosity distance from observations
DL_error = mu_error /5. * DL_obs * np.log(10.)


#def chi_union21(gammai,Omi,n,slopei):
def chi_union21(params):
    if model_type == 1:  
         gammai,Omi,slopei,n=params
         model=scalarpow(gammai,Omi,slopei,n)
    elif model_type == 2:
         gammai,Omi,slopei=params
         model=scalarexp(gammai,Omi,slopei)

    inv_sigma2 = 1.0/(DL_error**2)
    sum_obs = 0.0
    for i in xrange(len(z_sn)):
         sum_obs += ((model.lum_dis_z(z_sn[i])-DL_obs[i])**2)*inv_sigma2[i]
    return -0.5*(sum_obs)


######################################################
# BAO data
######################################################
z_BAO = BAO_data['Redshift']*1.
H_BAO = BAO_data['H']*1.
H_BAO_error = BAO_data['error']*1.

def chibao(params):
    if model_type == 1:
         gammai,Omi,slopei,n=params
         model=scalarpow(gammai,Omi,slopei,n)
    elif model_type == 2:
         gammai,Omi,slopei=params
         model=scalarexp(gammai,Omi,slopei)

    inv_sigma2_BAO = 1.0/(H_BAO_error**2)
    sum_th_BAO = 0.0
    for i in xrange(len(z_BAO)):
         sum_th_BAO += ((model.hubz(z_BAO[i])*70.-H_BAO[i])**2)*inv_sigma2_BAO[i]
    return -0.5*(sum_th_BAO)

#####################################################
#Alpha data
######################################################
z_alpha = alpha_data['Redshift']*1.
alpha_var_obs = alpha_data['alpha_var_ppm']*1e-6
alpha_var_error = alpha_data['error'] * 1e-6



def chi_alpha(params):
    if model_type == 1:
         gammai,Omi,slopei,n,zeta=params
         model=scalarpow(gammai,Omi,slopei,n)
    elif model_type == 2:
         gammai,Omi,slopei,zeta=params
         model=scalarexp(gammai,Omi,slopei)

    inv_sigma2_alpha = 1.0/(alpha_var_error**2)
    sum_th_alpha = 0.0
    for i in xrange(len(z_alpha)):
         sum_th_alpha += ((model.alpha_var(z_alpha[i],zeta)-alpha_var_obs[i])**2)*inv_sigma2_alpha[i]
    return -0.5*(sum_th_alpha)

#####################################################



def chi2(params):
    #gammai,Omi,n,slopei,zeta=params
    if 'zeta' in params2sample:
         return chi_union21(params[:-1]) + chibao(params[:-1]) + chi_alpha(params)
    else:
         return chi_union21(params) + chibao(params)


def lik(params):
    #gammai,Omi,n,slopei,zeta=params
    return np.exp(-chi2(params)/2.)

def lnprior(params):
    #gammai,Omi,n,slopei,zeta=params
    if model_type == 1:
         gammai,Omi,slopei,n,zeta=params
         # If w0 and Omega_mattar_0 are included in the priors: 
         #(otherwise comment the 3 next lines, it will save some comp. time)
         model=scalarpow(gammai,Omi,slopei,n)
         w0 = model.eqn_state_pres()
         Om0 = model.om_z(0.)
    elif model_type == 2:
         gammai,Omi,slopei,zeta=params
         # If w0 and Omega_mattar_0 are included in the priors: 
         #(otherwise comment the 3 next lines, it will save some comp. time)
         model=scalarexp(gammai,Omi,slopei)
         w0 = model.eqn_state_pres()
         Om0 = model.om_z(0.)

    if model_type == 1:
         if  (1e-4 < gammai < 2.) and (1e-11 < Omi < 1e-3) and (-20.<n<-0.1) and (0.1 < slopei<np.sqrt(6.)) and (-1.15 < w0 < -0.85) and (0.28 < Om0 < 0.35) and (-1e-4 < zeta < 1e-4):
         #if  (1e-4 < gammai < 2.) and (1e-11 < Omi < 1e-3) and (-20.<n<-0.1) and (0.1 < slopei<np.sqrt(6.)) and (-1e-4 < zeta < 1e-4):
              return 0.0
         return -np.inf
    elif model_type == 2:
         if  (1e-4 < gammai < 2.) and (1e-11 < Omi < 1e-3) and (0.1 < slopei<np.sqrt(6.)) and (-1.15 < w0 < -0.85) and (0.28 < Om0 < 0.35) and (-1e-4 < zeta < 1e-4):
         #if  (1e-4 < gammai < 2.) and (1e-11 < Omi < 1e-3) and (0.1 < slopei<np.sqrt(6.)) and (-1e-4 < zeta < 1e-4):
              return 0.0
         return -np.inf

def lnlike(params):
    #gammai,Omi,n,slopei,zeta=params
    return np.log(lik(params))

def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
         return -np.inf
    #return lp + lnlike(n,slopei,zeta)
    return lp + chi2(params)


def lnp(params):
    #gammai,Omi,n,slopei,zeta=params
    #zeta=0.
    return lnprob(params)


# Find the maximum likelihood value.

#result = opt.minimize(ff, [0.3,-1.0, 0.])
#Om0_ml,w0_ml,wa_ml = result['x']
#print("""Maximum likelihood result:
#    Om0 = {0} (truth: {1})
#    w0 = {2} (truth: {3})
#    wa = {4} (truth: {5})
#""".format(Om0_ml, 0.3, w0_ml, -1.0, wa_ml, 0.0))


# Set up the sampler.
pos = [pos0 + pos0_dispersion*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp,threads=num_threads,a=scale_parameter)
print ('Initial values:')
print (pos)
# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, steps)
print("\nDone.")

print("\nMean acceptance fraction before corrections: {0:.3f}\n"
          .format(np.mean(sampler.acceptance_fraction)))


#Save the lnprobabilty for each parameter at each time step
# lnprobability has the form (nb of walkers, nb of iterations)
values_to_save = sampler.lnprobability
f = open('%s/lnproba.npy' % results_dir,'w+b')
np.save(f,values_to_save)
f.close()

#save the value of each parameter for each walker at each step
#Chain has the form: (nb of walkers, nb of iterations, nb of parameters)

values_to_save = sampler.chain
f2 = open('%s/chain.npy' % results_dir,'w+b')
np.save(f2,values_to_save)
f2.close()

#store acceptance fraction for each walker
values_to_save = sampler.acceptance_fraction
f3 = open('%s/acceptance_fraction.npy' % results_dir,'w+b')
np.save(f3,values_to_save)
f3.close()

#store autocorrelation time for each parameter
values_to_save = sampler.acor
f4 = open('%s/acor.npy' % results_dir,'w+b')
np.save(f4,values_to_save)
f4.close()


#--------------------------------------------------------
#Correction of the samples
#--------------------------------------------------------
if samples_correction:
    print ("\nStart correction of the samples\n")
    #import the data
    data_raw = np.load(results_dir+'/chain.npy')
   
    data_corr = correct_sample(data_raw,burnin)
 
    f = open(results_dir+'/chain_corr.npy','w+b')
    np.save(f,data_corr)
    f.close()


#----------Plots--------------------------------------------#
#import the data
if samples_correction: data = np.load(results_dir+'/chain_corr.npy')
else:  data=np.load(results_dir+'/chain.npy')


#Plot the steps evolution
plot_steps(data,results_dir,disp_plot,params2sample)

#Plot the corner plot
corner_plot(data,results_dir,burnin,disp_plot,params2sample)



# Compute the quantiles.
ndim=len(data[0,0,:])
#Reshape because samples should be a 2 dimension array to be given to the triangle.corner function
samples = data[:, burnin:, :].reshape((-1, ndim))

params_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
# Following needs to be modified if more, or less, parameters are chosen
# or if different parameters are used
print("""MCMC result:
    Parameter = Median | 1st quantile | 3rd quantile
    Gammai = {0[0]} | +{0[1]} | -{0[2]}
    Omega_Phii = {1[0]} | +{1[1]} | -{1[2]}
    slopei = {2[0]} | +{2[1]} | -{2[2]}
    zeta = {3[0]} | +{3[1]} | -{3[2]}
""".format(params_mcmc[0], params_mcmc[1], params_mcmc[2], params_mcmc[3]))

#Write the median value and the first and thrid quantile for each parameters
file_out=open(results_dir+'/median_values.txt',"w")
file_out.write("""MCMC result:
    Parameter = Median | 1st quantile | 3rd quantile
    Gammai = {0[0]} | +{0[1]} | -{0[2]}
    Omega_Phii = {1[0]} | +{1[1]} | -{1[2]}
    slopei = {2[0]} | +{2[1]} | -{2[2]}
    zeta = {3[0]} | +{3[1]} | -{3[2]}
""".format(params_mcmc[0], params_mcmc[1], params_mcmc[2], params_mcmc[3]))
file_out.close()

