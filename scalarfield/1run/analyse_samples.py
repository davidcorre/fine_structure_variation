# -*- coding:utf-8 -*-

import numpy as np
from sample_correction import correct_sample
from plots import plot_steps, corner_plot

#Directory 
results_dir='results/scalarexp/'

#Import data
data_raw = np.load(results_dir+'/chain.npy')

params2sample=['gammai','omega_phii','slopei','zeta']


burnin=10     #The first step from which values are plotted

samples_correction=False
make_plots=True
disp_plot=True
#--------------------------------------------------------
#Correction of the samples
#--------------------------------------------------------
if samples_correction:
    print ("\nStart correction of the samples\n")
    data_corr = correct_sample(data_rawi,burnin)


#----------Plots--------------------------------------------#
if make_plots:
    #import the data
    if samples_correction: data = data_corr
    else:  data=data_raw


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

# Need to be updated if more, or less, parameters are chosen
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

