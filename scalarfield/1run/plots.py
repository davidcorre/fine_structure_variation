# -*- coding:utf-8 -*-


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner as triangle
import numpy as np


def plot_steps(data,results_dir,disp,params2sample):
    """ Plot the evolution of each parameters in function the steps"""

    ndim = len(data[0,0,:])
    
    #Plot the step evolution
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    
    axes[0].plot(data[:, :, 0].T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    #axes[0].axhline(Om0_fid, color="#888888", lw=2)
    axes[0].set_ylabel("$gamma_i$")
    
    axes[1].plot(data[:, :, 1].T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    #axes[1].axhline(w0_fid, color="#888888", lw=2)
    axes[1].set_ylabel("$omi$")
    axes[1].set_xlabel("step number")
    
    axes[2].plot(data[:, :, 2].T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    #axes[2].axhline(wa_fid, color="#888888", lw=2)
    axes[2].set_ylabel("$slopei$")
    axes[2].set_xlabel("step number")
    
    if ndim == 4:
             axes[3].plot(data[:, :, 3].T, color="k", alpha=0.4)
             axes[3].yaxis.set_major_locator(MaxNLocator(5))
             axes[3].set_xlabel("step number")
             if 'zeta' in params2sample:
                  #axes[3].axhline(zeta_fid, color="#888888", lw=2)
                  axes[3].set_ylabel("$zeta$")
             else:
                  axes[3].set_ylabel("n")
                  #axes[3].axhline(n_fid, color="#888888", lw=2)
    if ndim == 5:
             axes[4].plot(data[:, :, 4].T, color="k", alpha=0.4)
             axes[4].yaxis.set_major_locator(MaxNLocator(5))
             axes[4].axhline(zeta_fid, color="#888888", lw=2)
             axes[4].set_xlabel("step number")
             axes[4].set_ylabel("$zeta$")
    
    fig.tight_layout(h_pad=0.0)
    fig.savefig('%s/line-time_corr.png' % (results_dir))
    if disp: plt.show()

def corner_plot(data,results_dir,burnin,disp,params2sample):
    """ Plot the corner plot """


    ndim=len(data[0,0,:])
    #Reshape because samples should be a 2 dimension array to be given to the triangle.corner function
    samples = data[:, burnin:, :].reshape((-1, ndim))
    
    if ndim == 3:
        fig = triangle.corner(samples, bins=20,labels=["$gamma_i$","$omi$", "$slopei$"],
                          quantiles=[0.16, 0.5, 0.84],
                          plot_contours = True,
                          plot_datapoints = False,
                          plot_ellipse = False,
                          levels=[0.68,0.95,0.99],
                          range=[1.,1.,1.],color='blue',
                          show_titles=True, title_fmt='.2e',title_kwargs={"fontsize": 12})
    elif ndim == 4:
        if 'zeta' in params2sample:
             fig = triangle.corner(samples, bins=20,labels=["$gamma_i$","$omi$", "$slopei$","zeta"],
                          quantiles=[0.16, 0.5, 0.84],
                          plot_contours = True,
                          plot_datapoints = False,
                          plot_ellipse = False,
                          levels=[0.68,0.95,0.99],
                          range=[1.,1.,1.,1.],color='blue',
                          show_titles=True, title_fmt='.2e',title_kwargs={"fontsize": 12})
        else:
             fig = triangle.corner(samples, bins=20,labels=["$gamma_i$","$omi$", "$slopei$","n"],
                          quantiles=[0.16, 0.5, 0.84],
                          plot_contours = True,
                          plot_datapoints = False,
                          plot_ellipse = False,
                          levels=[0.68,0.95,0.99],
                          range=[1.,1.,1.,1.],color='blue',
                          show_titles=True, title_fmt='.2e',title_kwargs={"fontsize": 12})
    elif ndim==5:
        fig = triangle.corner(samples, bins=20,labels=["$gamma_i$","$omi$", "$slopei$","n","zeta"],
                          quantiles=[0.16, 0.5, 0.84],
                          plot_contours = True,
                          plot_datapoints = False,
                          plot_ellipse = False,
                          levels=[0.68,0.95,0.99],
                          range=[1.,1.,1.,1.,1.],color='blue',
                          show_titles=True, title_fmt='.2e',title_kwargs={"fontsize": 12})
    
    
    fig.set_size_inches(12,12)
    fig.savefig('%s/line-triangle_corr.png' % (results_dir))
    if disp: plt.show()

