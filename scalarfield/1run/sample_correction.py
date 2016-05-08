#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

def correct_sample(data_raw,burnin):
    """ Correct the samples of the emcee run in order to delete walker which did not move"""

    some_walker2delete=False
    delta_step = 50
    nb_steps = len(data_raw[0,:,0])
    ndim=len(data_raw[0,0,:])
    index=[]
    
    #delete the walkers with uneralistic values
    for walker in xrange(len(data_raw[:,0,0])):
        # delete walkers where Omega_phi is negtive after 300 iterations (should be burnin though)
        if (data_raw[walker,burnin:,0] < 0.).any():
             some_walker2delete=True
             index.append(walker)
    
    #Check if a walker changes its value between a given step 'step1' and 'step1'+'delta_step'
    #Check is done for all parameters togethers
    #If it all parameters have exatly the same values, the walker is deleted as it probably means that it 
    #got stocked and did not move anymore
    #one might use the variance of the chain instead
    for walker in xrange(len(data_raw[:,0,0])):
        counter_arr=np.zeros(nb_steps/delta_step-1)
        for step in xrange(0,nb_steps-delta_step,delta_step):
             if ndim == 3:
                  if (data_raw[walker,step,0] != data_raw[walker,step+delta_step,0]) and (data_raw[walker,step,1] != data_raw[walker,step+delta_step,1]) and (data_raw[walker,step,2] != data_raw[walker,step+delta_step,2]):
                       counter_arr[step/delta_step]=1
                  else:
                       continue
             elif ndim == 4:
                  if (data_raw[walker,step,0] != data_raw[walker,step+delta_step,0]) and (data_raw[walker,step,1] != data_raw[walker,step+delta_step,1]) and (data_raw[walker,step,2] != data_raw[walker,step+delta_step,2]) and (data_raw[walker,step,3] != data_raw[walker,step+delta_step,3]):
                       counter_arr[step/delta_step]=1
                  else:
                       continue
    
             elif ndim == 5:
                  if (data_raw[walker,step,0] != data_raw[walker,step+delta_step,0]) and (data_raw[walker,step,1] != data_raw[walker,step+delta_step,1]) and (data_raw[walker,step,2] != data_raw[walker,step+delta_step,2]) and (data_raw[walker,step,3] != data_raw[walker,step+delta_step,3]) and (data_raw[walker,step,4] != data_raw[walker,step+delta_step,4]):
                       counter_arr[step/delta_step]=1
                  else:
                       continue
    
        if (counter_arr==0).any():
             some_walker2delete=True
             index.append(walker)
    
    
    #delete duplicate
    index=sorted(set(index))
    if some_walker2delete:
        for i in index[::-1]:
             print ("walker deleted: %d" % i)
             data_corr=np.delete(data_raw,i,0)
    """
    f = open(results_dir+'/chain_corr.npy','w+b')
    np.save(f,data_corr)
    f.close()
    """

    return data_corr

