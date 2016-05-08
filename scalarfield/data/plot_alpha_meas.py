#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

filename_VLT = 'alpha_UVES.txt'
filename_KECK='alpha_keck.txt'
filename_small='alpha_ana_marta2.txt'

data_vlt = np.genfromtxt('%s' % filename_VLT, dtype=float, delimiter=' ',names=True,skip_header=0)
data_keck = np.genfromtxt('%s' % filename_KECK, dtype=float, delimiter=' ',names=True,skip_header=0)
data_small = np.genfromtxt('%s' % filename_small, dtype=float, delimiter=' ',names=True,skip_header=0)

z_vlt =data_vlt['Redshift']
alpha_vlt = data_vlt['alpha']
error_vlt =data_vlt['error']

z_keck =data_keck['Redshift']
alpha_keck = data_keck['alpha']
error_keck =data_keck['error']

z_small =data_small['Redshift']
alpha_small = data_small['alpha_var_ppm']*0.1
error_small =data_small['error']*0.1

fig, axis = plt.subplots(2, 3, sharex=True, figsize=(15, 10))

#KECK
axis[0,0].errorbar(z_keck,alpha_keck,error_keck,fmt='r.', markersize=3)
axis[0,0].set_ylabel(r'$\Delta \alpha / \alpha$ ($10^{-5})$')
axis[0,0].set_ylim([-30,40])
axis[0,0].set_title('alpha measurements from KECK')
axis[0,0].axhline(0, color='black',lw=2,ls='dashed')
axis[0,0].grid(True)

axis[1,0].errorbar(z_keck,alpha_keck,error_keck,fmt='r.', markersize=3)
axis[1,0].set_xlabel('$z$')
axis[1,0].set_ylabel(r'$\Delta \alpha / \alpha$ ($10^{-5})$')
axis[1,0].set_ylim([-10, 10])
axis[1,0].grid(True)
axis[1,0].axhline(0, color='black',lw=2,ls='dashed')

#VLT
axis[0,1].errorbar(z_vlt,alpha_vlt,error_vlt,fmt='r.', markersize=3)
axis[0,1].set_ylabel(r'$\Delta \alpha / \alpha$ ($10^{-5})$')
axis[0,1].set_ylim([-30,40])
axis[0,1].set_title('alpha measurements from VLT')
axis[0,1].axhline(0, color='black',lw=2,ls='dashed')
axis[0,1].grid(True)

axis[1,1].errorbar(z_vlt,alpha_vlt,error_vlt,fmt='r.', markersize=3)
axis[1,1].set_xlabel('$z$')
axis[1,1].set_ylabel(r'$\Delta \alpha / \alpha$ ($10^{-5})$')
axis[1,1].set_ylim([-10, 10])
axis[1,1].grid(True)
axis[1,1].axhline(0, color='black',lw=2,ls='dashed')

#Small
axis[0,2].errorbar(z_small,alpha_small,error_small,fmt='r.', markersize=3)
axis[0,2].set_ylabel(r'$\Delta \alpha / \alpha$ ($10^{-5})$')
axis[0,2].set_ylim([-30,40])
axis[0,2].set_title('recent alpha measurements')
axis[0,2].axhline(0, color='black',lw=2,ls='dashed')
axis[0,2].grid(True)

axis[1,2].errorbar(z_small,alpha_small,error_small,fmt='r.', markersize=3)
axis[1,2].set_xlabel('$z$')
axis[1,2].set_ylabel(r'$\Delta \alpha / \alpha$ ($10^{-5})$')
axis[1,2].set_ylim([-10, 10])
axis[1,2].grid(True)
axis[1,2].axhline(0, color='black',lw=2,ls='dashed')

fig.tight_layout(h_pad=0.0)
fig.savefig("alpha_datasets.png")
fig.show()
