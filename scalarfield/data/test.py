#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

filename='BAO_farook_rep.txt'
BAO_data = np.genfromtxt('%s' % filename, dtype=float, delimiter=' ',names=True)

####### BAO     ##########
z = BAO_data['Redshift']*1.
H = BAO_data['H']*1.
H_error = BAO_data['error']*1.

H_comov = H / (1.+z)
H_comov_err = H_error/ (1.+z)

ini = []

#z=np.linspace(0.,10.,1000)


#plt.plot(z,power_law,'r',label='exp')
plt.figure(1)
plt.errorbar(z,H_comov,H_comov_err,fmt='b.', markersize=3,label='iii')

plt.xlabel('$z$')
plt.ylabel('$H(z)/(1+z)$')
#plt.ylim([-1.5, 1])
plt.legend()
#plt.title("Espresso / ideal case / QSO only")
#plt.savefig("espresso_qso_ideal.png")
plt.show()

plt.figure(2)
plt.errorbar(z,H,H_error,fmt='b.', markersize=3,label='iii')
plt.xlabel('$z$')
plt.ylabel('$H(z)$')
#plt.ylim([-1.5, 1])
plt.legend()
#plt.title("Espresso / ideal case / QSO only")
#plt.savefig("espresso_qso_ideal.png")
plt.show()

