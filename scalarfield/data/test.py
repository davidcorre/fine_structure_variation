#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

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


#pl.plot(z,power_law,'r',label='exp')
pl.figure(1)
pl.errorbar(z,H_comov,H_comov_err,fmt='b.', markersize=3,label='iii')

pl.xlabel('$z$')
pl.ylabel('$H(z)/(1+z)$')
#pl.ylim([-1.5, 1])
pl.legend()
#pl.title("Espresso / ideal case / QSO only")
#pl.savefig("espresso_qso_ideal.png")
pl.show()

pl.figure(2)
pl.errorbar(z,H,H_error,fmt='b.', markersize=3,label='iii')
pl.xlabel('$z$')
pl.ylabel('$H(z)$')
#pl.ylim([-1.5, 1])
pl.legend()
#pl.title("Espresso / ideal case / QSO only")
#pl.savefig("espresso_qso_ideal.png")
pl.show()

