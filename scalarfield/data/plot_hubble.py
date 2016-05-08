#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

filename_BAO='BAO_farook.txt'

BAO_data = np.genfromtxt('%s' % (filename_BAO), dtype=float, delimiter=' ',names=True)


z_BAO = BAO_data['Redshift']*1.
H_BAO = BAO_data['H']*1.
H_BAO_error = BAO_data['error']*1.

pl.clf()

pl.errorbar(z_BAO,H_BAO,H_BAO_error,fmt='r.', markersize=3)
pl.xlabel('$z$')
pl.ylabel('$H(z)$')
pl.grid(True)
pl.savefig("H_farook.png")
pl.show()
