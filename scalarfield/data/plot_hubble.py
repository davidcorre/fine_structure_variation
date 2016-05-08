#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

filename_BAO='BAO_farook.txt'

BAO_data = np.genfromtxt('%s' % (filename_BAO), dtype=float, delimiter=' ',names=True)


z_BAO = BAO_data['Redshift']*1.
H_BAO = BAO_data['H']*1.
H_BAO_error = BAO_data['error']*1.


plt.errorbar(z_BAO,H_BAO,H_BAO_error,fmt='r.', markersize=3)
plt.xlabel('$z$')
plt.ylabel('$H(z)$')
plt.grid(True)
plt.savefig("H_farook.png")
plt.show()
