#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:47:24 2018

@author: Isaac Brown
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from astroML.density_estimation import KNeighborsDensity
from astroML.plotting import setup_text_plots

setup_text_plots(fontsize=8, usetex=False)
ra, dec = np.loadtxt('simbad3.tsv',usecols=(5,6),unpack=True)

X = np.vstack((dec,ra)).T
k=100
Nx = 300
Ny = 300
xmin, xmax = (-30, 70)
ymin, ymax = (270, 0)

#------------------------------------------------------------
# Evaluate density
Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx),
                                            np.linspace(ymin, ymax, Ny)))).T

knn = KNeighborsDensity('simple', k)
dens = knn.fit(X).eval(Xgrid).reshape((Ny, Nx))

plt.figure(figsize=(9,9))
plt.scatter(X[:, 1], X[:, 0], s=2, lw=0, c='r')
plt.imshow(dens.T, origin='lower',
           extent=(ymin, ymax, xmin, xmax), cmap='gray_r', norm=LogNorm())

s = 'k = '+str(k)
plt.xlabel(s)
name = 'sgrstream'+str(k)+'.png'
plt.savefig(name)
plt.show()
