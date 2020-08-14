#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:12:55 2018

@author: lin
"""

from concorde.tsp import TSPSolver
from matplotlib import collections  as mc
import numpy as np
import pandas as pd
import time
import pylab as pl
import matplotlib.pyplot as plt

cities = pd.read_csv('cities.csv')
#cities.drop(['CityId'],axis=1,inplace=True)
north_pole = cities[cities.CityId==0]
n_cluster=20
from sklearn.mixture import GaussianMixture
mclusterer = GaussianMixture(n_components=n_cluster, tol=0.01, random_state=66, verbose=1).fit(cities[['X', 'Y']].values)

cities['mclust'] = mclusterer.predict(cities[['X', 'Y']].values)
centers=north_pole[['X','Y']]
c1 = cities.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
c1.drop(['mclust'],axis=1,inplace=True)
centers=centers.append(c1,ignore_index=True)
centers=centers.append(north_pole[['X','Y']],ignore_index=True)
# Instantiate solver
solver = TSPSolver.from_data(
    centers.X,
    centers.Y,
    norm="EUC_2D"
)

t = time.time()
tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42) # solve() doesn't seem to respect time_bound for certain values?
print(time.time() - t)
print(tour_data.found_tour)

lines = [[(centers.X[tour_data.tour[i]],centers.Y[tour_data.tour[i]]),(centers.X[tour_data.tour[i+1]],centers.Y[tour_data.tour[i+1]])] for i in range(0,len(centers)-1)]
lc = mc.LineCollection(lines, linewidths=2)
#fig, ax = pl.subplots(figsize=(15,10))
fig=plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
cities.plot.scatter(x='X', y='Y', s=0.07)
north_pole = cities[cities.CityId==0]
ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
ax.add_collection(lc)
ax.autoscale()
plt.show()
