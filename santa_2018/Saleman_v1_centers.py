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
import math

cities = pd.read_csv('cities.csv')
#cities.drop(['CityId'],axis=1,inplace=True)
north_pole = cities[cities.CityId==0]

# Load the prime numbers we need in a set with the Sieve of Eratosthenes
def eratosthenes(n):
    P = [True for i in range(n+1)]
    P[0], P[1] = False, False
    p = 2
    l = np.sqrt(n)
    while p < l:
        if P[p]:
            for i in range(2*p, n+1, p):
                P[i] = False
        p += 1
        
    return P

# find cities that are prime numbers
prime_cities = eratosthenes(max(cities.CityId))
cities['prime'] = prime_cities

df=cities[prime_cities]
plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y, s=1, alpha=0.4,c = 'grey')
plit.show()

#******************************************************************************
#------------------- nearest point to cluster ---------------------------------
#******************************************************************************

def distance(a, b):
    return math.sqrt((b.X - a.X) ** 2 + (a.Y - b.Y) ** 2)

def nearest_point(df,mean,dest):
    obj= []
    for i in range(len(df)) :
        obj.append(distance(df.iloc[i][['X','Y']],mean))
        dest=df.iloc[obj.index(min(obj))]['CityId']
    print('min=',dest)
    return dest

i=2
nearest_point(df[df['mclust']==i],centers.iloc[i+1],0)

#******************************************************************************
#------------------- gaussian mixture -----------------------------------------
#******************************************************************************
from sklearn.mixture import GaussianMixture
n_cluster=4
mclusterer = GaussianMixture(n_components=n_cluster, tol=0.01, random_state=66, verbose=1).fit(df[['X', 'Y']].values)
df['mclust'] = mclusterer.predict(df[['X', 'Y']].values)
centers = df.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
clust_c=['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
colors = np.where(df["mclust"]%4==0,'#630C3A','-')
colors[df['mclust']%4==1] = '#39C8C6'
colors[df['mclust']%4==2] = '#D3500C'
colors[df['mclust']%4==3] = '#FFB139' 

t=nearest_point(df[df['mclust']==0],centers.iloc[1][['X','Y']],0)
test_point = df[df['CityId']==t][['X','Y']]     
plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y,  color=colors,alpha=0.5,s=1)
plt.scatter(test_point.X, test_point.Y, c='red', s=50)
for i in range(n_cluster):
    plt.scatter(centers.iloc[i].X, centers.iloc[i].Y, c='black', s=50)    
plt.show()
 
#------------------------------------------------------------------------------
#******************************************************************************
#-------------------       k-means    -----------------------------------------
#******************************************************************************
from sklearn.cluster import KMeans
n_cluster=3
# Number of clusters
kmeans = KMeans(n_cluster)
# Fitting the input data
kmeans = kmeans.fit(df[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(df[['X', 'Y']].values)
# Centroid values
centroids = kmeans.cluster_centers_
clust_c=['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
colors = np.where(labels%4==0,'#630C3A','-')
colors[labels%4==1] = '#39C8C6'
colors[labels%4==2] = '#D3500C'
colors[labels%4==3] = '#FFB139'
test_point = cities[cities.CityId==1214]
plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y,  color=colors,alpha=0.5,s=1)
plt.scatter(test_point.X, test_point.Y, c='red', s=50)
plt.show() 
#------------------------------------------------------------------------------

#******************************************************************************
#-------------------       sort       -----------------------------------------
#******************************************************************************
df.sort_values(by=['mclust'], inplace=True)
#reset index (not sure it's useful)
df2 = df.reset_index(drop=True)
#------------------------------------------------------------------------------






##-------------------    TPS   ----------------------------------------
centers=north_pole[['X','Y']]
c1 = df.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
c1.drop(['mclust'],axis=1,inplace=True)

centers=centers.append(c1,ignore_index=True)
centers=centers.append(north_pole[['X','Y']],ignore_index=True)
# add a few columns
centers = pd.concat([centers,pd.DataFrame(columns=['start_point','end_point'])],sort=False)
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
fig, ax = pl.subplots(figsize=(15,10))
#cities.plot.scatter(x='X', y='Y', s=0.07)
north_pole = cities[cities.CityId==0]
ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
ax.add_collection(lc)
ax.autoscale()
plt.show()
#------------------------------------------------------------------------------






##-------------------    TPS corrige   ----------------------------------------
n_p=north_pole[['X','Y']]
n_p = pd.concat([pd.DataFrame(columns=['mclust']),n_p],sort=False)
#centers.drop(['mclust'],axis=1,inplace=True)

route=n_p.append(centers,ignore_index=True)
route=route.append(n_p,ignore_index=True)
# add a few columns
route = pd.concat([route,pd.DataFrame(columns=['tour_data','start_point','end_point'])],sort=False)
# Instantiate solver
solver = TSPSolver.from_data(
    route.X,
    route.Y,
    norm="EUC_2D"
)

t = time.time()
tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42) # solve() doesn't seem to respect time_bound for certain values?
route['tour_data']=tour_data[0]
route.sort_values(by=['tour_data'], inplace=True)
route2 = route.reset_index(drop=True)

print(time.time() - t)
print(tour_data.found_tour)

lines = [[(route.X[tour_data.tour[i]],route.Y[tour_data.tour[i]]),(route.X[tour_data.tour[i+1]],route.Y[tour_data.tour[i+1]])] for i in range(0,len(route)-1)]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(15,10))
#cities.plot.scatter(x='X', y='Y', s=0.07)
ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
ax.add_collection(lc)
ax.autoscale()
plt.show()
#------------------------------------------------------------------------------