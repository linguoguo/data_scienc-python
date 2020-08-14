#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:49:26 2019

@author: lin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 19:45:06 2019

@author: lin
"""
from matplotlib.collections import LineCollection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
plt.style.use('seaborn-white')
import seaborn as sns
import random
import math
from concorde.tsp import TSPSolver
import time
import pylab as pl
#---------------- import data ----------------------------
cities =pd.read_csv('cities.csv')
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
b=len(cities)
num=random.sample(range(1,b ), 156)
num.append(0)
df=cities.iloc[num]
df = df.reset_index(drop=True)
len(np.unique(df))
plt.scatter(df.X, df.Y,  c='blue',alpha=0.5,s=1)

#------------------ clustering --------------------------

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

    
plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y,  color=colors,alpha=0.5,s=5)
for i in range(n_cluster):
    plt.scatter(centers.iloc[i].X, centers.iloc[i].Y, c='black', s=50)    
plt.show()

# ------------------ get the nearest point ---------------

def distance(a, b):
    return math.sqrt((b.X - a.X) ** 2 + (a.Y - b.Y) ** 2)

def nearest_point(df,mean):
    obj= []
    for i in range(len(df)) :
        obj.append(distance(df.iloc[i][['X','Y']],mean))
        dest=df.iloc[obj.index(min(obj))]['CityId']
    print('min=',dest)
    return dest

north_pole = df[df['CityId']==0]
n_p=north_pole[['mclust','X','Y']]
route=n_p.append(centers,ignore_index=True)
route=route.append(n_p,ignore_index=True)
centers = pd.concat([centers,pd.DataFrame(columns=['start_point','end_point'])],sort=False)
solver = TSPSolver.from_data(
    route.X,
    route.Y,
    norm="EUC_2D"
)

tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42)
centers_c=centers.copy() 
for i in range(len(centers)):
    print (tour_data[0][i+1]-1)
    centers.iloc[i]=centers_c.iloc[tour_data[0][i+1]-1]
 
for i in range(1,n_cluster):
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i-1].mclust)
    print ('a',a,'b',b)
    coord=nearest_point(df[df['mclust']==a],centers.iloc[b][['X','Y']])
    centers.loc[i,'start_point']=coord
    
for i in range(0,n_cluster-1): 
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i+1].mclust)
    print ('a',a,'b',b)
    coord=nearest_point(df[df['mclust']==a],centers.iloc[b][['X','Y']])
    centers.loc[i,'end_point']=coord

a=centers.iloc[0].mclust
coord=nearest_point(df[df['mclust']==a],north_pole[['X','Y']])
centers.loc[0,'start_point']=coord

b=centers.iloc[n_cluster-1].mclust
coord=nearest_point(df[df['mclust']==b],north_pole[['X','Y']])
centers.loc[n_cluster-1,'end_point']=coord 

plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y,  color=colors,alpha=0.5,s=5)
for i in range(n_cluster):
    plt.scatter(centers.iloc[i].X, centers.iloc[i].Y, c='black', s=50) 
    start=df[df['CityId']==centers.iloc[i].start_point]
    end=df[df['CityId']==centers.iloc[i].end_point]
    plt.scatter(start.X,start.Y,c='red',s=50)
    plt.scatter(end.X,end.Y,c='blue',s=50)
plt.show()

lines=[]
lines.append([(north_pole.X,north_pole.Y),(centers.X[0],centers.Y[0])])
for i in range(0,n_cluster-1):   
    print (i)
    lines.append([(centers.X[i],centers.Y[i]),(centers.X[i+1],centers.Y[i+1])]) 
lines.append([(north_pole.X,north_pole.Y),(centers.X[n_cluster-1],centers.Y[n_cluster-1])])
lc = LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(15,10))
#cities.plot.scatter(x='X', y='Y', s=0.07)
ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.scatter(df.X, df.Y,  color=colors,alpha=0.5,s=1)
plt.scatter(centers.X, centers.Y, c='black', s=15)
for i in range(n_cluster): 
    start=df[df['CityId']==centers.iloc[i].start_point]
    end=df[df['CityId']==centers.iloc[i].end_point]
    plt.scatter(start.X,start.Y,c='green',s=20)
    plt.scatter(end.X,end.Y,c='blue',s=20)
ax.add_collection(lc)
ax.autoscale()
plt.show()

def to_end(df,p):
    x=df.iloc[len(df)-1]
    df.iloc[len(df)-1]=df.iloc[p]
    df.iloc[p]=x
    
    
def to_begin(df,p):
    x=df.iloc[0]
    df.iloc[0]=df.iloc[p]
    df.iloc[p]=x 
    
i=0 
df_c=df[df['mclust']==i]    
df_c = df_c.reset_index(drop=True)
a=int(centers[centers['mclust']==i].start_point)    
b=int(centers[centers['mclust']==i].end_point) 
p1=list(df_c.CityId).index(a)
p2=list(df_c.CityId).index(b)
to_begin(df_c,p1)
to_end(df_c,p2)

#---------------- solve the tsp ----------------------------
solver = TSPSolver.from_data(
    df_c.X,
    df_c.Y,
    norm="EUC_2D"
)

tour_c = solver.solve(time_bound = 60.0, verbose = True, random_seed = 44) 
df_r=df_c.copy() 
for i in range(len(df_c)):
    df_c.iloc[i]=df_r.iloc[tour_c[0][i]]


lines_c = [[(df_c.X[j],df_c.Y[j]),(df_c.X[j+1],df_c.Y[j+1])]for j in range(0,len(df_c)-1)]
lc = LineCollection(lines_c, linewidths=2)
fig, ax = pl.subplots(figsize=(8,5))
#ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.scatter(df_c.X, df_c.Y,  color='grey',alpha=0.5,s=20)
plt.scatter(df_c[df_c['CityId']==a].X,df_c[df_c['CityId']==a].Y,c='black',s=20)
plt.scatter(df_c[df_c['CityId']==b].X,df_c[df_c['CityId']==b].Y,c='red',s=20)
#plt.scatter(centers.X, centers.Y, c='black', s=15) 
ax.add_collection(lc)
ax.autoscale()
plt.show()

tol=0
df_f=df.copy()
for i in range(n_cluster):
    print(i)
    df_c=df[df['mclust']==i]    
    df_c = df_c.reset_index(drop=True)
    n=len(df_c)
    print (n)
    a=int(centers[centers['mclust']==i].start_point)    
    b=int(centers[centers['mclust']==i].end_point) 
    p1=list(df_c.CityId).index(a)
    p2=list(df_c.CityId).index(b)
    to_begin(df_c,p1)
    to_end(df_c,p2)
    solver = TSPSolver.from_data(
            df_c.X,
            df_c.Y,
            norm="EUC_2D"
            )
    
    tour_c = solver.solve(time_bound = 60.0, verbose = True, random_seed = 44) 
    for j in range(n):
                df_f.iloc[j+tol]=df_c.iloc[tour_c[0][j]]
    tol+=n
    print (tol)
    