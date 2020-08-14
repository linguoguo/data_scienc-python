#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:20:39 2019

@author: lin
"""

test=cities[0:10]
solver = TSPSolver.from_data(
    test.X,
    test.Y,
    norm="EUC_2D"
)

tour = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42) # solv
test['tour']=tour.tour
lines = [[(test.X[tour.tour[i]],test.Y[tour.tour[i]]),(test.X[tour.tour[i+1]],test.Y[tour.tour[i+1]])] for i in range(0,len(test)-1)]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(15,10))
#cities.plot.scatter(x='X', y='Y', s=0.07)
#ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.scatter(test.X, test.Y,  c='blue',alpha=0.5,s=1)
#plt.scatter(centers.X, centers.Y, c='black', s=15)
ax.add_collection(lc)
ax.autoscale()
plt.show()

def distance(a, b):
    return math.sqrt((b.X - a.X) ** 2 + (a.Y - b.Y) ** 2)

def tol(df):
    d=0
    for i in range(0,len(df)-1):
        if (i+2)%10==0:
            d+=1.1*distance(df.iloc[df.tour[i]][['X','Y']],df.iloc[df.tour[i+1]][['X','Y']])
        else:
            d+=distance(df.iloc[df.tour[i]][['X','Y']],df.iloc[df.tour[i+1]][['X','Y']])
    return d
        
tol(test)        

def to_end(df,p):
    x=df.iloc[len(df)-1]
    df.iloc[len(df)-1]=df.iloc[p]
    df.iloc[p]=x


to_end(test,7)    


test2=cities[10:20]
solver = TSPSolver.from_data(
    test2.X,
    test2.Y,
    norm="EUC_2D"
)

tour2 = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42) # solv
test2['tour']=tour2.tour
test2 = test2.reset_index(drop=True)
lines = [[(test2.X[tour2.tour[i]],test2.Y[tour2.tour[i]]),(test2.X[tour2.tour[i+1]],test2.Y[tour2.tour[i+1]])] for i in range(0,len(test2)-1)]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(15,10))
#cities.plot.scatter(x='X', y='Y', s=0.07)
#ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.scatter(test2.X, test2.Y,  c='blue',alpha=0.5,s=1)
#plt.scatter(centers.X, centers.Y, c='black', s=15)
ax.add_collection(lc)
ax.autoscale()
plt.show()


to_end(test2,2)
tol(test2) 
