#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:31:07 2019

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
zeros=pd.DataFrame.from_dict({'X':[cities['X'].mean()],'Y':[cities['Y'].mean()]})

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

    
plt.figure(figsize=(8, 5))
plt.scatter(df.X, df.Y,  color=colors,alpha=0.5,s=5)
for i in range(n_cluster):
    plt.scatter(centers.iloc[i].X, centers.iloc[i].Y, c='black', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
plt.show()



def distance(a, b):
    return math.sqrt((b.X - a.X) ** 2 + (a.Y - b.Y) ** 2)

def nearest_point(df,mean):
    obj= []
    for i in range(len(df)) :
        obj.append(distance(df.iloc[i][['X','Y']],mean))
        dest=df.iloc[obj.index(min(obj))]['CityId']
    print('min=',dest)
    return dest

def to_end(df,p):
    x=df.iloc[len(df)-1]
    df.iloc[len(df)-1]=df.iloc[p]
    df.iloc[p]=x
        
def to_begin(df,p):
    x=df.iloc[0]
    df.iloc[0]=df.iloc[p]
    df.iloc[p]=x 
    
def tol(df):
    d=0
    for i in range(0,len(df)-1):
        dis=distance(df.iloc[i][['X','Y']],df.iloc[i+1][['X','Y']])
        if (i+1)%10==0 and df['prime'][i]==False:
            d+=1.1*dis
        else:
            d+=dis
    return d

def solve_tsp(df,start_pt):
    df_r=df.copy()
    to_begin(df_r,start_pt)
    solver = TSPSolver.from_data(
            df_r.X,
            df_r.Y,
    norm="EUC_2D"
    )
    tour_c = solver.solve(time_bound = 60.0, verbose = True, random_seed = 44) 
    for j in range(len(df)):
        df.iloc[j]=df_r.iloc[tour_c[0][j]]     

def plt_lines(df):
    lines_c = [[(df.X[m],df.Y[m]),(df.X[m+1],df.Y[m+1])]for m in range(0,len(df)-1)]
    lc = LineCollection(lines_c, linewidths=2)
    fig, ax = pl.subplots(figsize=(8,5))
    #ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
    plt.scatter(df.X, df.Y,  color='grey',alpha=0.5,s=20)
    plt.scatter(df.X[0],df.Y[0],  color='red',alpha=0.5,s=20)
   # plt.scatter(df.X[len(df)],df.Y[len(df)],  color='blue',alpha=0.5,s=20)
    ax.add_collection(lc)
    ax.autoscale()
    plt.show()
    told=tol(df)    
    print(i,' : ',df.CityId[0],'->',df.CityId[len(df)-1],'dist : ',told)  



#--------------- solver the tsp for centers ---------------------
north_pole = df[df['CityId']==0]
n_index=north_pole.mclust
to_begin(centers,int(n_index))

centers = pd.concat([centers,pd.DataFrame(columns=['start_point','end_point'])],sort=False)
solver = TSPSolver.from_data(
    centers.X,
    centers.Y,
    norm="EUC_2D"
)

tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42)
centers_c=centers.copy() 
for i in range(len(centers)):
    print (tour_data[0][i])
    centers.iloc[i]=centers_c.iloc[tour_data[0][i]]
    

'''
    for j in range(len(df)):
        df.iloc[j]=df_r.iloc[tour_c[0][j]]  


'''
    
# ------------------ get the nearest point --------------- 
for i in range(1,n_cluster):
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i-1].mclust)
    print ('a',a,'b',b)
    coord=nearest_point(df[df['mclust']==a],centers.iloc[i-1][['X','Y']])
    centers.loc[i,'start_point']=coord
    
for i in range(0,n_cluster-1): 
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i+1].mclust)
    print ('a',a,'b',b)
    coord=nearest_point(df[df['mclust']==a],centers.iloc[i+1][['X','Y']])
    centers.loc[i,'end_point']=coord

a=centers.iloc[0].mclust
coord=nearest_point(df[df['mclust']==a],north_pole[['X','Y']])
centers.loc[0,'start_point']=coord

b=centers.iloc[n_cluster-1].mclust
coord=nearest_point(df[df['mclust']==b],north_pole[['X','Y']])
centers.loc[n_cluster-1,'end_point']=coord 

plt.figure(figsize=(8, 5))
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
fig, ax = pl.subplots(figsize=(8,5))
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


# -------------------- solver the tsp by clusters -----------------------------
  
'''    
i=2 
df_c=df[df['mclust']==i]
df_c = df_c.reset_index(drop=True)
solve_tsp(df_c,3)
plt_lines(df_c)
'''

# ------------------------cut a cluster into 2 parts --------------------------
#up_down=0  : up
#up_down=1  : down    
           
'''
def two_parts(df,centers,tour,):
    m=int(centers.iloc[tour].mclust)
    df_c=df[df['mclust']==m]
    df_c = df_c.reset_index(drop=True)
    df_c = pd.concat([df_c,pd.DataFrame(columns=['up_down'])],sort=False)
    start=df_c[df_c['CityId']==centers.iloc[tour].start_point]
    end=df_c[df_c['CityId']==centers.iloc[tour].end_point]
    x=[float(start.X),float(end.X)]
    y=[float(start.Y),float(end.Y)]
    a=int(start.CityId)
    b=int(end.CityId) 
    p1=list(df_c.CityId).index(a)
    solve_tsp(df_c,p1)
    p2=list(df_c.CityId).index(b)
    dis1=distance(zeros,df_c.iloc[0][['X','Y']])
    dis2=distance(zeros,df_c.iloc[1][['X','Y']])
    print(dis1,dis2)
    if dis2>dis1:
     #   d_in= pd.DataFrame( columns=df_c.columns)        
        d_out=df_c.iloc[0:p2+1]
        d_out['up_down']=0
        d_in=df_c.iloc[p2+1:]
        d_in['up_down']=1
        d_in =d_in.reset_index(drop=True)
    else:  
        aa=list(range(1,p2))
        aa.reverse()
        d_in=df_c.iloc[aa]
        d_in =d_in.reset_index(drop=True)
        d_in['up_down']=1
        d_out=start
        bb=list(range(p2,len(df_c)))
        bb.reverse()
        dd=df_c.iloc[bb]
        dd =dd.reset_index(drop=True)
        d_out=d_out.append(dd,ignore_index=True,sort=False) 
        d_out['up_down']=0
        d_out =d_out.reset_index(drop=True) 
    return d_out,d_in
#    return d_up.append(d_down,ignore_index=True)               

'''

def two_parts(df,centers,tour,):
    m=int(centers.iloc[tour].mclust)
    df_c=df[df['mclust']==m]
    df_c = df_c.reset_index(drop=True)
    df_c = pd.concat([df_c,pd.DataFrame(columns=['up_down'])],sort=False)
    start=df_c[df_c['CityId']==centers.iloc[tour].start_point]
    end=df_c[df_c['CityId']==centers.iloc[tour].end_point]
    x=[float(start.X),float(end.X)]
    y=[float(start.Y),float(end.Y)]
    a=int(start.CityId)
    b=int(end.CityId) 
    p1=list(df_c.CityId).index(a)
    solve_tsp(df_c,p1)
    p2=list(df_c.CityId).index(b)
    d_out=df_c.iloc[0:p2+1]
    d_out['up_down']=0
    d_in=df_c.iloc[p2+1:]
    d_in['up_down']=1
    d_in =d_in.reset_index(drop=True)    
    return d_out,d_in
    
    
    

'''
tour=1
m=int(centers.iloc[tour].mclust)
df_o=df[df['mclust']==m]
df_o = df_o.reset_index(drop=True)
solve_tsp(df_o,23)
plt_lines(df_o)
d1,d2=two_parts(df,centers,tour)
plt_lines(d1)
plt_lines(d2)
plt_lines(d1.append(d2,ignore_index=True))
'''

df_up= pd.DataFrame( columns=df.columns)
df_down= pd.DataFrame( columns=df.columns)
for i in range(n_cluster):
    d1,d2=two_parts(df,centers,i)
    df_up=df_up.append(d1,ignore_index=True,sort=False)
    df_down=d2.append(df_down,ignore_index=True,sort=False)


solve_tsp(df,156)
plt_lines(df_up)
plt_lines(df_down)
dd=df_up.append(df_down,ignore_index=True,sort=False)
dd=dd.append(north_pole,ignore_index=True,sort=False)
plt_lines(dd)

df=df.append(north_pole,ignore_index=True,sort=False)
plt_lines(df)



tol(df)
tol(dd)


la=pd.DataFrame( columns=df.columns)
for i in range(n_cluster):
    la=la.append(cities[cities['CityId']==centers.iloc[i].start_point],ignore_index=True)
    la=la.append(cities[cities['CityId']==centers.iloc[i].end_point],ignore_index=True)
plt_lines(la)    

print ('df :',tol(df),' d_com : ',tol(dd))
