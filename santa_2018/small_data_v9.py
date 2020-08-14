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
num=random.sample(range(1,b ), 100)
num=[0]+num
df=cities.iloc[num]
df = df.reset_index(drop=True)
len(np.unique(df))
plt.scatter(df.X, df.Y,  c='blue',alpha=0.5,s=1)

#------------------ clustering --------------------------

from sklearn.mixture import GaussianMixture
n_cluster=5
mclusterer = GaussianMixture(n_components=n_cluster, tol=0.01, random_state=66, verbose=1).fit(df[['X', 'Y']].values)
df['mclust'] = mclusterer.predict(df[['X', 'Y']].values)+1
df.loc[0,'mclust']=0
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
'''
def nearest_point(df,mean):
    obj= []
    for i in range(len(df)) :
        obj.append(distance(df.iloc[i][['X','Y']],mean))
        dest=df.iloc[obj.index(min(obj))]['CityId']
    print('min=',dest)
    return dest
'''
def nearest_points(df,mean):
    dc=df.copy()
    dc =dc.reset_index(drop=True)
    obj= []
    for i in range(len(dc)) :
        obj.append(distance(dc.iloc[i][['X','Y']],mean))
    dc['dest']=obj
    dc=dc.sort_values(by=['dest'])
    dc =dc.reset_index(drop=True)
    #return dc.CityId[0:3]
    return dc.CityId[0],dc.CityId[1],dc.CityId[2]

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

def tol_euc(df):
    d=0
    for i in range(0,len(df)-1):    
        dis=distance(df.iloc[i][['X','Y']],df.iloc[i+1][['X','Y']])
        d+=dis
    return d    

def tol_entr(df,ent,sorti):
    d=distance(df.iloc[0][['X','Y']],ent[['X','Y']])
    for i in range(0,len(df)-1):
        dis=distance(df.iloc[i][['X','Y']],df.iloc[i+1][['X','Y']])
        if (i+1)%10==0 and df['prime'][i]==False:
            d+=1.1*dis
        else:
            d+=dis
    d+=distance(df.iloc[-1][['X','Y']],sorti[['X','Y']])       
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
    #print(i,' : ',df.CityId[0],'->',df.CityId[len(df)-1],'dist : ',told) 
    print('distance :', told )



#--------------- solver the tsp for centers ---------------------
north_pole = df[df['CityId']==0]

centers = pd.concat([centers,pd.DataFrame(columns=['start_point','s1','s2','end_point','e1','e2'])],sort=False)
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
    
    
# ------------------ get the nearest point --------------- 
for i in range(2,n_cluster+1):
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i-1].mclust)
    print ('a',a,'b',b,centers.iloc[i-1][['X','Y']])
    coord=nearest_points(df[df['mclust']==a],centers.iloc[i-1][['X','Y']])
    print (coord)
    centers.loc[i,'start_point']=coord[0]
    centers.loc[i,'s1']=coord[1]
    centers.loc[i,'s2']=coord[2]
    
for i in range(1,n_cluster): 
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i+1].mclust)
    print ('a',a,'b',b,centers.iloc[i+1][['X','Y']])
    coord=nearest_points(df[df['mclust']==a],centers.iloc[i+1][['X','Y']])
    print (coord)
    centers.loc[i,'end_point']=coord[0]
    centers.loc[i,'e1']=coord[1]
    centers.loc[i,'e2']=coord[2]

centers.loc[0,'start_point']=0
centers.loc[0,'end_point']=0
a=centers.iloc[1].mclust
coord=nearest_points(df[df['mclust']==a],north_pole[['X','Y']])
centers.loc[1,'start_point']=coord[0]
centers.loc[1,'s1']=coord[1]
centers.loc[1,'s2']=coord[2]
b=centers.iloc[n_cluster].mclust
coord=nearest_points(df[df['mclust']==b],north_pole[['X','Y']])
centers.loc[n_cluster,'end_point']=coord[0]
centers.loc[n_cluster,'e1']=coord[1]
centers.loc[n_cluster,'e2']=coord[2]


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
def two_parts(df_c,s,e):
    df_c = df_c.reset_index(drop=True)
    start=df_c[df_c['CityId']==s]
    end=df_c[df_c['CityId']==e]
    x=[float(start.X),float(end.X)]
    y=[float(start.Y),float(end.Y)] 
    p1=list(df_c.CityId).index(s)
    solve_tsp(df_c,p1)
    p2=list(df_c.CityId).index(e)
    d_1=df_c.iloc[1:p2].reset_index(drop=True)
    l=list(range(p2+1,len(df_c)))
    l.reverse()
    d_2=df_c.iloc[l]
    d_2 =d_2.reset_index(drop=True)    
    if len(d_1)>len(d_2):
        d_1=start.append(d_1).reset_index(drop=True)
        d_1=d_1.append(end).reset_index(drop=True)
        return d_1,d_2
    else:
        d_2=start.append(d_2).reset_index(drop=True)
        d_2=d_2.append(end).reset_index(drop=True)
        return d_2,d_1



def insert_pt(df,pt,pos):
    #pd.concat([dd.ix[:pos], line, dd.ix[pos+1:]]).reset_index(drop=True)
    dd=df.iloc[:pos].append(pt).reset_index(drop=True)
    dd=dd.append(df.iloc[pos:]).reset_index(drop=True)
    return dd
#-------------- mergeinto v1 ----------------------------
'''
def mergeinto(df,pt):
    obj=[]
   # print(df)
    #plt_lines(df)
    for i in range(len(df)-1):
       # obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
      # print(pt.CityId)
       #print(pt.CityId,'between',df.iloc[i].CityId,'and',df.iloc[i+1].CityId) 
       obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
    #obj.append(distance(df.iloc[-1][['X','Y']],pt))
    p=obj.index(min(obj))
    #print(p)
    if p==0:
        #print('0000000000000000000000')
        d1=d2=d3=d5=insert_pt(df,pt,p+1)
        d4=insert_pt(df,pt,p+2)
    elif p==1:  
       # print('111111111111111111111')
        d1=insert_pt(df,pt,p+1)
        d2=d5=insert_pt(df,pt,p)
        d4=insert_pt(df,pt,p+2)
        d3=insert_pt(df,pt,p)
    elif p==2:
        d1=insert_pt(df,pt,p+1)
        d3=d5=insert_pt(df,pt,p)
        d2=insert_pt(df,pt,p-1)
        d4=insert_pt(df,pt,p+2)
    else:
        d1=insert_pt(df,pt,p+1)
        d2=insert_pt(df,pt,p-1)
        d3=insert_pt(df,pt,p) 
        d4=insert_pt(df,pt,p+2)
        d5=insert_pt(df,pt,p-2)
    #print (tol(d1))
    dist=[tol_euc(d1),tol_euc(d2),tol_euc(d3),tol_euc(d4),tol_euc(d5)]
    #print (dist)
    q=dist.index(min(dist))
    #print(dist.index(min(dist)))
   # plt_lines(d1)
   # plt_lines(d2)
   # plt_lines(d3)
    if q==0:
        #plt_lines(d1)
        return d1
    elif q==1:
        #plt_lines(d2)
        return d2
    elif q==2:
        #plt_lines(d3)
        return d3
    elif q==3:
        #plt_lines(d4)
        return d4        
    else:
        #plt_lines(d5)
        return d5
    



#-------------- mergeinto v2 ----------------------------
def mergeinto(df,pt):
    obj=[]
    l=len(df)
   # print(df)
    #plt_lines(df)
    for i in range(len(df)-1):
       # obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
      # print(pt.CityId)
       #print(pt.CityId,'between',df.iloc[i].CityId,'and',df.iloc[i+1].CityId) 
       obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
    #obj.append(distance(df.iloc[-1][['X','Y']],pt))
    p=obj.index(min(obj))
    #print(p)
    print('p :',p,' len: ',l)
    if p==0 and p<l-2:
        #print('0000000000000000000000')
        d1=d2=d3=d5=insert_pt(df,pt,p+1)
        d4=insert_pt(df,pt,p+2)
    elif p==1 and p<l-1:  
       # print('111111111111111111111')
        d1=insert_pt(df,pt,p+1)
        d2=d5=insert_pt(df,pt,p)
        d4=insert_pt(df,pt,p+2)
        d3=insert_pt(df,pt,p)
    elif p==2 and p<l:
        d1=insert_pt(df,pt,p+1)
        d3=d5=insert_pt(df,pt,p)
        d2=insert_pt(df,pt,p-1)
        d4=insert_pt(df,pt,p+2)
    elif p<l:
        d1=insert_pt(df,pt,p+1)
        d2=insert_pt(df,pt,p-1)
        d3=insert_pt(df,pt,p) 
        d4=insert_pt(df,pt,p+2)
        d5=insert_pt(df,pt,p-2)        
    else:
        d1=d2=d3=d4=insert_pt(df,pt,p-1)
        d5=insert_pt(df,pt,p-2)
    #print (tol(d1))
    dist=[tol_euc(d1),tol_euc(d2),tol_euc(d3),tol_euc(d4),tol_euc(d5)]
    #print (dist)
    q=dist.index(min(dist))
    #print(dist.index(min(dist)))
   # plt_lines(d1)
   # plt_lines(d2)
   # plt_lines(d3)
    if q==0:
        #plt_lines(d1)
        print('p :',p+1,' len: ',l)
        return d1
    elif q==1:
        #plt_lines(d2)
        print('p :',p-1,' len: ',l)
        return d2
    elif q==2:
        #plt_lines(d3)
        print('p :',p,' len: ',l)
        return d3
    elif q==3:
        #plt_lines(d4)
        print('p :',p+1,' len: ',l)
        return d4        
    else:
        #plt_lines(d5)
        print('p :',p-1,' len: ',l)
        return d5

#-------------- mergeinto v3 ----------------------------
def mergeinto(df,pt):
    l=len(df)
    dc=df.iloc[range(l-1)]
    dc =dc.reset_index(drop=True)
    obj=[]
   # print (dc)    
    for i in range(l-1): 
       obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
    dc['dest']=obj
    dc=dc.sort_values(by=['dest'])
    dc =dc.reset_index(drop=True)
    r0=dc.CityId[0]
    r1=dc.CityId[1]
    aa=np.array(df['CityId'])
    p0=int(np.where(aa==r0)[0])
    p1=int(np.where(aa==r1)[0])
    #print (p0,p1)
    d1=insert_pt(df,pt,p0-1)
    d2=insert_pt(df,pt,p0)
    d3=insert_pt(df,pt,p0+1)
    d4=insert_pt(df,pt,p1-1)
    d5=insert_pt(df,pt,p1)
    d6=insert_pt(df,pt,p1+1)
    dist=[tol_euc(d1),tol_euc(d2),tol_euc(d3),tol_euc(d4),tol_euc(d5),tol_euc(d6)]
    q=dist.index(min(dist))
    if q==0:
        #plt_lines(d1)
        return d1
    if q==1:
        #plt_lines(d2)
        return d2
    if q==2:
        #plt_lines(d3)
        return d3
    if q==3:
        #plt_lines(d4)
        return d4
    if q==4:
        #plt_lines(d5)
        return d5    
    else:
        #plt_lines(d6)
        return d6
    

#-------------- mergeinto v4 ----------------------------
def mergeinto(df,pt):
    l=len(df)
    dc=df.iloc[range(l-1)]
    dc =dc.reset_index(drop=True)
    obj=[]
   # print (dc)    
    for i in range(l-1): 
       obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
    dc['dest']=obj
    dc=dc.sort_values(by=['dest'])
    dc =dc.reset_index(drop=True)
    p=[]
    j=0
    aa=np.array(df['CityId'])    
    while len(p)<4 and len(p)<l:
        r=dc.CityId[j]
        #print(r)
        #print('p0',np.where(aa==r)[0])
        p0=int(np.where(aa==r)[0])
        
        if p0 !=l and p0!=0 :
            j+=1
            p.append(p0)
            #print(p,j,l)
        else:
            j+=1
    d0=insert_pt(df,pt,p[0]) 
    d1=insert_pt(df,pt,p[1]) 
    d2=insert_pt(df,pt,p[2])
    d3=insert_pt(df,pt,p[3])
    dist=[tol_euc(d0),tol_euc(d1),tol_euc(d2),tol_euc(d3)]  
    q=dist.index(min(dist))
    #print('q',q)
    if q==3:
       # plt_lines(d3)
        return d3
    if q==1:
        #plt_lines(d1)
        return d1
    if q==2:
        #plt_lines(d2)
        return d2    
    if q==0:
        #plt_lines(d0)
        return d0
    else:
        print('faute :', q)           
    
'''
#-------------- mergeinto v5 ----------------------------
def mergeinto(df,pt,entr,sorti):
    l=len(df)
    dc=df.iloc[range(l-1)]
    dc =dc.reset_index(drop=True)
    obj=[]
   # print (dc)    
    for i in range(l-1): 
       obj.append(distance(df.iloc[i][['X','Y']],pt)+distance(df.iloc[i+1][['X','Y']],pt))
    dc['dest']=obj
    dc=dc.sort_values(by=['dest'])
    dc =dc.reset_index(drop=True)
    p=[]
    j=0
    aa=np.array(df['CityId'])    
    while len(p)<5 and len(p)<l:
        r=dc.CityId[j]
        #print(r)
        #print('p0',np.where(aa==r)[0])
        p0=int(np.where(aa==r)[0])
        
        if p0 !=l and p0!=0 :
            j+=1
            p.append(p0)
            #print(p,j,l)
        else:
            j+=1
    d0=insert_pt(df,pt,p[0]) 
    d1=insert_pt(df,pt,p[1]) 
    d2=insert_pt(df,pt,p[2])
    d3=insert_pt(df,pt,p[3])
    d4=insert_pt(df,pt,p[4])
    dist=[tol_entr(d0,entr,sorti),tol_entr(d1,entr,sorti),tol_entr(d2,entr,sorti),tol_entr(d3,entr,sorti),tol_entr(d4,entr,sorti)]  
    q=dist.index(min(dist))
    #print('q',q)
    if q==4:
       # plt_lines(d3)
        return d4
    if q==3:
       # plt_lines(d3)
        return d3
    if q==1:
        #plt_lines(d1)
        return d1
    if q==2:
        #plt_lines(d2)
        return d2    
    if q==0:
        #plt_lines(d0)
        return d0
    else:
        print('faute :', q)           
     

    

#----------------------------------------- ex ---------------------------------
'''
tour=1
m=int(centers.iloc[tour].mclust)
df_o=df[df['mclust']==m]
df_o = df_o.reset_index(drop=True)
solve_tsp(df_o,2)
plt_lines(df_o)
d1,d2=two_parts(df_o,centers['s1'][tour],centers['e1'][tour])
plt_lines(d1)
plt_lines(d2)
for i in range(len(d2)):
    print(i)
    d1=mergeinto(d1,d2.iloc[i],north_pole,df[df['CityId']==40462])
    plt_lines(d1)
i=7    

plt_lines(insert_pt(d1,d2.iloc[i],p[2]))
plt_lines(d1)
'''





'''
#----------------------------best_s_e v1-----------------------------------------
def best_s_e(df,cent):
    l_s=['start_point','s1','s2']
    l_e=['end_point','e1','e2']
    dd = pd.DataFrame(columns=['start_point','s1','s2'],index=['end_point','e1','e2'])
    for s in l_s:
        for e in l_e:
            #print ('start',s,' end',e)
            d1,d2=two_parts(df,cent[s],cent[e])
            #print(d1.iloc[0].CityId,d1.iloc[-1].CityId)
            #print('d1',len(d1),' d2 ',len(d2))
            for i in range(len(d2)):
                d1=mergeinto(d1,d2.iloc[i])
            #print(d1.iloc[0].CityId,d1.iloc[-1].CityId)    
            #plt_lines(d1)    
            dd.loc[e,s]=tol(d1)
            #print(dd)
    aa=np.array(dd)
    #print('aa',aa)
    re=np.where(aa==aa.min())
    #print('re',re)
    start= dd.columns[int(re[1])]
    end= dd.index[int(re[0])]
    #print('YO',start,end)
    d1,d2=two_parts(df,cent[start],cent[end])
    for i in range(len(d2)):
        d1=mergeinto(d1,d2.iloc[i])
    plt_lines(d1)    
    return d1



l_s=['start_point','s1','s2']
l_e=['end_point','e1','e2']
dd = pd.DataFrame(columns=['start_point','s1','s2'],index=['end_point','e1','e2'])
for s in l_s:
    for e in l_e:
        d1,d2=two_parts(df_o,centers[s][tour],centers[e][tour])
        for i in range(len(d2)):
            d1=mergeinto(d1,d2.iloc[i])
        dd.loc[e,s]=tol(d1) 
        plt_lines(d1)
aa=np.array(dd)
re=np.where(aa==aa.min())
s= dd.columns[int(re[1])]
e= dd.index[int(re[0])] 
d1,d2=two_parts(df_o,centers[s][tour],centers[e][tour])
for i in range(len(d2)):
    d1=mergeinto(d1,d2.iloc[i])       
    plt_lines(d1)

'''



#----------------------------best_s_e v2-----------------------------------------
def best_s_e(df,cent,entr,sorti,ls,le):
    dd = pd.DataFrame(columns=ls,index=le)
    for s in ls:
        for e in le:
            #print ('start',s,' end',e)
            d1,d2=two_parts(df,cent[s],cent[e])
            #print(d1.iloc[0].CityId,d1.iloc[-1].CityId)
            #print('d1',len(d1),' d2 ',len(d2))
            for i in range(len(d2)):
                d1=mergeinto(d1,d2.iloc[i],entr,sorti)
            #print(d1.iloc[0].CityId,d1.iloc[-1].CityId)    
            #plt_lines(d1)    
            dd.loc[e,s]=tol(d1)
           # print(dd)
    aa=np.array(dd)
    #print('aa',aa)
    re=np.where(aa==aa.min())
    #print('re',re)
    start= dd.columns[int(re[1])]
    end= dd.index[int(re[0])]
    #print('YO',start,end)
    d1,d2=two_parts(df,cent[start],cent[end])
    for i in range(len(d2)):
        d1=mergeinto(d1,d2.iloc[i],entr,sorti)
    plt_lines(d1)    
    return d1
#------------------------------------------------------------------------------

'''
tour=1
m=int(centers.iloc[tour].mclust)
df_o=df[df['mclust']==m]
df_o = df_o.reset_index(drop=True)
solve_tsp(df_o,2)
plt_lines(df_o)
yo=best_s_e(df_o,centers.iloc[tour],north_pole,north_pole,['start_point'],l_e)


plt_lines(yo)

'''


#------------------------------------------------------------------------------




tour=1
l_s=['start_point','s1','s2']
l_e=['end_point','e1','e2']
m=int(centers.iloc[tour].mclust)
df_o=df[df['mclust']==m]
sorti=centers.iloc[tour].start_point
dd=best_s_e(df_o,centers.iloc[1],north_pole,df[df['CityId']==int(sorti)],l_s,l_e)
ent=dd.iloc[-1]

for i in range(2,n_cluster):
    m=int(centers.iloc[i].mclust)
    print('m= ',m)
    df_o=df[df['mclust']==m]
    df_o = df_o.reset_index(drop=True)
    sorti=centers.iloc[i+1].start_point
    dc=best_s_e(df_o,centers.iloc[i],ent,df[df['CityId']==int(sorti)],l_s,l_e)
    dd=dd.append(dc,ignore_index=True,sort=False)
    ent=dd.iloc[-1]
    plt_lines(dd)


m=int(centers.iloc[n_cluster].mclust)
df_o=df[df['mclust']==m]
df_o = df_o.reset_index(drop=True)
sorti=centers.iloc[n_cluster].start_point
dc=best_s_e(df_o,centers.iloc[n_cluster],ent,df[df['CityId']==int(sorti)],l_s,l_e)
dd=dd.append(dc,ignore_index=True,sort=False)
plt_lines(dd)

solve_tsp(df,0)

plt_lines(df)

df=df.append(north_pole,ignore_index=True,sort=False)
plt_lines(df)



tol(df)
tol(dd)

'''
la=pd.DataFrame( columns=df.columns)
for i in range(n_cluster):
    la=la.append(cities[cities['CityId']==centers.iloc[i].start_point],ignore_index=True)
    la=la.append(cities[cities['CityId']==centers.iloc[i].end_point],ignore_index=True)
plt_lines(la)    
'''
print ('df :',tol(df),' d_com : ',tol(dd))
