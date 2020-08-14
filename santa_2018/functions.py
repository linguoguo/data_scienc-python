#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 05:53:26 2019

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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

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

def distance(a, b):
    return math.sqrt((b.X - a.X) ** 2 + (a.Y - b.Y) ** 2)

def nearest_points(df,mean):
    dc=df.copy()
    dc =dc.reset_index(drop=True)
    obj= []
    for i in range(len(dc)) :
        obj.append(distance(dc.iloc[i][['X','Y']],mean))
    dc['dest']=obj
    dc=dc.sort_values(by=['dest'])
    dc =dc.reset_index(drop=True)
    return dc.CityId.iloc[0:min(3,len(dc))]
   # return dc.CityId[0],dc.CityId[1],dc.CityId[2]

def to_end(df,p):
    x=df.iloc[len(df)-1]
    df.iloc[len(df)-1]=df.iloc[p]
    df.iloc[p]=x
        
def to_begin(df,p):
    x=df.iloc[0]
    df.iloc[0]=df.iloc[p]
    df.iloc[p]=x 
    
def tol(df,p):
    d=0
    for i in range(0,len(df)-1):
        dis=distance(df.iloc[i][['X','Y']],df.iloc[i+1][['X','Y']])
        p+=1
        if (p)%10==0 and df['prime'][i]==False:
            d+=1.1*dis
        else:
            d+=dis
    return d
  
def tol_entr(df,ent,sorti,p):
    d=distance(df.iloc[0][['X','Y']],ent[['X','Y']])
    for i in range(0,len(df)-1):
        dis=distance(df.iloc[i][['X','Y']],df.iloc[i+1][['X','Y']])
        p+=1
        if (p)%10==0 and df['prime'][i]==False:
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

def plt_lines(df,**kwargs):
    name=kwargs.get('name', None)
    lines_c = [[(df.X[m],df.Y[m]),(df.X[m+1],df.Y[m+1])]for m in range(0,len(df)-1)]
    lc = LineCollection(lines_c, linewidths=2)
    fig, ax = pl.subplots(figsize=(8,5))

    #ax.scatter(north_pole.X, north_pole.Y, c='red', s=15)
    plt.scatter(df.X, df.Y,  color='grey',alpha=0.5,s=20)
    plt.scatter(df.X[0],df.Y[0],  color='red',alpha=0.5,s=20)
   # plt.scatter(df.X[len(df)],df.Y[len(df)],  color='blue',alpha=0.5,s=20)
    ax.add_collection(lc)
    ax.autoscale()
    told=tol(df,0)
    if name is not None:
        print(name)
        plt.savefig(name)
        file = open('testfile.txt','a+') 
        file.write(name) 
        file.write('\n') 
        file.write(str(told)) 
        file.write('\n')
        file.close() 
    
    plt.show()
        
    #print(i,' : ',df.CityId[0],'->',df.CityId[len(df)-1],'dist : ',told) 
    print('distance :', told )
    
#

# ------------------------cut a cluster into 2 parts --------------------------
'''
def two_parts(df_c,s,e):
    df_c = df_c.reset_index(drop=True)
    start=df_c[df_c['CityId']==s]
    end=df_c[df_c['CityId']==e]
    #print(end)
    p1=list(df_c.CityId).index(s)
    solve_tsp(df_c,p1)
    #print(df_c)
    
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

'''
def two_parts(df_c,s,e):
    df_c = df_c.reset_index(drop=True)
    start=df_c[df_c['CityId']==s]
    p1=list(df_c.CityId).index(s)
    a=len(df_c)
    l2=list(np.linspace(0,a-1,a))*2
    ll=l2[p1:p1+a]
    liste = [int(i) for i in ll]
    df_c=df_c.iloc[liste]
    df_c = df_c.reset_index(drop=True)
    b=df_c[df_c['CityId']==e].index
   # print(df_c.iloc[b])
    end=df_c.iloc[b]
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
'''
s=66544
e=185848
solve_tsp(df_o,0)
two_parts(df_o,s,e)
'''







def insert_pt(df,pt,pos):
    #pd.concat([dd.ix[:pos], line, dd.ix[pos+1:]]).reset_index(drop=True)
    if pos<len(df):
        dd=df.iloc[:pos].append(pt).reset_index(drop=True)
        dd=dd.append(df.iloc[pos:]).reset_index(drop=True)
        return dd
    else:
        return df
        
    

    
#-------------- mergeinto v5 ----------------------------
def mergeinto(df,pt,entr,sorti,prime):
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
    while len(p)<5 and j<l-1:        
        r=dc.CityId[j]
       # print('r',r,' j ',j)
       # print('p0',np.where(aa==r)[0])
        p0=int(np.where(aa==r)[0])
       # print(p0)
        if p0!=0:
            j+=1
            p.append(p0)
           # print(p)
        else:
            j+=1
    p=p*5 
    #print(p)       
    d0=insert_pt(df,pt,p[0]) 
    d1=insert_pt(df,pt,p[1]) 
    d2=insert_pt(df,pt,p[2])
    d3=insert_pt(df,pt,p[3])
    d4=insert_pt(df,pt,p[4])
    dist=[tol_entr(d0,entr,sorti,prime),tol_entr(d1,entr,sorti,prime),tol_entr(d2,entr,sorti,prime),tol_entr(d3,entr,sorti,prime),tol_entr(d4,entr,sorti,prime)]  
    #print(dist)
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
        
#----------------------------best_s_e v2-----------------------------------------
def best_s_e(df,cent,entr,sorti,ls,le,prime):
    dd = pd.DataFrame(columns=ls,index=le)
    solve_tsp(df,0)
    for s in ls:
        for e in le:
            #print ('start',s,' end',e)
            d1,d2=two_parts(df,cent[s],cent[e])
            #print(d1.iloc[0].CityId,d1.iloc[-1].CityId)
            #print('d1',len(d1),' d2 ',len(d2))
            if not d2.empty:
                for i in range(len(d2)):
                    d1=mergeinto(d1,d2.iloc[i],entr,sorti,prime)
            #print(d1.iloc[0].CityId,d1.iloc[-1].CityId)    
            #plt_lines(d1)    
            dd.loc[e,s]=tol(d1,prime)
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
        d1=mergeinto(d1,d2.iloc[i],entr,sorti,prime)
    plt_lines(d1)    
    return d1        