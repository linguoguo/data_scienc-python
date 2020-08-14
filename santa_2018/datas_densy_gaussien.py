#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 05:55:09 2019

@author: lin
"""

from functions import *

cities =pd.read_csv('cities.csv')
prime_cities = eratosthenes(max(cities.CityId))
cities['prime'] = prime_cities

df=cities[(cities['X']<110) & (cities['Y']>3100) &(cities['Y']<3250)]
df = df.reset_index(drop=True)
to_begin(df,5)

df = pd.concat([df,pd.DataFrame(columns=['visited'])],sort=False)
north_pole=df[df['CityId']==4529]
plt.scatter(df.X, df.Y,  c='blue',alpha=0.5,s=1)


#n_cluster=20
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

#--------------- solver the tsp for centers ---------------------
north_pole['visited']=1

centers = pd.concat([centers,pd.DataFrame(columns=['start_point','s1','s2','end_point','e1','e2'])],sort=False)
solver = TSPSolver.from_data(
    centers.X,
    centers.Y,
    norm="EUC_2D"
)

tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42)
centers_c=centers.copy() 
for i in range(len(centers)):
    #print (tour_data[0][i])
    centers.iloc[i]=centers_c.iloc[tour_data[0][i]]
    
    
# ------------------ get the nearest point --------------- 
for i in range(2,n_cluster+1):
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i-1].mclust)
    #print ('a',a,'b',b,centers.iloc[i-1][['X','Y']])
    coord=nearest_points(df[df['mclust']==a],centers.iloc[i-1][['X','Y']])
    #print (coord)
    centers.loc[i,'start_point']=coord[0]
    centers.loc[i,'s1']=coord[1]
    centers.loc[i,'s2']=coord[2]
    
for i in range(1,n_cluster): 
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i+1].mclust)
    #print ('a',a,'b',b,centers.iloc[i+1][['X','Y']])
    coord=nearest_points(df[df['mclust']==a],centers.iloc[i+1][['X','Y']])
    #print (coord)
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
    #print (i)
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

tour=1
l_s=['start_point','s1','s2']
l_e=['end_point','e1','e2']
m=int(centers.iloc[tour].mclust)
df_o=df[df['mclust']==m]
sorti=centers.iloc[tour].start_point
prime=1
dc=best_s_e(df_o,centers.iloc[1],north_pole,df[df['CityId']==int(sorti)],l_s,l_e,prime)
dc['visited']=1
dd=north_pole.append(dc,ignore_index=True,sort=False)
ent=dd.iloc[-1]
dd['visited']=1

for i in range(2,n_cluster):
    m=int(centers.iloc[i].mclust)
    print('m= ',m)
    df_o=df[df['mclust']==m]
    df_o = df_o.reset_index(drop=True)
    sorti=centers.iloc[i+1].start_point
    prime=dd['visited'].sum()
    dc=best_s_e(df_o,centers.iloc[i],ent,df[df['CityId']==int(sorti)],l_s,l_e,prime)
    dc['visited']=1
    dd=dd.append(dc,ignore_index=True,sort=False)
    ent=dd.iloc[-1]
    plt_lines(dd)


m=int(centers.iloc[n_cluster].mclust)
df_o=df[df['mclust']==m]
df_o = df_o.reset_index(drop=True)
sorti=centers.iloc[n_cluster].start_point
prime=dd['visited']=1
dc=best_s_e(df_o,centers.iloc[n_cluster],ent,df[df['CityId']==int(sorti)],l_s,l_e,prime)
dc['visited']=1
dd=dd.append(dc,ignore_index=True,sort=False)
dd=dd.append(north_pole,ignore_index=True,sort=False)

mingzi1='photo/nc_dd_'+str(n_cluster)+'.png'
mingzi2='photo/nc_tsp_'+str(n_cluster)+'.png'

plt_lines(dd,name=mingzi1)
solve_tsp(df,0)
df=df.append(north_pole,ignore_index=True,sort=False)
plt_lines(df,name=mingzi2)

print('dist tsp', tol(df,0),'\n')
print('dist tsp prime', tol(dd,0),'\n')





