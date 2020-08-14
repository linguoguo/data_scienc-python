#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:56:46 2019

@author: lin
"""

cities = pd.read_csv('cities.csv')
#cities.drop(['CityId'],axis=1,inplace=True)
north_pole = cities[cities.CityId==0]
prime_cities = eratosthenes(max(cities.CityId))
cities['prime'] = prime_cities

df=cities[prime_cities]
plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y, s=1, alpha=0.4,c = 'grey')
plt.show()

#--------- seperate the image into 2 parts

up=cities[cities['Y']>2000]
plt.figure(figsize=(15,5))
plt.scatter(up.X,up.Y,s=1,alpha=0.4,c='grey')
plt.show()

down=cities[cities['Y']<=2000]
plt.figure(figsize=(15,5))
plt.scatter(down.X,down.Y,s=1,alpha=0.4,c='grey')
plt.show()

clustering = DBSCAN(eps=40, min_samples=40).fit(up[['X', 'Y']].values)
labels = clustering.labels_
up['label']=labels
plt.figure(figsize=(8,3))
plt.scatter(up[up['label']==0].X, up[up['label']==0].Y, c = 'grey', s = 1)
plt.show()
        
d_up_1=up[up['label']==0]
d_up_2=up[up['label']!=0]
#df.drop(['B', 'C'], axis=1)
plt.figure(figsize=(8,3))
plt.scatter(d_up_2.X, d_up_2.Y, c = 'grey', s = 1)
plt.show()

colors = np.where(labels%4==0,'#630C3A','-')
colors[labels%4==1] = '#39C8C6'
colors[labels%4==2] = '#D3500C'
colors[labels%4==3] = '#FFB139'

clustering = DBSCAN(eps=50, min_samples=10).fit(down[['X', 'Y']].values)
labels = clustering.labels_
down['label']=labels
plt.figure(figsize=(8,3))
plt.scatter(down[down['label']==0].X, down[down['label']==0].Y, c = 'grey', s = 1)
plt.show()

d_down_0=down[down['label']==0]
d_down_3=down[down['label']!=0]
d_down_0 = d_down_0.reset_index(drop=True)
plt.figure(figsize=(8,3))
plt.scatter(d_down_1.X, d_down_1.Y, c = 'grey', s = 1)
plt.scatter(d_down_3.X, d_down_3.Y, c = 'green', s = 1)
plt.show()


clustering = DBSCAN(eps=30, min_samples=10).fit(d_down_0[['X', 'Y']].values)
labels = clustering.labels_
d_down_0['label']=labels
plt.figure(figsize=(8,3))
plt.scatter(d_down_0[d_down_0['label']==0].X, d_down_0[d_down_0['label']==0].Y, c = 'grey', s = 1)
plt.show()

d_down_1=d_down_0[d_down_0['label']==0]
d_down_2=d_down_0[d_down_0['label']!=0]
plt.figure(figsize=(8,3))
plt.scatter(d_down_1.X, d_down_1.Y, c = 'grey', s = 1)
plt.scatter(d_down_2.X, d_down_2.Y, c = 'green', s = 1)
plt.show()


'''

plt.figure(figsize=(15,5))
plt.scatter(d_down_1.X, d_down_1.Y, c = 'grey', s = 1)
plt.scatter(d_down_2.X, d_down_2.Y, c = 'green', s = 1)
plt.scatter(d_down_3.X, d_down_3.Y, c = 'red', s = 1)
plt.scatter(dd.X, dd.Y, c = 'black', s = 1)
plt.show()
''''
#******************************************************************************
#-------------------   up clustering  -----------------------------------------
#******************************************************************************
# Number of clusters
kmeans = KMeans(300)
# Fitting the input data
kmeans = kmeans.fit(d_up_1[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(d_up_1[['X', 'Y']].values)
d_up_1['mclust'] = labels+1
d_up_1.loc[0,'mclust']=0
centers = d_up_1.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
plt.scatter(d_up_1.X, d_up_1.Y,  color=colors,alpha=0.5,s=1)
plt.show() 
plt.figure(figsize=(8, 3))
plt.scatter(d_up_1.X, d_up_1.Y,  color=colors,alpha=0.5,s=5)
for i in range(300):
    plt.scatter(centers.iloc[i].X, centers.iloc[i].Y, c='black', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
plt.show()
#------------------------------------------------------------------------------
# Number of clusters
kmeans = KMeans(100)
# Fitting the input data
kmeans = kmeans.fit(d_up_2[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(d_up_2[['X', 'Y']].values)
d_up_2['mclust'] = labels+301
centers_tmp = d_up_2.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
plt.figure(figsize=(8, 5))
plt.scatter(d_up_2.X, d_up_2.Y,  color=colors,alpha=0.5,s=1)
plt.show() 

plt.figure(figsize=(8, 3))
plt.scatter(d_up_2.X, d_up_2.Y,  color=colors,alpha=0.5,s=5)
for i in range(100):
    plt.scatter(centers_tmp.iloc[i].X, centers_tmp.iloc[i].Y, c='black', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
plt.show()


centers_up=centers.append(centers_tmp,ignore_index=True,sort=False)


plt.figure(figsize=(15, 5))
plt.scatter(up.X, up.Y,  color='grey',alpha=0.5,s=1)
for i in range(300):
    plt.scatter(centers.iloc[i].X, centers.iloc[i].Y, c='red', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
for i in range(100):
    plt.scatter(centers_tmp.iloc[i].X, centers_tmp.iloc[i].Y, c='black', s=50)    
plt.show()

#--------------- solver the tsp for centers_up ---------------------

centers_up = pd.concat([centers_up,pd.DataFrame(columns=['start_point','s1','s2','end_point','e1','e2'])],sort=False)
solver = TSPSolver.from_data(
    centers_up.X,
    centers_up.Y,
    norm="EUC_2D"
)

tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42)
centers_up_c=centers_up.copy() 
for i in range(len(centers_up)):
    #print (tour_data[0][i])
    centers_up.iloc[i]=centers_up_c.iloc[tour_data[0][i]]
plt_lines(centers_up)    

#******************************************************************************
#-------------------  down clustering  -----------------------------------------
#******************************************************************************


# Number of clusters
kmeans = KMeans(100)
# Fitting the input data
kmeans = kmeans.fit(d_down_3[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(d_down_3[['X', 'Y']].values)
d_down_3['mclust'] = labels+401
centers_down = d_down_3.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
plt.scatter(d_down_3.X, d_down_3.Y,  color=colors,alpha=0.5,s=1)
plt.show() 
plt.figure(figsize=(8, 3))
plt.scatter(d_down_3.X, d_down_3.Y,  color=colors,alpha=0.5,s=5)
for i in range(100):
    plt.scatter(centers_down.iloc[i].X, centers_down.iloc[i].Y, c='black', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
plt.show()
#------------------------------------------------------------------------------

# Number of clusters
kmeans = KMeans(100)
# Fitting the input data
kmeans = kmeans.fit(d_down_2[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(d_down_2[['X', 'Y']].values)
d_down_2['mclust'] = labels+501
centers_down_2 = d_down_2.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
plt.scatter(d_down_2.X, d_down_2.Y,  color=colors,alpha=0.5,s=1)
plt.show() 
plt.figure(figsize=(8, 3))
plt.scatter(d_down_2.X, d_down_2.Y,  color=colors,alpha=0.5,s=5)
for i in range(100):
    plt.scatter(centers_down_2.iloc[i].X, centers_down_2.iloc[i].Y, c='black', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
plt.show()
#------------------------------------------------------------------------------
# Number of clusters
kmeans = KMeans(300)
# Fitting the input data
kmeans = kmeans.fit(d_down_1[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(d_down_1[['X', 'Y']].values)
d_down_1['mclust'] = labels+501
centers_down_1 = d_down_1.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
plt.scatter(d_down_1.X, d_down_1.Y,  color=colors,alpha=0.5,s=1)
plt.show() 
plt.figure(figsize=(8, 3))
plt.scatter(d_down_1.X, d_down_1.Y,  color=colors,alpha=0.5,s=5)
for i in range(300):
    plt.scatter(centers_down_1.iloc[i].X, centers_down_1.iloc[i].Y, c='black', s=50) 
    #plt.scatter(zeros[0],zeros[1],c='green',s=50)
plt.show()
#------------------------------------------------------------------------------

centers_tol=centers_up.append(centers_down,ignore_index=True,sort=False)
centers_tol=centers_tol.append(centers_down_2,ignore_index=True,sort=False)
centers_tol=centers_tol.append(centers_down_1,ignore_index=True,sort=False)
centers_copy=centers_tol.copy()
#--------------- solver the tsp for centers ---------------------

centers_tol = pd.concat([centers_tol,pd.DataFrame(columns=['start_point','s1','s2','end_point','e1','e2'])],sort=False)
solver = TSPSolver.from_data(
    centers_tol.X,
    centers_tol.Y,
    norm="EUC_2D"
)

tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42)
centers_tol_c=centers_tol.copy() 
for i in range(len(centers_tol)):
    #print (tour_data[0][i])
    centers_tol.iloc[i]=centers_tol_c.iloc[tour_data[0][i]]


# ------------------ get the new cities  --------------- 
new_cities=d_up_1
new_cities=new_cities.append(d_up_2,ignore_index=True,sort=False) 
new_cities=new_cities.append(d_down_1,ignore_index=True,sort=False)  
new_cities=new_cities.append(d_down_2,ignore_index=True,sort=False) 
new_cities=new_cities.append(d_down_3,ignore_index=True,sort=False)  
new_cities = pd.concat([new_cities,pd.DataFrame(columns=['visited'])],sort=False)
# ------------------ get the nearest point --------------- 
for i in range(64,901):
    print(i)
    a=int(centers_tol.iloc[i].mclust)
    b=int(centers_tol.iloc[i-1].mclust)
    #print ('a',a,'b',b,centers_tol.iloc[i-1][['X','Y']])
    coord=nearest_points(new_cities[new_cities['mclust']==a],centers_tol.iloc[i-1][['X','Y']])
    print (coord)
    print(len(coord))

    if len(coord)==3:
        centers_tol.loc[i,'start_point']=coord[0]
        centers_tol.loc[i,'s1']=coord[1]
        centers_tol.loc[i,'s2']=coord[2]
    if len(coord)==3:
        centers_tol.loc[i,'start_point']=coord[0]
        centers_tol.loc[i,'s1']=coord[1]
    else:
        centers_tol.loc[i,'start_point']=coord[0]
for i in range(1,900): 
    a=int(centers_tol.iloc[i].mclust)
    b=int(centers_tol.iloc[i+1].mclust)
    #print ('a',a,'b',b,centers_tol.iloc[i+1][['X','Y']])
    coord=nearest_points(new_cities[new_cities['mclust']==a],centers_tol.iloc[i+1][['X','Y']])
    print (coord)
    if len(coord)==3:
        centers_tol.loc[i,'end_point']=coord[0]
        centers_tol.loc[i,'e1']=coord[1]
        centers_tol.loc[i,'e2']=coord[2]
    if len(coord)==2:
        centers_tol.loc[i,'end_point']=coord[0]
        centers_tol.loc[i,'e1']=coord[1]
    else:
        centers_tol.loc[i,'end_point']=coord[0]        
        

centers_tol.loc[0,'start_point']=0
centers_tol.loc[0,'end_point']=0
a=centers_tol.iloc[1].mclust
coord=nearest_points(new_cities[new_cities['mclust']==a],north_pole[['X','Y']])
centers_tol.loc[1,'start_point']=coord[0]
centers_tol.loc[1,'s1']=coord[1]
centers_tol.loc[1,'s2']=coord[2]
b=centers_tol.iloc[900].mclust
coord=nearest_points(new_cities[new_cities['mclust']==b],north_pole[['X','Y']])
centers_tol.loc[n_cluster,'end_point']=coord[0]
centers_tol.loc[n_cluster,'e1']=coord[1]
centers_tol.loc[n_cluster,'e2']=coord[2]


#-------- first cluster ------------------------------------
tour=1
m=int(centers_tol.iloc[tour].mclust)
df_o=new_cities[new_cities['mclust']==m]
n_cluster=int(len(df_o)/35)
kmeans = KMeans(n_cluster)
# Fitting the input data
kmeans = kmeans.fit(df_o[['X', 'Y']].values)
# Getting the cluster labels
labels = kmeans.predict(df_o[['X', 'Y']].values)
df_o['mclust'] = labels
centers = df_o.groupby('mclust')['X', 'Y'].agg('mean').reset_index()

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

for i in range(1,n_cluster):
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i-1].mclust)
    #print ('a',a,'b',b,centers.iloc[i-1][['X','Y']])
    coord=nearest_points(df_o[df_o['mclust']==a],centers.iloc[i-1][['X','Y']])
    print (coord)
    centers.loc[i,'start_point']=coord[0]
    centers.loc[i,'s1']=coord[1]
    centers.loc[i,'s2']=coord[2]
    
for i in range(0,n_cluster-1): 
    a=int(centers.iloc[i].mclust)
    b=int(centers.iloc[i+1].mclust)
    #print ('a',a,'b',b,centers.iloc[i+1][['X','Y']])
    coord=nearest_points(df_o[df_o['mclust']==a],centers.iloc[i+1][['X','Y']])
    #print (coord)
    centers.loc[i,'end_point']=coord[0]
    centers.loc[i,'e1']=coord[1]
    centers.loc[i,'e2']=coord[2]
centers.loc[0,'start_point']=0
a=centers.iloc[1].mclust
coord=nearest_points(df_o[df_o['mclust']==a],north_pole[['X','Y']])
centers.loc[1,'start_point']=coord[0]
centers.loc[1,'s1']=coord[1]
centers.loc[1,'s2']=coord[2]

b=centers.iloc[n_cluster-1].mclust
coord=nearest_points(df_o[df_o['mclust']==b],centers_tol.iloc[tour+1][['X','Y']])
centers.loc[n_cluster-1,'end_point']=coord[0]
centers.loc[n_cluster-1,'e1']=coord[1]
centers.loc[n_cluster-1,'e2']=coord[2]




def ls(centers_tol,i):
    b=len([centers[['start_point','s1','s2']].iloc[i].isna()==True])
    if b==1:
        return['start_point']
    if b==2:
        return['start_point','s1']
    else:
        return ['start_point','s1','s2']
    
def le(centers_tol,i):
    b=len([centers[['end_point','e1','e2']].iloc[i].isna()==True])
    if b==1:
        return['end_point']
    if b==2:
        return['end_point','e1']
    else:
        return ['end_point','e1','e2']      

def solve_clust(df,centers_tol,tour):
    m=int(centers_tol.iloc[tour].mclust)
    df_o=df[df['mclust']==m]
    n_cluster=int(len(df_o)/35)
    kmeans = KMeans(n_cluster)
    kmeans = kmeans.fit(df_o[['X', 'Y']].values)
    # Getting the cluster labels
    labels = kmeans.predict(df_o[['X', 'Y']].values)
    df_o['mclust'] = labels+1
    centers = df_o.groupby('mclust')['X', 'Y'].agg('mean').reset_index()

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
    for i in range(1,n_cluster):
        a=int(centers.iloc[i].mclust)
        b=int(centers.iloc[i-1].mclust)
        #print ('a',a,'b',b,centers.iloc[i-1][['X','Y']])
        coord=nearest_points(df_o[df_o['mclust']==a],centers.iloc[i-1][['X','Y']])
        print (coord)
        centers.loc[i,'start_point']=coord[0]
        centers.loc[i,'s1']=coord[1]
        centers.loc[i,'s2']=coord[2]    
    for i in range(0,n_cluster-1): 
        a=int(centers.iloc[i].mclust)
        b=int(centers.iloc[i+1].mclust)
        #print ('a',a,'b',b,centers.iloc[i+1][['X','Y']])
        coord=nearest_points(df_o[df_o['mclust']==a],centers.iloc[i+1][['X','Y']])
        #print (coord)
        centers.loc[i,'end_point']=coord[0]
        centers.loc[i,'e1']=coord[1]
        centers.loc[i,'e2']=coord[2]
    centers.loc[0,'start_point']=0
    a=centers.iloc[1].mclust
    coord=nearest_points(df_o[df_o['mclust']==a],centers_tol.iloc[tour-1][['X','Y']])
    centers.loc[1,'start_point']=coord[0]
    centers.loc[1,'s1']=coord[1]
    centers.loc[1,'s2']=coord[2]

    b=centers.iloc[n_cluster-1].mclust
    coord=nearest_points(df_o[df_o['mclust']==b],centers_tol.iloc[tour+1][['X','Y']])
    centers.loc[n_cluster-1,'end_point']=coord[0]
    centers.loc[n_cluster-1,'e1']=coord[1]
    centers.loc[n_cluster-1,'e2']=coord[2]    

    for i in range(0,n_cluster):
        m=int(centers.iloc[i].mclust)
        print('m= ',m)
        df_c=df_o[df_o['mclust']==m]
        df_c = df_c.reset_index(drop=True)
        sorti=centers.iloc[i+1].start_point
        prime=final['visited'].sum()
        dc=best_s_e(df_c,centers.iloc[i],ent,df[df['CityId']==int(sorti)],l_s,l_e,prime)
        dc['visited']=1
        final=final.append(dc,ignore_index=True,sort=False)
        ent=dd.iloc[-1]
        plt_lines(dd)    
    

for i in range(2,10):
      solve_clust(new_cities,centers_tol,i)