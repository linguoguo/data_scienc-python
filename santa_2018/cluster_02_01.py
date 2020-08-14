# -*- coding: utf-8 -*-
"""
Lin GUO

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
plt.style.use('seaborn-white')
import seaborn as sns

cities =pd.read_csv('cities.csv')
'''
cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
north_pole = cities[cities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()
'''
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

print(cities.head())


plt.figure(figsize=(15, 10))
plt.scatter(cities[cities['prime'] == False].X, cities[cities['prime'] == False].Y, s=1, alpha=0.4,c = 'grey')
plt.scatter(cities[cities['prime'] == True].X, cities[cities['prime'] == True].Y, s=1, alpha=0.6, c='blue')
plt.scatter(cities.iloc[0: 1, 1], cities.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.title('Visualisation of cities')
plt.show()



density_cities=sns.jointplot(cities.X, cities.Y, kind="hex", color="#4CB391")
plt.title('Density of cities')
plt.show()

density_primes=sns.jointplot(cities[cities['prime'] == True].X, cities[cities['prime']== True].Y, kind="hex", color="#a6bddb")

 

print('the number of prime citys :' , cities[cities['prime']==True].X.count(), 'among', cities['X'].count())
print('cities/prime cities',cities[cities['prime']==False].X.count()/cities[cities['prime']==True].X.count())

#--------- seperate the image into 2 parts

up=cities[cities['Y']>2000]
plt.figure(figsize=(15,5))
plt.scatter(up.X,up.Y,s=1,alpha=0.4,c='grey')
plt.show()

down=cities[cities['Y']<=2000]
plt.figure(figsize=(15,5))
plt.scatter(down.X,down.Y,s=1,alpha=0.4,c='grey')
plt.show()


'''
#------------------  clustering by density  -------------------------------------------

from sklearn.cluster import DBSCAN
#tol=[10,20,30,40,50,60,70,80,90]
esp=[10,20,30,40,50,60,70,80,90]
samples=[10,20,30,40,50,60,70,80,90]


for i in esp :
    for j in samples :
        print('i',i,'j',j)
        name='down'+str(i)+'_'+str(j)+'.png'
        print (name)
        clustering = DBSCAN(eps=i, min_samples=j).fit(down[['X', 'Y']].values)
        labels = clustering.labels_
        down['label']=labels
        plt.figure(figsize=(15,5))
        plt.scatter(down[down['label']==0].X, down[down['label']==0].Y, c = 'grey', s = 1)
        plt.savefig(name)
        plt.close()
        
#----------------------------------------------------------------------------        
''' 

'''
# -------archives---------------      
cdict = {0: 'red', 1: 'blue', 2: 'green',3:'grey'}

for i in esp :
    for j in samples :
        print('i',i,'j',j)
        name='upup'+str(i)+'_'+str(j)+'.png'
        print (name)
        clustering = DBSCAN(eps=i, min_samples=j).fit(up[['X', 'Y']].values)
        plt.figure(figsize=(15, 5))
        plt.scatter(up.X, up.Y, c=clustering.labels_)
        plt.savefig(name)
        plt.close()
        
 

 

       
clustering = DBSCAN(eps=30, min_samples=20).fit(cities[['X', 'Y']].values)
labels = clustering.labels_
#labels[np.nonzero(labels)]=-1
cities['label']=labels
cdict = {0: 'red', 1: 'blue', 2: 'green',3:'grey'}
centers = cities.groupby('label')['X', 'Y'].agg('mean').reset_index()
fig, ax = plt.subplots()
for g in np.unique(labels):
    centers['counts'][g]=cities[cities['label']==g].X.count()
    print(cities[cities['label']==g].X.count())
    ax.scatter(cities[cities['label']==g].X, cities[cities['label']==g].Y, c = cdict[g%4], label = g, s = 1)
ax.legend()
plt.show()



plt.figure(figsize=(15,10))
plt.scatter(cities[cities['label']==0].X, cities[cities['label']==0].Y, c = 'grey', label = g, s = 1)
plt.show()


    fig=plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    # Set gridsize just to make them visually large
    image = plt.hexbin(cities.X,cities.Y,gridsize=50)
    # Note that mincnt=1 adds 1 to each count
    counts = image.get_array()
    ncnts = np.count_nonzero(np.power(10,counts))
    verts = image.get_offsets()
    for offc in xrange(verts.shape[0]):
        binx,biny = verts[offc][0],verts[offc][1]
        if counts[offc]:
            plt.plot(binx,biny,'k.',zorder=100)
    ax.set_xlim(xbnds)
    ax.set_ylim(ybnds)
    plt.grid(True)
    cb = plt.colorbar(image,spacing='uniform',extend='max')
    plt.show()


#---------------------
'''

n_cluster=200
from sklearn.mixture import GaussianMixture
mclusterer = GaussianMixture(n_components=n_cluster, tol=0.01, random_state=66, verbose=1).fit(cities[['X', 'Y']].values)

cities['mclust'] = mclusterer.predict(cities[['X', 'Y']].values)

centers = cities.groupby('mclust')['X', 'Y'].agg('mean').reset_index()

plt.figure(figsize=(15, 10))
plt.scatter(centers.X, centers.Y,  color='darkblue',alpha=0.5,s=1)
plt.scatter(cities.iloc[0: 1, 1], cities.iloc[0: 1, 2], s=10, c="red")
plt.show()

clust_c=['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
         
colors = np.where(cities["mclust"]%4==0,'#630C3A','-')
colors[cities['mclust']%4==1] = '#39C8C6'
colors[cities['mclust']%4==2] = '#D3500C'
colors[cities['mclust']%4==3] = '#FFB139'       

plt.figure(figsize=(15, 10))
plt.scatter(cities.X, cities.Y,  color=colors,alpha=0.5,s=1)
plt.show() 

centers['nb_prime']=cities[cities['prime']==1].mclust.value_counts()
centers['nb_no_prime']=cities[cities['prime']==0].mclust.value_counts()


print('number of cities by cluster','\n',nb_non_prime_cluster)
print('number of prime cities by cluster','\n',centers['nb_prime'])
print('cities/prime cities','\n',nb_non_prime_cluster/nb_prime_cluster)



       
#sns.scatterplot( x="X", y="Y",data=cities,hue='mclust') 

clust_c=['#630C3A', '#39C8C6', '#D3500C', '#FFB139']


cities['groupe_c']=cities['mclust']%4
colors = np.where(cities["c"]==0,'r','-')   
colors[cities["c"]==1] = 'b'  
colors[cities["c"]==2] = 'g'  
colors[cities["c"]==3] = '#39C8C6'        
      

sns.scatterplot( x="X", y="Y", data=cities)    

df = pd.DataFrame(np.random.normal(10,1,30).reshape(10,3), index = pd.date_range('2010-01-01', freq = 'M', periods = 10), columns = ('one', 'two', 'three'))
df['key1'] = (4,4,4,6,6,6,8,8,8,8)
colors = np.where(df["key1"]==4,'r','-')
colors[df["key1"]==6] = 'g'
colors[df["key1"]==8] = 'b'
print(colors)
df.plot.scatter(x="one",y="two",c=colors)
plt.show()

cluster_plot=plt.scatter(cities.X, cities.Y,alpha=0.5,s=1,c=clust_c);
for i in range(n_cluster):
    plt.setp(cluster_plot, color = clust_c[i%3])
plt.show()   



'''