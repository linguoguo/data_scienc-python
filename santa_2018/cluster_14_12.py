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

df=cities
plt.figure(figsize=(15, 10))
plt.scatter(df[df['prime'] == False].X, df[df['prime'] == False].Y, s=1, alpha=0.4,c = 'grey')
plt.scatter(df[df['prime'] == True].X, df[df['prime'] == True].Y, s=1, alpha=0.6, c='blue')
plt.scatter(df.iloc[0: 1, 1], df.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.title('Visualisation of cities')
plt.show()



density_cities=sns.jointplot(cities.X, cities.Y, kind="hex", color="#4CB391")
plt.title('Density of cities')
plt.show()

density_primes=sns.jointplot(cities[cities['prime'] == True].X, cities[cities['prime']== True].Y, kind="hex", color="#a6bddb")

 

print('the number of prime citys :' , cities[cities['prime']==True].X.count(), 'among', cities['X'].count())

from sklearn.mixture import GaussianMixture
mclusterer = GaussianMixture(n_components=35, tol=0.01, random_state=66, verbose=1).fit(cities[['X', 'Y']].values)

cities['mclust'] = mclusterer.predict(cities[['X', 'Y']].values)

centers = cities.groupby('mclust')['X', 'Y'].agg('mean').reset_index()

plt.figure(figsize=(15, 10))
plt.scatter(centers.X, centers.Y,  color='darkblue',alpha=0.5,s=1)
plt.scatter(cities.iloc[0: 1, 1], cities.iloc[0: 1, 2], s=10, c="red")
plt.show()