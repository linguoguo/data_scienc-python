#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:36:28 2019

@author: lin
"""

"""--------------- report file : np_choice ------------------------"""

data.head()

'''
x = np.array([2,3,1])
y = np.array([2,8,1])
z = np.vdot(x,y)
print(z) # 
'''




n_tol=data['n_people'].sum()
n_max=data['n_people'].max()
n_min=data['n_people'].min()

plt.figure(figsize=(16,9))
sns.countplot(data['n_people'])

choice_0_f8=data['choice_0'][data['n_people']==8]

plt.figure(figsize=(18,10))
sns.countplot(choice_0_f8)

choice_0_f7=data['choice_0'][data['n_people']==7]

plt.figure(figsize=(18,10))
sns.countplot(choice_0_f7)

choice_0_f6=data['choice_0'][data['n_people']==6]

plt.figure(figsize=(18,10))
sns.countplot(choice_0_f6)

""" --------------nb of ppl by choices ------------"""
choices=['days','choice_0','choice_1','choice_2','choice_3','choice_4','choice_5','choice_6','choice_7','choice_8','choice_9']
    
pByChoice= pd.DataFrame(np.zeros((100,11)),columns=choices)
pByChoice['days']=days


for c in range(10):
    choice=choices[c+1]
    print(choice)
    for i in range(5000):
        j= data.iloc[i][choice] 
        print('familly',i,j)
        pByChoice[choice][pByChoice['days']==j]+=data.iloc[i]['n_people']    
pByChoice.to_csv(mug_data_path+'pByChoice.csv')        

    
    
for c in range(10):
    choice=choices[c+1]
    print(choice) 
    plt.figure(figsize=(18,10))
    sns.barplot(x='days',y=choice,data=pByChoice.iloc[1:-1][['days',choice]])


