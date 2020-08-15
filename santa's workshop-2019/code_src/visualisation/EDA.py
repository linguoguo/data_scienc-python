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


path_figures=report_path+'/EDA/figures/'

n_tol=data['n_people'].sum()
n_max=data['n_people'].max()
n_min=data['n_people'].min()


plt.figure(figsize=(16,9))
sns.countplot(data['n_people'])
plt.title('familly size')
plt.savefig(path_figures+'fammily_size.png')





choice_0_f8=data['choice_0'][data['n_people']==8]
plt.figure(figsize=(18,10))
sns.countplot(choice_0_f8)
plt.title('choice 0 of 8p familly')
plt.savefig(path_figures+'choice_0_f8.png')


choice_0_f7=data['choice_0'][data['n_people']==7]
plt.figure(figsize=(18,10))
sns.countplot(choice_0_f7)
plt.title('choice 0 of 7p familly')
plt.savefig(path_figures+'choice_0_f7.png')

choice_0_f6=data['choice_0'][data['n_people']==6]
plt.figure(figsize=(18,10))
sns.countplot(choice_0_f6)
plt.title('choice 0 of 6p familly')
plt.savefig(path_figures+'choice_0_f6.png')

""" --------------nb of ppl by choices ------------"""

       
pByChoice=pd.read_csv(mug_data_path+'pByChoice.csv')        

    

for c in choices:
    print(c) 
    plt.figure(figsize=(25,10))
    sns.barplot(x='days',y=c,data=pByChoice.iloc[0:-1][['days',c]])
    plt.title('number of ppl of '+c+' -eve')
    plt.savefig(path_figures+'npl_'+c+'_withouEve.png')
    

for c in choices:
    print(c) 
    plt.figure(figsize=(25,10))
    sns.barplot(x='days',y=c,data=pByChoice[['days',c]])
    plt.title('number of ppl of '+c)
    plt.savefig(path_figures+'npl_'+c+'.png')    

"""
n1=npl_init(data_diff,b[1]) 
plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=n1[1:])

   
"""    

