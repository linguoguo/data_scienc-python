#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:29:06 2019

@author: lin
"""

path_figures=report_path+'/reduce_samples/figures/'

from classe.Weekend_days import *
red=weekday(days)

def reduc(days):
    dif = []


    for day in days:
        print('day : ',day)
        wd = weekday([day])
        if day == 1:
            dif.append(1)
        elif wd[0]=='Monday' or wd[0]=='Tuesday' or wd[0]=='Wednesday' or wd[0]=='Thursday':

            dif.append(((day+4)//7)*2+2)
            print('new day :' ,wd[0],((day+4)//7)*2+2)
        else:

            dif.append(((day+4)//7)*2+1)


            print('new day :',wd[0], ((day+4)//7)*2+1)
    return dif


#redwk=reduc(days)

reduc_list = [reduc(data[choices].values[i][:].tolist()) for i in range(5000)]
reduc_list_df= pd.DataFrame(reduc_list,columns=choices)
reduc_list_df['n_people']=data['n_people']
familly_difficile=pd.read_csv(mug_data_path+'difficult_famillys.csv', index_col='family_id') 
reduc_list_df['difficulty']=familly_difficile['difficulty']


samples = reduc_list_df.sample(frac =.5) 
samples.to_csv(mug_data_path+'sample_reduc_2_d.csv') 


plt.figure(figsize=(16,9))
sns.countplot(samples['n_people'])
plt.title('familly size')
plt.savefig(path_figures+'fammily_size_2.png')


choice_0_f8=samples['choice_0'][samples['n_people']==8]
plt.figure(figsize=(18,10))
sns.countplot(choice_0_f8)
plt.title('choice 0 of 8p familly')
plt.savefig(path_figures+'choice_0_f8_2.png')


choice_0_f7=samples['choice_0'][samples['n_people']==7]
plt.figure(figsize=(18,10))
sns.countplot(choice_0_f7)
plt.title('choice 0 of 7p familly')
plt.savefig(path_figures+'choice_0_f7_2.png')

choice_0_f6=samples['choice_0'][samples['n_people']==6]
plt.figure(figsize=(18,10))
sns.countplot(choice_0_f6)
plt.title('choice 0 of 6p familly')
plt.savefig(path_figures+'choice_0_f6_2.png')


plt.figure(figsize=(16,9))
sns.countplot(samples['difficulty'])
plt.title('difficult famillys')
plt.savefig(path_figures+'difficult_famillys_2.png')

