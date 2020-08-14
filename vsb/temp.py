# -*- coding: utf-8 -*-
"""
Lin GUO

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv('data_subsets/0_6.csv') 
meta_train = pd.read_csv('metadata_train.csv')


def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

sig_fault=meta_train[meta_train['target']==1].index

s_id = 0
p1,p2,p3 = phase_indices(s_id)
plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(train.iloc[:,p1],marker="o", linestyle="none")
plt.plot(train.iloc[:,p2])
plt.plot(train.iloc[:,p3])

plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(train.iloc[:,p1],marker="o", linestyle="none")
plt.plot(train.iloc[:,p2],marker="o", linestyle="none")
plt.plot(train.iloc[:,p3],marker="o", linestyle="none")

f, axarr = plt.subplots(10, sharex=True,figsize=(10,15))
n=0
for i in range(0,10):
   # print(train.iloc[:,subset[i]:subset[i+1]].head())
    name='data_fault/'+str(sig_fault[n+i])+'.csv'
    print(name)
    #print(df.tail(2))
    df=pd.read_csv(name)   
    axarr[i].plot(df,marker="o", linestyle="none")
