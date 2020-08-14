# -*- coding: utf-8 -*-
"""
Lin GUO

"""

import pandas as pd
import pyarrow.parquet as pq
import os
import csv
import numpy as np

train = pq.read_pandas('train.parquet').to_pandas()
train.info()

subset=np.linspace(0,8712,1453, dtype = int)

for i in range(0,10):
   # print(train.iloc[:,subset[i]:subset[i+1]].head())
    df=train.iloc[:,subset[i]:subset[i+1]]  
    name='data_subsets/'+str(subset[i])+'_'+str(subset[i+1])+'.csv'
    print(name)
    df.to_csv(name,index=False)
    
cc=pd.read_csv('data_subsets/12_24.csv')    
print(cc.head())



#    get the power lines fault
meta_train = pd.read_csv('metadata_train.csv')
sig_fault=meta_train[meta_train['target']==1].id_measurement.unique()

def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

    
for i in range(30,60):
    p1,p2,p3 = phase_indices(sig_fault[i])
    print(p1,p2,p3)
    df=train.iloc[:,[p1,p2,p3] ] 
    print(df.head())
    name='data_fault/'+str(sig_fault[i])+'.csv'
    print(name)
  #  print(df.tail(2))
    df.to_csv(name,index=False)    
    
    
    
sig_no_fault=meta_train[meta_train['target']==0].id_measurement.unique()    
    
for i in range(0,1):
    p1,p2,p3 = phase_indices(sig_no_fault[i])
    print(p1,p2,p3)
    df=train.iloc[:,[p1,p2,p3] ] 
    print(df.head())
    name='data_no_fault/'+str(sig_no_fault[i])+'.csv'
    print(name)
  #  print(df.tail(2))
    df.to_csv(name,index=False)     