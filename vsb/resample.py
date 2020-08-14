#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:54:21 2019

@author: lin
"""

# -*- coding: utf-8 -*-
"""
Lin GUO

"""

import pandas as pd
import pyarrow.parquet as pq
import os
import csv
import numpy as np
import pygame
from pygame.locals import *
train = pq.read_pandas('train.parquet').to_pandas()
train.info()





#    get the power lines fault
meta_train = pd.read_csv('metadata_train.csv')
sig_fault=meta_train[meta_train['target']==1].id_measurement.unique()

def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

def chunks(df,n):
    c=np.split(df,n) 
    a=[]
    for i in c:
        a.append(i.mean())
    return pd.DataFrame(a) 
    
for i in range(0,0):
    p1,p2,p3 = phase_indices(sig_fault[i])
    print(p1,p2,p3)
    df=train.iloc[:,[p1,p2,p3] ] 
    ddf=chunks(df,80000)
    print(ddf.head())
    name='resamples/data_fault/'+str(sig_fault[i])+'.csv'
    print(name)
  #  print(df.tail(2))
    ddf.to_csv(name,index=False)    
pygame.init()
#fenetre = pygame.display.set_mode((300,300))
son = pygame.mixer.Sound("son.WAV")
son.play()   
    
    
sig_no_fault=meta_train[meta_train['target']==0].id_measurement.unique()    
  
for i in range(1,2):
    p1,p2,p3 = phase_indices(sig_no_fault[i])
    print(p1,p2,p3)
    df=train.iloc[:,[p1,p2,p3] ] 
    ddf=chunks(df,160000)
    print(ddf.head())
    name='resamples/data_no_fault/'+str(sig_no_fault[i])+'.csv'
    print(name)
  #  print(df.tail(2))
    ddf.to_csv(name,index=False)  
    
pygame.init()
#fenetre = pygame.display.set_mode((300,300))
son = pygame.mixer.Sound("son.WAV")
son.play()     
