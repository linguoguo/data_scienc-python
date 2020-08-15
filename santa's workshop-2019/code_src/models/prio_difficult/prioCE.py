#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:27:53 2019

@author: lin
"""
def npl_init(data,ass):
    npl_days=np.zeros((100,), dtype=int)
    for i in range(5000):
        j= data.iloc[i]['n_people'] 
        npl_days[ass[i]-1]+=j
    return npl_days



def npl_permute(data,nInit,ass,f_id,d1,d2):
    npl=data.iloc[f_id]['n_people']
    nInit[d1-1]-=npl
    nInit[d2-1]+=npl
    return nInit
        

        
        
npl_init(data,submission['assigned_day'].to_numpy())      
npl_permute(data,submission['assigned_day'].to_numpy(),0,2,1)  
