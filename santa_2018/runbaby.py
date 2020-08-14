#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 06:50:45 2019

@author: lin
"""

import os
cluster1=[4,10,20,30]
cluster2=[6,8,12,14,16,18]
cluster3=[22,24,26,28,32]
for i in cluster1:
    print(i)
    n_cluster=i
    exec(open("datas_densy.py").read())

for i in cluster2:
    print(i)
    n_cluster=i
    exec(open("datas_densy.py").read()) 
    
for i in cluster3:
    print(i)
    n_cluster=i
    exec(open("datas_densy.py").read())