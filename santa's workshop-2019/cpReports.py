# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import os
import time
import datetime
import pandas as pd


dir_path=os.getcwd()
file_list=[]
file_path_list=[]
file_lm_time=[]
now = datetime.datetime.now()


            
            
            
file_list_unique=list(set(file_list))            


for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".odt"):
            file_list.append(file)
            file_path_list.append(os.path.join(root, file))
            file_lm_time.append(os.path.getmtime(os.path.join(root, file)))
            print(os.path.join(root, file))
            t=os.path.getmtime(os.path.join(root, file))
            
            print("last modified:" , datetime.datetime.fromtimestamp(t)>now)






data={'file':file_list,'path':file_path_list,'lm_time':file_lm_time}      
df=pd.DataFrame(data)       




'''
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".odt"):
            file_list.append(file)
            file_path_list.append(os.path.join(root, file))
            file_lm_time.append(time.ctime(os.path.getmtime(os.path.join(root, file))))
            print(os.path.join(root, file))
            t=os.path.getmtime(os.path.join(root, file))
            
            print("last modified: %s" % datetime.datetime.fromtimestamp(t))
            
'''            