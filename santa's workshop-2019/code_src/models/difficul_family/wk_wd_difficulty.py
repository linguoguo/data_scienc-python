#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:41:11 2019

@author: lin
"""

path_figures=report_path+'/EDA/figures/'

from classe.Weekend_days import *

dif_list = [difficulty(data[choices].values[i][:].tolist()) for i in range(5000)]
dif_list_df= pd.DataFrame(dif_list,columns=choices)

dif_list_df['christ_eve']=(data['choice_0']==1)*1
dif_list_df['n_ppl']=data['n_people']
c=[0,1,2,3,4,5,6,7,8,9,4,1]
a=np.matmul(dif_list_df.to_numpy(),c)
dif_list_df['difficulty']=a
dif_list_df['family_id']=range(5000)
dif_list_df=dif_list_df.set_index('family_id')
dif_list_df.to_csv(mug_data_path+'difficult_wk_wd.csv') 
# weekday=1; weekend=0

familys_difficult=data.copy()
familys_difficult['difficulty']=a
familys_difficult['family_id']=range(5000)
familys_difficult=familys_difficult.set_index('family_id')
familys_difficult.to_csv(mug_data_path+'familys_difficult.csv') 

familys_difficult_sort=familys_difficult.sort_values(by='difficulty', ascending=False)
familys_difficult_sort.to_csv(mug_data_path+'familys_difficult_sort.csv')



plt.figure(figsize=(16,9))
sns.countplot(dif_list_df['difficulty'])
plt.title('difficult famillys')
plt.savefig(path_figures+'difficult_famillys.png')
    
    
    




