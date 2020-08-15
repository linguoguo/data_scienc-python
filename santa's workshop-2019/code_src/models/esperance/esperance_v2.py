#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:38:30 2020

@author: lin
"""


proba_choice_1=npl_init(data,data['choice_1'].to_numpy())/total_people
esperance=(50/data['n_people'])*proba_choice_1[data['choice_1']]*data['n_people']
proba_choice_2=npl_init(data,data['choice_2'].to_numpy())/total_people
esperance+=(9+50/data['n_people'])*proba_choice_2[data['choice_2']]*data['n_people']
proba_choice_3=npl_init(data,data['choice_3'].to_numpy())/total_people
esperance+=(9+100/data['n_people'])*proba_choice_3[data['choice_3']]*data['n_people']
proba_choice_4=npl_init(data,data['choice_4'].to_numpy())/total_people
esperance+=(9+200/data['n_people'])*proba_choice_4[data['choice_4']]*data['n_people']
proba_choice_5=npl_init(data,data['choice_5'].to_numpy())/total_people
esperance+=(18+200/data['n_people'])*proba_choice_5[data['choice_5']]*data['n_people']
proba_choice_6=npl_init(data,data['choice_6'].to_numpy())/total_people
esperance+=(18+300/data['n_people'])*proba_choice_6[data['choice_6']]*data['n_people']
proba_choice_7=npl_init(data,data['choice_7'].to_numpy())/total_people
esperance+=(36+300/data['n_people'])*proba_choice_7[data['choice_7']]*data['n_people']
proba_choice_8=npl_init(data,data['choice_8'].to_numpy())/total_people
esperance+=(36+400/data['n_people'])*proba_choice_8[data['choice_8']]*data['n_people']
proba_choice_9=npl_init(data,data['choice_9'].to_numpy())/total_people
esperance+=(235+500/data['n_people'])*proba_choice_9[data['choice_9']]*data['n_people']



data_esp=data.copy()
data_esp['esperance']=esperance

data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'), index_col='family_id')
data_diff['esperance']=esperance

data_diff.sort_values(by='esperance',ascending=False)    

data_diff.to_csv(mug_data_path+'familys_difficult_esperance.csv') 
data_esp.to_csv(mug_data_path+'familys_esperance.csv') 
