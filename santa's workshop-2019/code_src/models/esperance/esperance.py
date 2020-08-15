#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:38:30 2020

@author: lin
"""
top_7_choices=[f'choice_{i}' for i in range(8)]
bot_3_choices=['choice_8','choice_9']
top_7_occupancy=npl_init(data,data['choice_0'].to_numpy())

for c in top_7_choices[1:]:
    top_7_occupancy+=npl_init(data,data[c].to_numpy())
    
bot_3_occupancy=npl_init(data,data['choice_8'].to_numpy())+npl_init(data,data['choice_9'].to_numpy())

proba_top_7=top_7_occupancy/top_7_occupancy.sum()
proba_bot_3=bot_3_occupancy/bot_3_occupancy.sum()



plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=proba_top_7[1:])


plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=proba_bot_3[1:])

esperance=(50/data['n_people'])*proba_top_7[data['choice_1']]*data['n_people']
esperance+=(9+50/data['n_people'])*proba_top_7[data['choice_2']]*data['n_people']
esperance+=(9+100/data['n_people'])*proba_top_7[data['choice_3']]*data['n_people']
esperance+=(9+200/data['n_people'])*proba_top_7[data['choice_4']]*data['n_people']
esperance+=(18+200/data['n_people'])*proba_top_7[data['choice_5']]*data['n_people']
esperance+=(18+300/data['n_people'])*proba_top_7[data['choice_6']]*data['n_people']
esperance+=(36+300/data['n_people'])*proba_top_7[data['choice_7']]*data['n_people']
esperance+=(36+400/data['n_people'])*proba_bot_3[data['choice_8']]*data['n_people']
esperance+=(235+500/data['n_people'])*proba_bot_3[data['choice_9']]*data['n_people']

data_esp=data.copy()
data_esp['esperance']=esperance

data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
data_diff['esperance']=esperance

data_diff.sort_values(by='esperance',ascending=False)    

data_diff.to_csv(mug_data_path+'familys_difficult_esperance.csv') 
data_esp.to_csv(mug_data_path+'familys_esperance.csv') 
