#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:38:30 2020

@author: lin
"""




data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult_esperance.csv'), index_col='family_id')

data_diff_sort = data_diff.sort_values(by='esperance',ascending=False)    


submission = pd.read_csv(os.path.join(mug_data_path,'smooth_v3.csv'), index_col='family_id')
best=submission['assigned_day'].values

data_diff['best']=best
data_diff['preference_cost']=0

def preference_cost(prediction):
    for f, d in enumerate(prediction):
        #print(f,d)
        choice_0 = data_diff['choice_0'].iloc[f]
        #print(choice_0)
        choice_1 = data_diff['choice_1'].iloc[f]
        choice_2 = data_diff['choice_2'].iloc[f]
        #print(choice_2)
        choice_3 = data_diff['choice_3'].iloc[f]
        choice_4 = data_diff['choice_4'].iloc[f]
        choice_5 = data_diff['choice_5'].iloc[f]
        choice_6 = data_diff['choice_6'].iloc[f]
        choice_7 = data_diff['choice_7'].iloc[f]
        choice_8 = data_diff['choice_8'].iloc[f]
        choice_9 = data_diff['choice_9'].iloc[f]
        n=data_diff.iloc[f]['n_people']
        if d == choice_0:
            penalty=0
        elif d == choice_1:
            penalty = 50
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_2:            
            penalty = 50 + 9 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_3:
            penalty = 100 + 9 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_4:
            penalty = 200 + 9 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_5:
            penalty = 200 + 18 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_6:
            penalty = 300 + 18 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_7:
            penalty = 300 + 36 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_8:
            penalty = 400 + 36 * n
            data_diff['preference_cost'].iloc[f]=penalty
        elif d == choice_9:
            penalty = 500 + 36 * n + 199 * n
            data_diff['preference_cost'].iloc[f]=penalty
        else:
            penalty = 500 + 36 * n + 398 * n
            data_diff['preference_cost'].iloc[f]=penalty

preference_cost(best)

daily_occupancy=npl_init(data_diff,best)
df_days=pd.DataFrame([0]+days_inverse,columns=['days'])
df_days['occupancy']=daily_occupancy

df_days['penality']=0
df_days['penality'].iloc[days[0]] = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
yesterday_count = daily_occupancy[days[0]]
for day in days[1:]:
    today_count = daily_occupancy[day]
    diff = abs(today_count - yesterday_count)
    accounting_cost = max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
    print('day: ',day,'accounting_cost : ',accounting_cost)
    yesterday_count = today_count
    df_days['penality'].iloc[day] = accounting_cost
   

df_days['penality_mean']=df_days['penality']/df_days['occupancy']

df_days['familys']=[[]]*101
for day in days:
    print(day)
    list_family = data_diff[data_diff['best']==day].index.to_numpy()
    df_days['familys'].iloc[day] = list_family

df_days['preference_cost']=0

for day in days:
    list_family=df_days.iloc[day]['familys']
    df_days['preference_cost'].iloc[day] = data_diff.iloc[list_family]['preference_cost'].sum()

df_days['total_cost']=df_days['penality']+df_days['preference_cost']

df_days['average_cost']=df_days['total_cost']/df_days['occupancy']




df_days['average_cost'].max()

"""
for c in choices:
    df_days[c]=[[]]*101
df_days['others']=[[]]*101

for day in days[1:]:
    for c in choices:
        print('day',day,'choice',c)
        cond1=data_diff[c]==data_diff['best']
        cond2=data_diff[c]==day
        a=data_diff[cond1 & cond2].index.to_numpy()
        print(a)
        df_days[c].iloc[day]=a


for day in days[1:]:
    print( df_days.iloc[day]['choice_6'])


b=data_diff[data_diff['choice_8']==data_diff['best']]['choice_8']


d=df_days.copy(deep=False)
for i in range(len(b)):
    fam_id=b.index[i]
    d=b.iloc[i]
    actu=df.iloc[d]['choice_8']
    print('f',fam_id,'d',d)
    print(actu)
    actu.append(fam_id)
    df.iloc[d]['choice_8']=actu
    
"""