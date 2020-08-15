#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:17:33 2020

@author: lin
"""

data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult_esperance.csv'))
data_diff_sort = data_diff.sort_values(by='esperance',ascending=False)    
submission = pd.read_csv(os.path.join(mug_data_path,'submission_77445.csv'), index_col='family_id')
best=submission['assigned_day'].values

data_diff['best']=best
data_diff['preference_cost']=0
data_diff['penality_cost']=0
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
        data_diff['penality_cost'].iloc[f]=n*df_days.iloc[d]['penality_mean']
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
preference_cost(best)

data_preference_cost_sort= data_diff.sort_values(by='preference_cost',ascending=False) 
data_penality_cost_sort = data_diff.sort_values(by='penality_cost',ascending=False) 


#beging with day 93 fam_id=4250
from classe.Cost_function import *
cost_function = build_cost_function(data_diff,N_DAYS=100)
print('initial score :', cost_function(best) )


def preference_cost_simple(prediction):
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

            
   

def loop(best,iteration):
    new = best.copy()

    for i in range(iteration):
        fam_to_replace = int( data_preference_cost_sort.iloc[i]['family_id'])
        n_ple =  data_preference_cost_sort.iloc[i]['n_people']    
        best_to_replace = data_preference_cost_sort.iloc[i]['best']
        l=[]
        for c in choices[:8]:
            cond1=data_diff[c]==best_to_replace
            cond2=data_diff['best']!=best_to_replace
            cond3=data_diff['n_people']>=n_ple
            l=np.append(l,data_diff[cond1 & cond2 & cond3]['family_id'].values)
        best_score=cost_function(new)
        for c in choices: 
            #print('',c)
            temp = new.copy()
            temp[fam_to_replace]=data_diff.iloc[fam_to_replace][c]
            for i in l:
                #print('i ',i)
                temp_de_temp=temp.copy()
                temp_de_temp[int(i)]=best_to_replace
                temp_score=cost_function(temp_de_temp)
                
                if temp_score < best_score:
                    new = temp_de_temp.copy()
                    best_score = temp_score
                    print(best_score)
                    
                      
    return new                
              

pred=best.copy()    
for j in [5000]:
    print(j)
    for i in range(10):
        print(i)
        data_diff['best']=pred
        preference_cost_simple(pred)
        data_preference_cost_sort = data_diff.sort_values(by='preference_cost',ascending=False) 
        pred=loop(pred,j)

    
best_score=int(best_score)

submission['assigned_day'] = pred


submission.to_csv(mug_data_path+f'submission_{best_score}.csv')
submission.to_csv(output_path+f'submission_{best_score}.csv')
