#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:17:33 2020

@author: lin
"""

data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult_esperance.csv'))
data_diff_sort = data_diff.sort_values(by='esperance',ascending=False)    
submission = pd.read_csv(os.path.join(mug_data_path,'submission_88984.csv'), index_col='family_id')
best=submission['assigned_day'].values
data_diff['best']=best

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
    
df_days_sort=df_days.sort_values(by='penality',ascending=False)    


plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=df_days['penality'].iloc[1:])


def sort_by_esperance_k_top_wd(data_range,k_top): #  weekend=0
    data_sort=data_diff_sort.iloc[data_range]#esperance decroissante
    #print(data_sort)
    wd_sort=wk_wd_sort.iloc[data_range]
    #print(wd_sort)
    fam_id_sort=data_sort['family_id'].to_numpy()
    n_expl=len(data_sort)
    a=abs(wd_sort[['family_id']+choices]-np.full((n_expl,11),1))
    a['family_id']=1
    b=data_sort[['family_id']+choices]
    data_wk_wd=b*a
    #print('--------------data sorted by difficulty; 0: weekend--------------\n')
    #print(data_wk_wd.head,'\n')

    ll=data_wk_wd.to_numpy()
    liste_nonzero=data_wk_wd[choices].to_numpy().nonzero()
    result=[[],[]]
    for i in range(len(liste_nonzero[0])):
        fam_index=ll[liste_nonzero[0][i]][0]
        choice=ll[liste_nonzero[0][i]][liste_nonzero[1][i]+1]
        if liste_nonzero[1][i]<=k_top:
            result[0].append(fam_index)
            result[1].append(choice)
    #print(result[0])
    #print(result[1])
    return result




liste_famille = df_days_sort.iloc[0]['familys']
wk_wd_sort = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
data_diff_sort = data_diff.copy()
r2 = sort_by_esperance_k_top_wd(data_range = liste_famille,k_top=8)







def penalty(ass):
    daily_occupancy=npl_init(data_diff,ass)
    penalty=0
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost = max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        #print('day: ',day,'accounting_cost : ',accounting_cost)
        yesterday_count = today_count
        penalty += accounting_cost
    return penalty
    
pen=penalty(best)




def loop_choices_penalty(data,list_choice,ass):
    n = len(list_choice[0])
    new = ass.copy()
    best_score=penalty(new)
    for i in range(n):
        temp = new.copy()
        temp[list_choice[0][i]]=list_choice[1][i]
        #print('fam_id : ',list_choice[0][i])
        temp_score=penalty(temp)
        if temp_score < best_score:
            
            print(i,temp_score)
            new = temp.copy()
            best_score = temp_score
        pred=new    
        
    return best_score, pred



p=loop_choices_penalty(data_diff,list_choice=r2,ass=best)

pre=best.copy()

for i in range(5):
    liste_famille = df_days_sort.iloc[i]['familys']
    r2 = sort_by_esperance_k_top_wd(data_range = liste_famille,k_top=7)
    p=loop_choices_penalty(data_diff,list_choice=r2,ass=pre)
    pre=p[1]
    




"""

#beging with day 93 fam_id=4250
from classe.Cost_function import *
cost_function = build_cost_function(data_diff,N_DAYS=100)
print('initial score :', cost_function(best) )


l=[]
for c in choices[:8]:
    cond1=data_diff[c]==79
    cond2=data_diff['best']!=79
    cond3=data_diff['n_people']>=4
    l=np.append(l,data_diff[cond1 & cond2 & cond3]['family_id'].values)



new = pred.copy()
best_score=cost_function(new)
for c in choices:    
    temp = new.copy()
    temp[891]=data_diff.iloc[891][c]
    for i in l:
        temp_de_temp=temp.copy()
        temp_de_temp[int(i)]=79
        temp_score=cost_function(temp_de_temp)
        
        if temp_score < best_score:
            new = temp_de_temp.copy()
            best_score = temp_score
            print(best_score)
        pred=new    
            



pred=best.copy()
for i in range(5000):
    fam_to_replace = int( data_preference_cost_sort.iloc[i]['family_id'])
    print(fam_to_replace)
    n_ple =  data_preference_cost_sort.iloc[i]['n_people']  
    print(n_ple)     
    best_to_replace = data_preference_cost_sort.iloc[i]['best']
    print(best_to_replace)
    
    l=[]
    for c in choices[:8]:
        cond1=data_diff[c]==best_to_replace
        cond2=data_diff['best']!=best_to_replace
        cond3=data_diff['n_people']>=n_ple
        l=np.append(l,data_diff[cond1 & cond2 & cond3]['family_id'].values)
    new = pred.copy()
    best_score=cost_function(new)
    for c in choices:    
        temp = new.copy()
        temp[fam_to_replace]=data_diff.iloc[fam_to_replace][c]
        for i in l:
            temp_de_temp=temp.copy()
            temp_de_temp[int(i)]=best_to_replace
            temp_score=cost_function(temp_de_temp)
            
            if temp_score < best_score:
                new = temp_de_temp.copy()
                best_score = temp_score
                print(best_score)
            pred=new         
  
best_score=int(best_score)

submission['assigned_day'] = pred


submission.to_csv(mug_data_path+f'submission_{best_score}.csv')
submission.to_csv(output_path+f'submission_{best_score}.csv')
"""