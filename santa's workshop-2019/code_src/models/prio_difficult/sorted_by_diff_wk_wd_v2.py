


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:04:36 2019

@author: lin
"""


from classe.Weekend_days import *
from classe.Cost_function import *







data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
submission = pd.read_csv(os.path.join(mug_data_path,'smooth_v2.csv'), index_col='family_id')
wk_wd = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
cost_function = build_cost_function(data_diff,N_DAYS=100)
best=submission['assigned_day'].values
print('initial score :', cost_function(best) )




    





def sort_by_difficulty_k_top(data,wk,diff_nv,k_top,n_expl):# version weekend=0
    print('-------------------------data:--------------------------------')
    print(data.head(3),'\n')
    
    data_sort=data.sort_values(by='difficulty', ascending=False)
    d_n=data_sort[data_sort['difficulty']>=diff_nv].shape[0]
    n_expl=min(n_expl,d_n)
    
    fam_id_sort=data_sort['family_id'].to_numpy()
    a=abs(wk_wd[['family_id']+choices].iloc[fam_id_sort[:n_expl]]-np.full((n_expl,11),1))
    a['family_id']=1
    b=data[['family_id']+choices].iloc[fam_id_sort[:n_expl]]
    data_wk_wd=b*a
    print('--------------data sorted by difficulty; 0: weekend--------------\n')
    print(data_wk_wd.head(3),'\n')

    ll=data_wk_wd.to_numpy()
    liste_nonzero=data_wk_wd[choices].to_numpy().nonzero()
    result=[[],[]]
    for i in range(len(liste_nonzero[0])):
        fam_index=ll[liste_nonzero[0][i]][0]
        choice=ll[liste_nonzero[0][i]][liste_nonzero[1][i]+1]
        if liste_nonzero[1][i]<=k_top:
            result[0].append(fam_index)
            result[1].append(choice)
    print(result)
    return result




sort_by_difficulty_k_top(data_diff,wk=wk_wd,diff_nv=50,k_top=6,n_expl=20)



def sort_by_difficulty_k_top_v2(data,wk,diff_nv,k_top,n_expl): # version weekend=0
    #print('-------------------------data:--------------------------------')
    #print(data.head(3),'\n')
    
    data_sort=data.sort_values(by='difficulty', ascending=False)
    wk_wd_sort=wk.sort_values(by='difficulty', ascending=False)
    d_n=data_sort[data_sort['difficulty']>=diff_nv].shape[0]
    n_expl=min(n_expl,d_n)
    
    fam_id_sort=data_sort['family_id'].to_numpy()
    a=abs(wk_wd_sort[['family_id']+choices].iloc[:n_expl]-np.full((n_expl,11),1))
    a['family_id']=1
    b=data_sort[['family_id']+choices].iloc[:n_expl]
    data_wk_wd=b*a
    #print('--------------data sorted by difficulty; 0: weekend--------------\n')
    #print(data_wk_wd.head(3),'\n')

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


r2=sort_by_difficulty_k_top_v2(data_diff.iloc[-30:-1],wk=wk_wd.iloc[-30:-1],diff_nv=50,k_top=6,n_expl=10)
r3=sort_by_difficulty_k_top_v2(data_diff,wk=wk_wd,diff_nv=50,k_top=6,n_expl=20)


def loop_choices(data,list_choice,ass):
    n = len(list_choice[0])
    new = ass.copy()
    best_score=cost_function(new)
    for i in range(n):
        temp = new.copy()
        temp[list_choice[0][i]]=list_choice[1][i]
        #print('fam_id : ',list_choice[0][i])
        temp_score=cost_function(temp)
        if temp_score < best_score:
            
            #print(i,temp_score)
            new = temp.copy()
            best_score = temp_score
        pred=new    
        
    return best_score, pred


r4=sort_by_difficulty_k_top_v2(data_diff,wk=wk_wd,diff_nv=20,k_top=9,n_expl=50)
loop_choices(data_diff,list_choice=r4,ass=best)


def sort_by_difficulty_k_top_v_weekend(data,wk,diff_nv,k_top,n_expl):# version weekend=1
    
    #print('-------------------------data:--------------------------------')
    #print(data.head(3),'\n')
    
    data_sort=data.sort_values(by='difficulty', ascending=False)
    wk_wd_sort=wk.sort_values(by='difficulty', ascending=False)
    d_n=data_sort[data_sort['difficulty']>=diff_nv].shape[0]
    n_expl=min(n_expl,d_n)
    
    fam_id_sort=data_sort['family_id'].to_numpy()
    a=wk_wd_sort[['family_id']+choices].iloc[:n_expl]
    a['family_id']=1
    b=data_sort[['family_id']+choices].iloc[:n_expl]
    data_wk_wd=b*a
    #print('--------------data sorted by difficulty; 1: weekend--------------\n')
    #print(data_wk_wd.head(3),'\n')

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


r5=sort_by_difficulty_k_top_v_weekend(data_diff,wk=wk_wd,diff_nv=20,k_top=9,n_expl=50)

loop_1=loop_choices(data_diff,list_choice=r5,ass=best)


def loop_sorted(best,diff_nv,k_top,n_expl,wd_iter,wk_iter):
    pred=best.copy()
    best_score=cost_function(pred)
    r_wd=sort_by_difficulty_k_top_v2(data_diff,wk_wd,diff_nv,k_top,n_expl=5000)
    r_wk=sort_by_difficulty_k_top_v_weekend(data_diff,wk_wd,diff_nv,k_top,n_expl=5000)
    for i in range(wd_iter):
        lp=loop_choices(data_diff,list_choice=r_wd,ass=pred)
        if lp[0]==best_score:
            break
        if lp[0]<best_score:
            best_score=lp[0]
            pred=lp[1]
            print  (i,best_score)

    for j in range(wk_iter):
        lp=loop_choices(data_diff,list_choice=r_wk,ass=pred)
        if lp[0]==best_score:
            break
        if lp[0]<best_score:
            best_score=lp[0]
            pred=lp[1]
            print  (j,best_score)           

    
    return best_score,pred


p=loop_sorted(best,diff_nv=50,k_top=9,n_expl=5000,wd_iter=20,wk_iter=20)


best_score_v1=cost_function(best)
best_v1=best.copy()
for i in range(55,0,-1):    
    print(i)
    p=loop_sorted(best_v1,diff_nv=i,k_top=8,n_expl=5000,wd_iter=20,wk_iter=20)
    if p[0]<best_score_v1:
        best_score_v1=p[0]
        best_v1=p[1]


n1=npl_init(data_diff,best_v1) 
plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=n1)
#submission['assigned_day'] = best_v1    
#submission.to_csv(  mug_data_path+'sorted_by_wk_wd_v1.csv')  




