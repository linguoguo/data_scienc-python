


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:04:36 2019

@author: lin
"""


from classe.Weekend_days import *
from classe.Cost_function import *







data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
submission = pd.read_csv(os.path.join(raw_data_path,sample_submission_path), index_col='family_id')
wk_wd = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
cost_function = build_cost_function(data_diff,N_DAYS=100)
best=submission['assigned_day'].values
print('initial score :', cost_function(best) )




    





def sort_by_difficulty_k_top(data,wk,diff_nv,k_top,n_expl):
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



def sort_by_difficulty_k_top_v2(data,wk,diff_nv,k_top,n_expl):
    print('-------------------------data:--------------------------------')
    print(data.head(3),'\n')
    
    data_sort=data.sort_values(by='difficulty', ascending=False)
    wk_wd_sort=wk.sort_values(by='difficulty', ascending=False)
    d_n=data_sort[data_sort['difficulty']>=diff_nv].shape[0]
    n_expl=min(n_expl,d_n)
    
    fam_id_sort=data_sort['family_id'].to_numpy()
    a=abs(wk_wd_sort[['family_id']+choices].iloc[:n_expl]-np.full((n_expl,11),1))
    a['family_id']=1
    b=data_sort[['family_id']+choices].iloc[:n_expl]
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
    print(result[0])
    print(result[1])
    return result


r2=sort_by_difficulty_k_top_v2(data_diff.iloc[-30:-1],wk=wk_wd.iloc[-30:-1],diff_nv=50,k_top=6,n_expl=20)

r3=sort_by_difficulty_k_top_v2(data_diff,wk=wk_wd,diff_nv=50,k_top=6,n_expl=20)




