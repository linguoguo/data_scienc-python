

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:04:36 2019

@author: lin
"""


from classe.Weekend_days import *
from classe.Cost_function import *


data_reduc = pd.read_csv(os.path.join(mug_data_path,'sample_reduc_2_d.csv'))
#data_sort=data_reduc.sort_values(by=['difficulty'], ascending=False)


cost_function = build_cost_function(data,N_DAYS=100)
#cost_function = build_cost_function(data)
#original = submission['assigned_day'].values
#best=original = submission['assigned_day'].values
#original_score = cost_function(best)

"""
l = [random.randrange(1,30) for i in range(2500)]
best=np.asarray(l)
best_score = cost_function(best)
choice_matrix = data_reduc.loc[:, 'choice_0': 'choice_9'].values

"""

best=submission['assigned_day'].values

print('initial score :', cost_function(best) )

new = best.copy()

score = cost_function(new)

print(f'Score: {score}')


data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))


data_diff_sort=data_diff.sort_values(by='difficulty', ascending=False)
fam_id=data_diff_sort['family_id'].to_numpy()
best_sort=data['choice_0'].to_numpy()

best_score=cost_function(best)
new = best.copy()
print('init : ',cost_function(new))
for i in fam_id[:50]:
    temp = new.copy()
    temp[i]=best_sort[i]
    temp_score=cost_function(temp)
    
    if temp_score < best_score:
        print(i,temp_score)
        new = temp.copy()
        best_score = temp_score
    


wk_wd = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
n_expl=10
a=abs(wk_wd[['family_id']+choices].iloc[fam_id[:n_expl]]-np.full((n_expl,11),1))
a['family_id']=1
b=data_diff[['family_id']+choices].iloc[fam_id[:n_expl]]
data_wk_wd=b*a


best_score=cost_function(best)
new = best.copy()
print('init : ',cost_function(new))


data_wd=data_wk_wd.tail(5).to_numpy()
liste_nonzero=data_wd.nonzero()
liste_fam=data_wd[sorted(set(liste_nonzero[0])),0]
#index_wd=np.nonzero(liste_nonzero[1])

n_wd=0
for i in range(len(liste_fam)):
    
    liste_wd=np.nonzero(liste_nonzero[0]==i)
    print (i,liste_fam[i],liste_wd)
    for c in liste_wd[0]:
        new_day=data_wd[i,liste_nonzero[1][c]]
        print('c ',new_day)

        
    n_wd+=1
    
    for j in liste_wd[0][i]
    
    for fam_id[i]
    
    temp = new.copy()
    temp[i]=best_sort[i]
    temp_score=cost_function(temp)
    
    if temp_score < best_score:
        print(i,temp_score)
        new = temp.copy()
        best_score = temp_score
    

