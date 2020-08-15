#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:03:23 2020

@author: lin
"""
from classe.Cost_function import *
data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_esperance.csv'))
submission = pd.read_csv(os.path.join(raw_data_path,sample_submission_path), index_col='family_id')
wk_wd = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
wk_wd['esperance']=data_diff['esperance']
cost_function = build_cost_function(data_diff,N_DAYS=100)
best = submission['assigned_day'].values
print('initial score :', cost_function(best) )


data_diff_sort=data_diff.sort_values(by='esperance', ascending=False)
wk_wd_sort=wk_wd.sort_values(by='esperance', ascending=False)

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


r2=sort_by_esperance_k_top_wd(data_range = list(range(0,10,1)),k_top=6)

def sort_by_esperance_k_top_weekend(data_range,k_top):# weekend=1
    
    #print('-------------------------data:--------------------------------')
    #print(data.head(3),'\n')
    
    data_sort=data_diff_sort.iloc[data_range]
    wk_sort=wk_wd_sort.iloc[data_range]
    fam_id_sort=data_sort['family_id'].to_numpy()
    a=wk_sort[['family_id']+choices]
    a['family_id']=1
    b=data_sort[['family_id']+choices]
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

r3=sort_by_esperance_k_top_weekend(data_range = list(range(0,10,1)),k_top=6)


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

def loop_sorted(best,data_range,k_top,wd_iter,wk_iter):
    pred=best.copy()
    best_score=cost_function(pred)
    
    r_wd=sort_by_esperance_k_top_wd(data_range,k_top)
    r_wk=sort_by_esperance_k_top_weekend(data_range,k_top)
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


temp=loop_sorted(best,data_range = list(range(0,5000,1)),k_top=9,wd_iter=20,wk_iter=20)

for i in range(30):
    b = loop_sorted(temp[1],data_range = list(range(0,5000,1)),k_top=9,wd_iter=20,wk_iter=20)
    if temp[0]-b[0]>=10:
        temp = b
    else:
        print('final : ',int(b[0]))
        break

n1=npl_init(data_diff,b[1]) 
plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=n1[1:])    
    





#submission['assigned_day'] = b[1]


#submission.to_csv(mug_data_path+f'submission_{score}.csv')


"""
from random import randint

a=[randint(1,100) for i in range(5000)]
smooth = np.asarray(a)
temp=loop_sorted(smooth,data_range = list(range(0,5000,1)),k_top=9,wd_iter=20,wk_iter=20)

for i in range(300):
    b = loop_sorted(temp[1],data_range = list(range(0,5000,1)),k_top=9,wd_iter=20,wk_iter=20)
    if temp[0]-b[0]>=10:
        temp = b
    else:
        print('final : ',int(b[0]))
        break
    
 
    
    
    
    

a=[randint(1,100) for i in range(5000)]
smooth = np.asarray(a)
temp=loop_sorted(smooth,data_range = list(range(0,500,1)),k_top=9,wd_iter=20,wk_iter=20)       
l=[list(range(0,500,1)),list(range(0,1000,1)),list(range(0,1500,1)),list(range(0,2000,1)),list(range(0,2500,1)),list(range(0,3000,1)),list(range(0,3500,1)),list(range(0,4000,1)),list(range(0,4500,1)),list(range(0,5000,1))]

for k in range(4,10,1):
    print(k)
    for j in l:
        for i in range(30):
            b = loop_sorted(temp[1],data_range = j,k_top=k,wd_iter=20,wk_iter=20)
            if temp[0]-b[0]>=10:
                temp = b
            else:
                print('final : ',int(b[0]))
                break

"""

