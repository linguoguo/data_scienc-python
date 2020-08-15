


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:04:36 2019

@author: lin
"""


from classe.Weekend_days import *
from classe.Cost_function import *
from classe.Np_days import *  
from classe.Sort_by_difficulty import *         






data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
submission = pd.read_csv(os.path.join(mug_data_path,'smooth_v3.csv'), index_col='family_id')
wk_wd = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
cost_function = build_cost_function(data_diff,N_DAYS=100)
best=submission['assigned_day'].values
print('initial score :', cost_function(best) )




    
r1=sort_by_difficulty_k_top(data_diff,wk=wk_wd,diff_nv=50,k_top=6,n_expl=20) # version weekend=0

r2=sort_by_difficulty_k_top_v2(data_diff.iloc[-30:-1],wk=wk_wd.iloc[-30:-1],diff_nv=50,k_top=6,n_expl=10)# version weekend=0
r3=sort_by_difficulty_k_top_v2(data_diff,wk=wk_wd,diff_nv=50,k_top=6,n_expl=20)# version weekend=0


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
test_1=loop_choices(data_diff,list_choice=r4,ass=best)

r5=sort_by_difficulty_k_top_v_weekend(data_diff,wk=wk_wd,diff_nv=20,k_top=9,n_expl=50) # version weekend=1

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


best_score_v1=cost_function(best_vt)
best_v1=best_vt.copy()
for i in range(55,54,-1):    
    print(i)
    p=loop_sorted(best_v1,diff_nv=i,k_top=10,n_expl=5000,wd_iter=20,wk_iter=20)
    if p[0]<best_score_v1:
        best_score_v1=p[0]
        best_v1=p[1]

submission['assigned_day'] = best_v1    
#submission.to_csv(  mug_data_path+'sorted_by_wk_wd_v1.csv')  




best_v1 = pd.read_csv(mug_data_path+'sorted_by_wk_wd_v1.csv', index_col='family_id')['assigned_day'].to_numpy()
best_others = pd.read_csv("/home/lin/kaggle/santa's workshop/reports/best_others.csv", index_col='family_id')['assigned_day'].to_numpy()
n1=npl_init(data_diff,best_v1) 
n2=npl_init(data_diff,best_others)   

 


plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=n2)

n3=npl_permute(data_diff,nInit=n2,ass=submission['assigned_day'].to_numpy(),f_id=0,d1=2,d2=1) 
plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=n3)


for c in choices:
    c9=data_diff[c].to_numpy()
    print(c,' : ',((c9==best_v1)*1).sum())
    np=((c9==best_v1)*1).nonzero()
    print(data_diff.iloc[np]['n_people'])
    
    



liste_weekend=np.array(difficulty(days)).nonzero()

def loop_others(best,fam_id):
    list_c=data_diff.iloc[fam_id][choices].to_numpy()
    list_c=np.concatenate((liste_weekend, list_c), axis=None)
    #print('list_c', list_c,len(list_c))
    np_days=npl_init(data_diff,best)
    best_score=cost_function(best)
    pred=best.copy()
    for i in days_inverse:        
        temp=pred.copy()
        if i not in list_c :
            #print(i)
            temp[fam_id]=i
            new_score=cost_function(temp)
            if new_score<best_score:                    
                pred=temp
                best_score=new_score
                print(new_score)
    return pred

p1=loop_others(submission['assigned_day'].to_numpy(),0)  



def loop_sorted_others(best,diff_bot,k_bot,n_pl):
    pred=best.copy()
    best_score=cost_function(pred)
    cond1=data_diff['n_people']==n_pl
    cond2=data_diff['difficulty']>=diff_bot
    resu=data_diff[cond1 & cond2]['family_id']
    for fam in resu:
        print(fam)
        temp=pred.copy()
        pred=loop_others(temp,fam)
    return best_score,pred   

p=loop_sorted_others(best_v1,diff_bot=49,k_bot=8,n_pl=2)


l_mafan_2=[]
for i in range(1):
    c=choices[i]
    cond1=data_diff['n_people']==2
    cond2=data_diff['difficulty']>=47
    resu=data_diff[cond1 & cond2]['family_id']
    l_mafan_2=np.concatenate((l_mafan_2, resu), axis=None)



data_diff[data_diff['n_people']==7]['family_id']










best_score_v2=cost_function(best)
best_v2=best.copy()
for i in range(2,25,1):    
    print(i)
    p=loop_sorted(best_v2,diff_nv=i,k_bot=8,n_expl=5000)
    if p[0]<best_score_v2:
        best_score_v1=p[0]
        best_v1=p[1]



best_score_v1=cost_function(best)
best_v1=best.copy()
for i in range(55,0,-1):    
    print(i)
    p=loop_sorted(best_v1,diff_nv=i,k_top=10,n_expl=5000,wd_iter=20,wk_iter=20)
    if p[0]<best_score_v1:
        best_score_v1=p[0]
        best_v1=p[1]


n2=npl_init(data_diff,best_v1) 
plt.figure(figsize=(25,10))
sns.barplot(x=days,y=n2)




'''

l1=data_diff.iloc[0][choices].to_numpy()
if 52 in l1:
    print('ok')
    
        

def loop_others(best,fam_id):
    np=data_diff.iloc[fam_id]['n_people']
    list_c=data_diff.iloc[fam_id][choices].to_numpy()
    np_days=npl_init(data_diff,best)
    best_score=cost_function(best)
    pred=best.copy()
    for i in days_inverse:
        #print(i)
        temp=pred.copy()
        if i in list_c or np_days[i-1]>=300-np :
            i+=1
            #print('nop')
        temp[fam_id]=i
        new_score=cost_function(temp)
        
        if new_score<best_score:
            pred=temp
            best_score=new_score
            print(new_score)
    return pred

p1=loop_others(submission['assigned_day'].to_numpy(),62)        
      

data_diff_sort = pd.read_csv(os.path.join(mug_data_path,'familys_difficult_sort.csv'))
pred=best_v1.copy()
for fam in data_diff_sort['family_id']:
    print (fam)
    temp=loop_others(pred,fam)
    pred=temp.copy()






data_diff_cr=data_diff.sort_values(by='difficulty', ascending=True)
best_v2=best_v1.copy()
for i in range(5000):
    print(i)
    temp=loop_others(best_v2,data_diff_cr.iloc[i]['family_id'])
    best_v2=temp.copy()    











def loop_sorted(best,diff_nv,k_top,n_expl,wd_iter,wk_iter):
    pred=best.copy()
    best_score=cost_function(pred)
    r_wd=sort_by_difficulty_k_top_v2(data_diff,wk_wd,diff_nv,k_top,n_expl=5000)
    print(r_wd)
    r_wk=sort_by_difficulty_k_top_v_weekend(data_diff,wk_wd,diff_nv,k_top,n_expl=5000)
    print(r_wk)
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
        if lp[0]==best_score:count of each unique values in column
            break
        if lp[0]<best_score:
            best_score=lp[0]
            pred=lp[1]
            print  (j,best_score)           
    
    for fam in list(set(r_wd[0])):
        #print(fam)
        loop_others(pred,fam)
    print(best_score)    
    return best_score,pred


p=loop_sorted(best,diff_nv=55,k_top=9,n_expl=5000,wd_iter=20,wk_iter=20)


best_score_v3=cost_function(best)
best_v3=best.copy()
for i in range(55,0,-1):    
    print(i)
    p=loop_sorted(best_v3,diff_nv=i,k_top=10,n_expl=5000,wd_iter=20,wk_iter=20)
    if p[0]<best_score_v3:
        best_score_v3=p[0]
        best_v3=p[1]


'''    
