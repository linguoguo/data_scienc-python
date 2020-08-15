#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:14:22 2019

@author: lin
"""
from classe.Weekend_days import *
from classe.Cost_function import *
from classe.Np_days import *  
from classe.Sort_by_difficulty import *    




data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
wk_wd = pd.read_csv(os.path.join(mug_data_path,'difficult_wk_wd.csv'))
cost_function = build_cost_function(data_diff,N_DAYS=100)
top_four_choices=choices[0:4]

def first_4():  
    daily_occupancy = np.zeros((101,), dtype=int)
    pred = np.zeros((5000,), dtype=int)
    f_n=0
    for i in range(56,5,-1):
        print(i)
        l_fam_id = data_diff[data_diff['difficulty']==i]['family_id']
        for f in l_fam_id:
            #print('fam ',f)
            print(data_diff.iloc[f][choices])
            ocp = data_diff.iloc[f]['n_people']
            for d in data_diff.iloc[f][top_four_choices]:
                                
                if daily_occupancy[d] <= 280-ocp:
                    print('fam :',f,'day ',d)
                    daily_occupancy[d] += ocp
                    print(daily_occupancy)
                    pred[f]=d
                    f_n+=1
                    print('familly traitée ',f_n)
                    break
    return daily_occupancy,pred
    
    
k=first_4()    


plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=k[0][1:])

n1=npl_init(data_diff,k[1]) 
plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=n1)




def bot_4(d):
    daily_occupancy=d[0].copy()
    pred=d[1].copy()
    f_n=0
    for n in range(2,5,1):
        print(n)
        cond1=data_diff['n_people'] == n
        for i in range(56,35,-1):
            cond2=data_diff['difficulty'] == i
            l_fam_id=data_diff[cond1 & cond2]['family_id']
            for f in l_fam_id:
                print('fam ',f)
                m=np.argmin(daily_occupancy[1:])+1
                print(daily_occupancy[m])
                if daily_occupancy[m]<=240-n:
                    daily_occupancy[pred[f]]-=n
                    daily_occupancy[m]+=n
                    pred[f]=m
                    print(daily_occupancy[m])
                    f_n+=1
                    print('familly traitée ',f_n)                    
                else:
                    break
            
    return daily_occupancy, pred       
    
    
d1=bot_4(k)  
plt.figure(figsize=(25,10))
sns.barplot(x=days_inverse,y=d1[0][1:])  



new_df=data_diff.copy()
new_df['pred']=d1[1]
ll=d1[1].copy()
p=0
for i in range(5000):
    if new_df.iloc[i]['pred']==0:
        for c in new_df.iloc[i][choices]:
            if c not in liste_weekend[0]+1 :
                print(i,c)
                ll[i]=c
                break


n2=npl_init(data_diff,ll) 
plt.figure(figsize=(25,10))
sns.barplot(x=days,y=n2)


#submission = pd.read_csv(os.path.join(raw_data_path,sample_submission_path), index_col='family_id')
#submission['assigned_day'] = ll   
#submission.to_csv(  mug_data_path+'smooth_v3.csv') 
'''
def penality(nd,nd_1):
    return (nd-125)*nd**(0.5+abs(nd-nd_1)/50)


f = open("penality.txt", "a")
f.write("for nd+1=250!")
f.write('\n')
f.close()

f = open("penality.txt", "a")
for i in range(240,300,1):
    x=penality(i,250)
    print(i,x,'~ 8p others:',x//3972,'~ 4p others:',x//2236)
    t=i,x,'~ 8p others:',x//3972,'~ 4p others:',x//2236
    f.write(str(t))
    f.write('\n')
    
f.close()
'''   