#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 23:45:47 2019

@author: lin
"""

def cost_function_starter(prediction):

    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}
    
    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):

        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty


family_size_dict = data[['n_people']].to_dict()['n_people']
cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].to_dict()

best = submission['assigned_day'].tolist()
start_score = cost_function_starter(best)

new = best.copy()
'''
# loop over each family

f = open("stater_notebook.txt", "a")


for fam_id, _ in enumerate(best):
    # loop over each family choice
    print('fam_id ',fam_id)
    f.write('fam_id '+str(fam_id)+'\n')
    for pick in range(10):
        day = choice_dict[f'choice_{pick}'][fam_id]
        temp = new.copy()
        temp[fam_id] = day # add in the new pick
        if cost_function_starter(temp) < start_score:
            new = temp.copy()
            start_score = cost_function_starter(new)
    print('best score ', start_score)        
    f.write('best score' +str(start_score)+'\n')
score = cost_function_starter(new)

print(f'Score: {score}')
f.close()

'''



data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
data_diff_sort=data_diff.sort_values(by='difficulty', ascending=False)
fam_id=data_diff_sort['family_id'].to_numpy()
best_sort=data['choice_0'].to_numpy()

best_score=cost_function(best)
new = best.copy()
print('init : ',cost_function_starter(new))
for i in fam_id[:5000]:
    temp = new.copy()
    temp[i]=best_sort[i]
    temp_score=cost_function_starter(temp)
    
    if temp_score < best_score:
        print(i, fam_id[i],temp_score)
        new = temp.copy()
        best_score = temp_score
     


plt.figure(figsize=(18,10))
sns.countplot(new)





'''

data_diff = pd.read_csv(os.path.join(mug_data_path,'familys_difficult.csv'))
data_diff_sort=data_diff.sort_values(by='difficulty', ascending=False)
data_diff_sort=data_diff_sort.reset_index(drop=True)
best_sort=data_diff_sort['choice_0'].to_numpy()
family_size_dict_sort = data_diff_sort[['n_people']].to_dict()['n_people']
choice_dict_sort = data_diff_sort[cols].to_dict()
data_diff_sort['fam_id_cp']=data_diff_sort['family_id']
data_diff_sort['family_id']=data_diff_sort.index

best_score_sort = cost_function_starter(best_sort)


for fam_id, _ in enumerate(best_sort):
    # loop over each family choice
    print('fam_id ',fam_id)

    for pick in range(10):
        day = choice_dict_sort[f'choice_{pick}'][fam_id]
        temp = new.copy()
        temp[fam_id] = day # add in the new pick
        if cost_function_starter(temp) < best_score_sort:
            new = temp.copy()
            best_score = cost_function(new)
    print('best score ', best_score)  
      

plt.figure(figsize=(18,10))
sns.countplot(best_sort)

score = cost_function(new)
'''