#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:22:05 2019

@author: lin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import pyarrow.parquet as pq
import os
import csv
from tqdm import tqdm # Processing time measurement
from sklearn.metrics import matthews_corrcoef

train = pq.read_pandas('train.parquet').to_pandas()
meta_train = pd.read_csv('metadata_train.csv')
df_train = pd.read_csv('metadata_train.csv')
df_train = df_train.set_index(['id_measurement', 'phase'])
align_phase = 0.25
max_num = 127
min_num = -128
def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

sig_fault=meta_train[meta_train['target']==1].id_measurement.unique()

num_samples = 800000
period = 0.02 # over a 20ms period
fs = num_samples / period # 40MHz sampling rate
# time array support
t = np.array([i / fs for i in range(num_samples)])

# frequency vector fro FFT
freqs = fftpack.fftfreq(num_samples, d=1/fs)

# get fft coeffs
def get_fft_coeffs(sig):
    return fftpack.fft(sig)

# get coeff with highest norm
def get_highest_coeff(fft_coeffs, freqs, verbose=True):
    coeff_norms = np.abs(fft_coeffs) # get norms (fft coeffs are complex)
    max_idx = np.argmax(coeff_norms)
    max_coeff = fft_coeffs[max_idx] # get max coeff
    max_freq = freqs[max_idx] # assess which is the dominant frequency
    max_amp = (coeff_norms[max_idx] / num_samples) * 2 # times 2 because there are mirrored freqs
    if verbose:
        print('Dominant frequency is {:,.1f}Hz with amplitude of {:,.1f}\n'.format(max_freq, max_amp))
    
    return max_coeff, max_amp, max_freq

# get max coeff phase
def get_max_coeff_phase(max_coeff):
    return np.angle(max_coeff)

# construct the instant angular phase vector indexed by pi, i.e. ranges from 0 to 2
def get_instant_w(time_vector, f0, phase_shift):
    w_vector = 2 * np.pi * time_vector * f0 + phase_shift
    w_vector_norm = np.mod(w_vector / (2 * np.pi), 1) * 2 # range between cycle of 0-2 
    return w_vector, w_vector_norm

# find index of chosen phase to align
def get_align_idx(w_vector_norm, align_value=0.5):
    candidates = np.where(np.isclose(w_vector_norm, align_value))
    # since we are in discrete time, threre could be many values close to the desired one
    # so let's take the one in the middle
    atol=1e-08
    while len(candidates[0])==0:
        #print ('yo')
        atol*=10
        candidates = np.where(np.isclose(w_vector_norm, align_value,atol))
    return int(np.median(candidates))
#get_align_idx(w_norm, align_value=0.5)

def denoise(id):  
    liste_columns=phase_indices(id)
    #signals=pd.read_csv('data/'+str(id)+'.csv') 
    signals=train.iloc[:,list(liste_columns) ] 
    plot_number=0
    l_origin=[]
    for signal_id in liste_columns:  
        print(signal_id)
        sig = signals[format(signal_id)] 
        fft_coeffs = get_fft_coeffs(sig)
        max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
        ps = get_max_coeff_phase(max_coeff)
        w, w_norm = get_instant_w(t, f0, ps)
        origin = get_align_idx(w_norm, align_value=align_phase)
        l_origin.append(origin)
        sig_rolled = np.roll(sig, num_samples - origin) 
        signals[str(signal_id)]=sig_rolled
    c=signals.median(axis=1)    
    return c,l_origin

#c=denoise(48)


def denoise_train(id,signals):  
    liste_columns=phase_indices(id) 
    fig = plt.figure(figsize=(16, 9))
    plot_number=0
    l_origin=[]
    for signal_id in liste_columns:  
        sig = signals[format(signal_id)] 
        fft_coeffs = get_fft_coeffs(sig)
        max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
        ps = get_max_coeff_phase(max_coeff)
        w, w_norm = get_instant_w(t, f0, ps)
        origin = get_align_idx(w_norm, align_value=align_phase)
        l_origin.append(origin)
        sig_rolled = np.roll(sig, num_samples - origin) 
        signals[str(signal_id)]=sig_rolled
    c=signals.median(axis=1)    
    return c,l_origin

def std_ts(ts, n_dim=1000):

    bucket_size = int(num_samples / n_dim)
    new_ts_std = []
    new_ts_mean=[]
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, num_samples, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts[i:i + bucket_size]
        # calculate each feature
        std = ts_range.std()
        mean = ts_range.mean()
        new_ts_std+=bucket_size*[std]
        new_ts_mean+=bucket_size*[mean]
        
    return np.concatenate(([new_ts_mean],[new_ts_std]), axis=0)    


def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]

def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    bucket_size = int(num_samples / n_dim)

    new_ts = []
    #a=0
    for i in range(0, num_samples, bucket_size):
        # cut each bucket to ts_range
        ts_r = ts[i:i + bucket_size]
        l=ts_r.nonzero()
        #print(l)
         
       
        if  len(l[0])>0:
            #print('0?')
            #print(len(l[0]))
            #a+=len(l[0])
           
            ts_range=ts_r.iloc[l]
           # print(ts_range.head())
            mean = ts_range.mean()
            std = ts_range.std() # standard deviation
            std_top = mean + std # I have to test it more, but is is like a band
            std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
            percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 
            max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
            relative_percentile = percentil_calc - mean
        else:
            #print('yi')
            mean=std=std_top=std_bot=max_range=0
            percentil_calc = np.percentile(0, [0, 1, 25, 50, 75, 99, 100])
            relative_percentile = percentil_calc - mean
            #print('mean')
            

        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))
    #print (a)
    return np.asarray(new_ts)


#dd=transform_ts(train.iloc[:,9 ] )
def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def noisetest_fault_0(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data/'+str(id)+'.csv') 
    sig_c,liste=denoise(id) 
    epaisseur=std_ts(sig_c)
    max=signals.max().max()
    min=signals.min().min()
    fig = plt.figure(figsize=(16, 9))
    plot_number=0
    signals['std_top']=epaisseur[0]+3*epaisseur[1]
    signals['std_bot']=epaisseur[0]-3*epaisseur[1]
    
    std_top=epaisseur[0]+3*epaisseur[1]
    std_bot=epaisseur[0]-3*epaisseur[1]
    #print('epaisseur',transform_ts(sig_c))
    
    for i in range(3):
        print(i)
        print(liste_columns[i])
        print(liste[i])
        id=format(liste_columns[i])
        sig = signals[format(liste_columns[i])]

        sig_c_rolled = np.roll(sig_c,  liste[i] - num_samples)
        std_top_rolled=np.roll(std_top,  liste[i] - num_samples)
        std_bot_rolled=np.roll(std_bot,  liste[i] - num_samples)
        

        n=np.asarray(sig).copy()
        l_top=np.where(sig>std_top_rolled)
        l_bot=np.where(sig<std_bot_rolled)

        m=np.zeros(num_samples)
        m[l_top]=n[l_top]-std_top_rolled[l_top]
        m[l_bot]=n[l_bot]-std_bot_rolled[l_bot]

        m=min_max_transf(m, min, max, range_needed=(-1, 1))
        signals['noise_'+id]=m

        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig, label='Original') # original signal

        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig_c_rolled, label='Original')

        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, m, label='noise')
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal {}'.format(liste_columns[i]))         
    fig.tight_layout()

    print(signals.head()) 
  
    return signals    

#cc=noisetest_fault_0(4)


def denoise_csv(id):  
    liste_columns=phase_indices(id) 
    signals=train.iloc[:,list(liste_columns) ] 
    #print(df.head())
    print(signals.head())
    sig_c,liste=denoise(id) 
    epaisseur=std_ts(sig_c)
    plot_number=0    
    std_top=epaisseur[0]+3*epaisseur[1]
    std_bot=epaisseur[0]-3*epaisseur[1]
   # print('std')
    #print(len(std_top))
    
    #print('epaisseur',transform_ts(sig_c))
    
    for i in range(3):
        id=format(liste_columns[i])
        sig = signals[id]
        sig_c_rolled = np.roll(sig_c,  liste[i] - num_samples)
        std_top_rolled=np.roll(std_top,  liste[i] - num_samples)
        std_bot_rolled=np.roll(std_bot,  liste[i] - num_samples)
        
        n=np.asarray(sig).copy()
        l_top=np.where(sig>std_top_rolled)
        l_bot=np.where(sig<std_bot_rolled)

        m=np.zeros(num_samples)
        m[l_top]=n[l_top]-std_top_rolled[l_top]
        m[l_bot]=n[l_bot]-std_bot_rolled[l_bot]

        signals['noise_'+id]=m
    #name='data_noise/'+str(id)+'.csv'
    #signals.to_csv(name,index=False)  
    return signals

#cc=denoise_csv(90)


def noise(id):  
    liste_columns=phase_indices(id) 
    signals=train.iloc[:,list(liste_columns) ] 
    #print(df.head())
   # print(signals.head())
    sig_c,liste=denoise(id) 
    epaisseur=std_ts(sig_c)
    plot_number=0    
    std_top=epaisseur[0]+3*epaisseur[1]
    std_bot=epaisseur[0]-3*epaisseur[1]

    max = signals.max().max()
    min = signals.min().min()
    noise= pd.DataFrame()
    for i in range(3):
        id=format(liste_columns[i])
        sig = signals[id]
        sig_c_rolled = np.roll(sig_c,  liste[i] - num_samples)
        std_top_rolled=np.roll(std_top,  liste[i] - num_samples)
        std_bot_rolled=np.roll(std_bot,  liste[i] - num_samples)
        
        n=np.asarray(sig).copy()
        l_top=np.where(sig>std_top_rolled)
        l_bot=np.where(sig<std_bot_rolled)

        m=np.zeros(num_samples)
        m[l_top]=n[l_top]-std_top_rolled[l_top]
        m[l_bot]=n[l_bot]-std_bot_rolled[l_bot]
        m = min_max_transf(m, min, max, range_needed=(-1, 1))
        noise[id]=m
    #name='data_noise/'+str(id)+'.csv'
    #signals.to_csv(name,index=False)  
    return noise

#cc=noise(2)











def prep_data(start, end):
    # load a piece of data from file
    praq_train = pq.read_pandas('train.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    print(praq_train.head())
    # using tdqm to evaluate processing time
    # takes each index from df_train and iteract it from start to end
    # it is divided by 3 because for each id_measurement there are 3 id_signal, and the start/end parameters are id_signal
    for id_measurement in tqdm(df_train.index.levels[0].unique()[int(start/3):int(end/3)]):
        X_signal = []
        # for each phase of the signal
        #print(id_measurement)
        df=noise(id_measurement)
        #print('----')
        #print(df.head())
        for phase in [0,1,2]:
            # extract from df_train both signal_id and target to compose the new data sets
            signal_id, target = df_train.loc[id_measurement].loc[phase]
            #print('signal_id')
            # but just append the target one time, to not triplicate it
            print('id,target :',signal_id, target)
            if phase == 0:
                y.append(target)
            # extract and transform data into sets of features
            xx=transform_ts(df[str(signal_id)])
            print(xx[0])
            X_signal.append(xx)
        # concatenate all the 3 phases in one matrix
        X_signal = np.concatenate(X_signal, axis=1)
        # add the data to X
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

#prep_data(3, 6)


#y=np.load('y.npy')

#y_val=np.load('y_val.npy')
#preds_val=np.load('preds_val.npy')

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = matthews_corrcoef(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result

#best_threshold = threshold_search(y_val, preds_val)['threshold']





'''
def load_all():
    total_size = len(df_train)
    #total_size = 6
    for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
        X_temp, y_temp = prep_data(ini, end)
        X.append(X_temp)
        y.append(y_temp)
load_all()
X = np.concatenate(X)
y = np.concatenate(y)

print(X.shape, y.shape)
# save data into file, a numpy specific format
np.save("X_normalise_chacun.npy",X)
np.save("y.npy",y)

'''


p=[]
for i in range(160):
    p.append(X[0][i])