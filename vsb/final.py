#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:22:05 2019

@author: lin
"""
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq # Used to read the data
import os
import numpy as np
from scipy import fftpack

train = pq.read_pandas('train.parquet').to_pandas()
meta_train = pd.read_csv('metadata_train.csv')
num_samples = 800000
period = 0.02 # over a 20ms period
fs = num_samples / period # 40MHz sampling rate
# time array support
t = np.array([i / fs for i in range(num_samples)])
align_phase = 0.25
max_num = 127
min_num = -128
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


def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

def denoise_courbe(id):
    liste_columns=phase_indices(id)
    signals=train.iloc[:,list(liste_columns) ]
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

c,l=denoise_courbe(4)

def std_mean(ts, n_dim=1000):
    bucket_size = int(num_samples / n_dim)
    new_ts_std = []
    new_ts_mean = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, num_samples, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts[i:i + bucket_size]
        # calculate each feature
        std = ts_range.std()
        mean = ts_range.mean()
        new_ts_std += bucket_size * [std]
        new_ts_mean += bucket_size * [mean]
            
    return np.concatenate(([new_ts_mean], [new_ts_std]), axis=0)

def noise(id):
    liste_columns = phase_indices(id)
    signals = train.iloc[:, list(liste_columns)]

    sig_c, liste = denoise_courbe(id)
    epaisseur = std_mean(sig_c)

    std_top = epaisseur[0] + 3 * epaisseur[1]
    std_bot = epaisseur[0] - 3 * epaisseur[1]

    noise = pd.DataFrame()
    for i in range(3):
        id = format(liste_columns[i])
        sig = signals[id]
        std_top_rolled = np.roll(std_top, liste[i] - num_samples)
        std_bot_rolled = np.roll(std_bot, liste[i] - num_samples)

        n = np.asarray(sig).copy()
        l_top = np.where(sig > std_top_rolled)
        l_bot = np.where(sig < std_bot_rolled)

        m = np.zeros(num_samples)
        m[l_top] = n[l_top] - std_top_rolled[l_top]
        m[l_bot] = n[l_bot] - std_bot_rolled[l_bot]

        noise[id] = m

    return noise


sig = noise(2)




def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    bucket_size = int(num_samples / n_dim)
    new_mean = []
    new_std=[]
    for i in range(0, num_samples, bucket_size):
        ts_r = ts[i:i + bucket_size]
        l=ts_r.nonzero()
        if  len(l[0])>0:
            ts_range=ts_r.iloc[l]
            mean = ts_range.mean()
            std = ts_range.std() 
        else:
            mean=std=0
        #print(np.asarray([mean, std]))    
        new_mean.append(mean)
        new_std.append(std)
    #return new_mean,new_std   
    a=np.asarray(new_mean)
    b=np.asarray(new_std)
    return np.asarray(new_mean),np.asarray(new_std)
    #return new_mean+new_std


aa,t1=transform_ts(sig['6'])

bb,t2=transform_ts(sig['7'])
cc,t3=transform_ts(sig['8'])



for i in range(76,77):
    sig=noise(i)
    liste_columns=phase_indices(i)
    aa,sta=transform_ts(sig[str(liste_columns[0])])
    bb,stb=transform_ts(sig[str(liste_columns[1])])
    cc,stc=transform_ts(sig[str(liste_columns[2])])
    at=meta_train[meta_train['signal_id']==liste_columns[0]].target
    bt=meta_train[meta_train['signal_id']==liste_columns[1]].target
    ct=meta_train[meta_train['signal_id']==liste_columns[2]].target
    print('target :\n',at,bt,ct)
    print(np.corrcoef([aa,bb,cc]))
    print(np.corrcoef([sta,stb,stc]))
    

def corr(id, n_dim=5):
    print('-----------')
    sig=noise(id)
    bucket_size = int(160 / n_dim)
    liste_columns=phase_indices(id)
    at=meta_train[meta_train['signal_id']==liste_columns[0]].target
    bt=meta_train[meta_train['signal_id']==liste_columns[1]].target
    ct=meta_train[meta_train['signal_id']==liste_columns[2]].target
    aa,sta=transform_ts(sig[str(liste_columns[0])])
    bb,stb=transform_ts(sig[str(liste_columns[1])])
    cc,stc=transform_ts(sig[str(liste_columns[2])])
    fig = plt.figure(figsize=(10, 5))
    plt.plot( sta, label='Original')
    plt.plot( stb, label='Original')
    plt.plot( stc, label='Original')
    #plt.close()
    print('target :\n',at,bt,ct)
    new_std=[]
   # meme=(sta.std()+stb.std()+stc.std())
    meme=(aa.mean()+bb.mean()+cc.mean()+sta.mean()+stb.mean()+stc.mean())/3
    print('std?',sta.std(),stb.std(),stc.std())
    print('meme',meme)
    for i in range(0, 160, bucket_size):
        st1 = sta[i:i + bucket_size]
        st2 = stb[i:i + bucket_size]
        st3 = stc[i:i + bucket_size]
        #print(np.corrcoef([st1,st2,st3]))
        p1=st1.mean()+st1.std()
        p2=st2.mean()+st2.std()
        p3=st3.mean()+st3.std()
        p=np.array([p1,p2,p3])
        if np.any(p>meme):
            print('---')
            print('oui,',p)
            new_std.append(np.corrcoef([st1,st2,st3]).min())
        else:
            new_std.append(1)
    return new_std

print(corr())    


if np.any(b>2):
    print ('oui')
































def denoise_plot(id):
    liste_columns = phase_indices(id)
    signals = train.iloc[:, list(liste_columns)]
    sig_c, liste = denoise_courbe(id)
    epaisseur = std_mean(sig_c)
    fig = plt.figure(figsize=(16, 9))
    plot_number = 0
    signals['std_top'] = epaisseur[0] + 3 * epaisseur[1]
    signals['std_bot'] = epaisseur[0] - 3 * epaisseur[1]

    std_top = epaisseur[0] + 3 * epaisseur[1]
    std_bot = epaisseur[0] - 3 * epaisseur[1]
    # print('epaisseur',transform_ts(sig_c))

    for i in range(3):

        id = format(liste_columns[i])
        sig = signals[format(liste_columns[i])]
        sig_c_rolled = np.roll(sig_c, liste[i] - num_samples)
        std_top_rolled = np.roll(std_top, liste[i] - num_samples)
        std_bot_rolled = np.roll(std_bot, liste[i] - num_samples)

        n = np.asarray(sig).copy()
        l_top = np.where(sig > std_top_rolled)
        l_bot = np.where(sig < std_bot_rolled)

        m = np.zeros(num_samples)
        m[l_top] = n[l_top] - std_top_rolled[l_top]
        m[l_bot] = n[l_bot] - std_bot_rolled[l_bot]

        signals['noise_' + id] = m

        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig, label='Original')  # original signal

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


cc = denoise_plot(2)



