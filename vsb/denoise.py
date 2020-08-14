#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:20:06 2019

@author: lin
"""

# -*- coding: utf-8 -*-
"""
Lin GUO

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
#signals=pd.read_csv('data_subsets/0_6.csv')
meta_train = pd.read_csv('metadata_train.csv')
align_phase = 0.25

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
    return int(np.median(candidates))


def denoise_plt(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    print(signals.head())
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
        print(signals.head())
        plot_number += 1
        ax = fig.add_subplot(3, 2,plot_number)
        
        ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal

    c=signals.median(axis=1)
  #  print(c.head())
    plot_number += 1
    ax = fig.add_subplot(3, 2,plot_number)
    ax.plot(t * 1000, c,color='red' ,label='Rolled Original')    
    ax.legend()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('amptitude : '+str(amp)) 
    
    
    return c,l_origin
cc,ll=denoise_plt(1)



def denoise(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    #fig = plt.figure(figsize=(16, 9))
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
cc,ll=denoise(67)


def noise_plt(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    fig = plt.figure(figsize=(25, 9))
    plot_number=0
    for signal_id in liste_columns:  
        sig = signals[format(signal_id)] 
        fft_coeffs = get_fft_coeffs(sig)
        max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
        ps = get_max_coeff_phase(max_coeff)
        w, w_norm = get_instant_w(t, f0, ps)

        plot_number += 1
        ax = fig.add_subplot(2, 3, plot_number)
    
        ax.plot(t * 1000, sig, label='Original') # original signal
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal {}'.format(signal_id))

    sig_c,liste=denoise(id)
    for i in liste:
        sig_rolled = np.roll(sig_c,  i - num_samples) 
        plot_number += 1
        ax = fig.add_subplot(2, 3,plot_number)
        ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal
   
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        plt.show()


noise_plt(1)






def noise(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    sig_c,liste=denoise(id) 
    fig = plt.figure(figsize=(16, 9))
    plot_number=0
    
    for i in range(3):
        print(i)
        print(liste_columns[i])
        print(liste[i])
        sig = signals[format(liste_columns[i])]
        sig_c_rolled = np.roll(sig_c,  liste[i] - num_samples)
        
        print(sig.head())
        
        signals['noise_'+format(liste_columns[i])]=sig - sig_c_rolled
        
        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig, label='Original') # original signal
        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig_c_rolled, label='Original')
        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig - sig_c_rolled, label='Original')
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal {}'.format(liste_columns[i]))  
    
    l=['noise_'+format(liste_columns[0]),'noise_'+format(liste_columns[1]),'noise_'+format(liste_columns[2])]
    signals['noise_max-min']=signals[l].max(axis=1)-signals[l].min(axis=1)
    plot_number += 1
    ax = fig.add_subplot(4, 3, plot_number)
    ax.plot(t * 1000, signals['noise_max-min'], label='Original')
    plot_number += 1
    ax = fig.add_subplot(4, 3, plot_number)
    ax.hist(signals['noise_max-min'], bins=40) 
    plt.xlim(0,10)
    fig.tight_layout()  
    #fig.savefig('data_fault_plot/hist'+str(id)+'.png')
    print(signals.head())        

cc=noise(67)


for i in range(0,0):
    print(sig_fault[i])
    noise(sig_fault[i])


def noise_std(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    sig_c,liste=denoise(id) 
    fichier = open("data_fault.txt", "a")
    fichier.write(str(id))
    fichier.write('\n')
    
    
    for i in range(3):
        
        print('----------------------------------------')
        fichier.write("----------------------------------------\n")
        print('target',meta_train[meta_train['signal_id']==liste_columns[i]].target)
        fichier.write('target ')
        fichier.write(str(meta_train[meta_train['signal_id']==liste_columns[i]].target))
        fichier.write('\n')
        sig = signals[format(liste_columns[i])]
        sig_c_rolled = np.roll(sig_c,  liste[i] - num_samples)
        
        print(sig.head())
        noise=sig - sig_c_rolled
        
        signals['noise_'+format(liste_columns[i])]= noise 
        #signals['noise_std_'+format(liste_columns[i])]= noise
        std=noise.std()
        print('std:',std)
        fichier.write('std ')
        fichier.write(str(std))
        fichier.write('\n')
        
    
    l=['noise_'+format(liste_columns[0]),'noise_'+format(liste_columns[1]),'noise_'+format(liste_columns[2])]
    nmn=signals[l].max(axis=1)-signals[l].min(axis=1)
    signals['noise_max-min']=nmn
    #signals['nmn_std']=nmn.std()
    print('noise_max-min_std :',nmn.std())
    fichier.write('noise_max-min_std :')
    fichier.write(str(nmn.std()))
    fichier.write('\n')
    
    print('----------------------------------------\n')
    #fig.savefig('data_fault_plot/hist'+str(id)+'.png')
    print(signals.head()) 
    fichier.write('\n')
    fichier.write('\n')
    fichier.close()
    
    
    
    
cc=noise_std(67)    

for i in range(0,0):
    print(sig_fault[i])
    noise_std(sig_fault[i])
    
    
def transform_ts(ts, n_dim=1000):

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
    
    
def noisetest_fault(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    sig_c,liste=denoise(id) 
    fig = plt.figure(figsize=(16, 9))
    plot_number=0
    #print('epaisseur',transform_ts(sig_c))
    
    for i in range(3):
        print(i)
        print(liste_columns[i])
        print(liste[i])
        sig = signals[format(liste_columns[i])]
        sig_c_rolled = np.roll(sig_c,  liste[i] - num_samples)
        
        print(sig.head())
        noise=sig - sig_c_rolled
        signals['noise_'+format(liste_columns[i])]=noise
        
        print(noise.describe())
        nn=np.where(np.absolute(noise)>10)
        print('yoho',np.unique(np.asarray(nn)//1000))
        print('biubiu',np.count_nonzero(np.where(np.absolute(noise)>40)))
        
        
        epaisseur=transform_ts(sig_c_rolled)
        
        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig, label='Original') # original signal
        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, sig_c_rolled, label='Original')
        ax.plot(t * 1000,epaisseur[0]+3*epaisseur[1],color='red',label='noise std top')
        ax.plot(t * 1000,epaisseur[0]-3*epaisseur[1],color='red',label='noise std bot')
        plot_number += 1
        ax = fig.add_subplot(4, 3, plot_number)
        ax.plot(t * 1000, noise, label='noise')

        
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal {}'.format(liste_columns[i]))          
    
    l=['noise_'+format(liste_columns[0]),'noise_'+format(liste_columns[1]),'noise_'+format(liste_columns[2])]
    signals['noise_max-min']=signals[l].max(axis=1)-signals[l].min(axis=1)
    plot_number += 1
    ax = fig.add_subplot(4, 3, plot_number)
    ax.plot(t * 1000, signals['noise_max-min'], label='Original')
    plot_number += 1
    ax = fig.add_subplot(4, 3, plot_number)
    ax.hist(signals['noise_max-min'], bins=40)
    plt.xlim(0,10)
    fig.tight_layout()
    #fig.savefig('data_no_fault_plot/hist_max_min'+str(id)+'.png')
    print(l)
    print(signals.head()) 
    print(signals[l].head().max(axis=1))   
    return signals    

cc=noisetest_fault(96)      
  

def noisetest_fault_0(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    sig_c,liste=denoise(id) 
    epaisseur=transform_ts(sig_c)
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

cc=noisetest_fault_0(90) 







  