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
signals=pd.read_csv('data_subsets/0_6.csv') 
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
        ax = fig.add_subplot(3, 3, plot_number)
        ax.plot(t * 1000, sig, label='Original') # original signal
        plot_number += 1
        ax = fig.add_subplot(3, 3, plot_number)
        ax.plot(t * 1000, sig_c_rolled, label='Original')
        plot_number += 1
        ax = fig.add_subplot(3, 3, plot_number)
        ax.plot(t * 1000, sig - sig_c_rolled, label='Original')
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal {}'.format(liste_columns[i]))  
    fig.tight_layout()  
    fig.savefig('data_fault_plot/'+str(id)+'.png')
    print(signals.head())        

cc=noise(67)


for i in range(0,1):
    print(sig_fault[i])
    noise(sig_fault[i])


