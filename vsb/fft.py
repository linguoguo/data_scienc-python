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





def print_fft_fault(ii):
    liste_columns=phase_indices(ii)
    signals=pd.read_csv('data_fault/'+str(ii)+'.csv')
    print(signals.head())
    align_phase = 0.25
    fig = plt.figure(figsize=(16, 9))
    plot_number = 0
    for signal_id in liste_columns:
        print('=== Signal {} ==='.format(signal_id))
        sig = signals[format(signal_id)]
        print(sig.head())
        fft_coeffs = get_fft_coeffs(sig)
        max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
        ps = get_max_coeff_phase(max_coeff)
        w, w_norm = get_instant_w(t, f0, ps)
        dominant_wave = amp * np.cos(w) # if np.sin(), then need to ajust by pi/2
        origin = get_align_idx(w_norm, align_value=align_phase)
    
    # roll signal and dominant wave
        sig_rolled = np.roll(sig, num_samples - origin)
        dominant_wave_rolled = np.roll(dominant_wave, num_samples - origin)
        
        
        
        
        plot_number += 1
        ax = fig.add_subplot(3, 2, plot_number)
    
        ax.plot(t * 1000, sig, label='Original') # original signal
        ax.plot(t * 1000, dominant_wave, color='red', label='Wave at {:.0f}Hz'.format(f0)) # wave at f0
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal {}'.format(signal_id))
    
        # plot phase
        plot_number += 1
        ax = fig.add_subplot(3, 2, plot_number)
        
        ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal
        ax.plot(t * 1000, dominant_wave_rolled, color='red', label='Rolled Wave at {:.0f}Hz'.format(f0)) # wave at f0
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('amptitude : '+str(amp))
    fig.savefig('data_fault_plot/'+str(ii)+'.png')
    #fig.savefig('plot.png')
    fig.tight_layout()
    
    
#print_fft_fault(1)    

for i in range(21,22):
    print(sig_fault[i])
    print_fft_fault(sig_fault[i])
    
    
def sort_signals(df):
    c=[]
    for i in range(0,len(df)):
        #print(df.iloc[i].sort_values(ascending=True)[1])
        c.append(df.iloc[i].sort_values(ascending=True)[1])
    return c    
sort_signals(signals)


    
def denoise(id):  
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    print(signals.head())
    fig = plt.figure(figsize=(16, 9))
    plot_number=0
    for signal_id in liste_columns:  
        sig = signals[format(signal_id)] 
        fft_coeffs = get_fft_coeffs(sig)
        max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
        ps = get_max_coeff_phase(max_coeff)
        w, w_norm = get_instant_w(t, f0, ps)
        origin = get_align_idx(w_norm, align_value=align_phase)
        sig_rolled = np.roll(sig, num_samples - origin) 
        signals[str(signal_id)]=sig_rolled
        print(signals.head())
        plot_number += 1
        ax = fig.add_subplot(3, 2,plot_number)
        
        ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal

    c=signals.median(axis=1)
    print(c.head())
    plot_number += 1
    ax = fig.add_subplot(3, 2,plot_number)
    ax.plot(t * 1000, c,color='red' ,label='Rolled Original')    
    ax.legend()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('amptitude : '+str(amp))    
    
    return c
cc=denoise(67)


def noise_signals(id):
    liste_columns=phase_indices(id)
    signals=pd.read_csv('data_fault/'+str(id)+'.csv') 
    fig = plt.figure(figsize=(16, 9))
    plot_number=0
    for signal_id in liste_columns:  
        sig = signals[format(signal_id)] 
        fft_coeffs = get_fft_coeffs(sig)
        max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
        ps = get_max_coeff_phase(max_coeff)
        w, w_norm = get_instant_w(t, f0, ps)
        origin = get_align_idx(w_norm, align_value=align_phase)    
        sig_rolled = np.roll(sig, num_samples - origin)
        dominant_wave_rolled = np.roll(dominant_wave, num_samples - origin)
        plot_number += 1
        ax = fig.add_subplot(3, 2, plot_number)    
    
    
    
'''
    signals['max']=signals.max(axis=1)
    signals['min']=signals.min(axis=1)
    print (signals.head(10))
'''

noise_signals(67)   


'''  
liste_columns=['228','229','230']
fig = plt.figure(figsize=(16, 9))
plot_number = 0 
for signal_id in liste_columns:
    # get samples
    print('=== Signal {} ==='.format(signal_id))
    sig = signals[signal_id]
    print(sig.head())
    # fft
    fft_coeffs = get_fft_coeffs(sig)
    
    # asses dominant frequency
    max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)
    
    # phase shift
    ps = get_max_coeff_phase(max_coeff)
    
    # get angular phase vector
    w, w_norm = get_instant_w(t, f0, ps)
    
    # generate dominant signal at f0
    dominant_wave = amp * np.cos(w) # if np.sin(), then need to ajust by pi/2
    (w)
    
    # plot signals
    plot_number += 1
    ax = fig.add_subplot(3, 2, plot_number)
    
    ax.plot(t * 1000, sig, label='Original') # original signal
    ax.plot(t * 1000, dominant_wave, color='red', label='Wave at {:.0f}Hz'.format(f0)) # wave at f0
    ax.legend()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal {}'.format(signal_id))
    
    # plot phase
    plot_number += 1
    ax = fig.add_subplot(3, 2, plot_number)
    
    ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal
    ax.plot(t * 1000, dominant_wave_rolled, color='red', label='Rolled Wave at {:.0f}Hz'.format(f0)) # wave at f0
    ax.legend()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('amptitude : '+str(amp))
    
fig.tight_layout()    
'''

