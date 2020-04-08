# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:13:01 2020

@author: Lucas English
"""
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def prepareWav(data):
    tmp = data.astype(float)
    tmp = tmp/np.amax(tmp)
    return tmp

def generate_lowpass_filter(fs, data, cutoff):
    N = len(data)
    length = round(cutoff/fs * N)
    fil = np.concatenate([np.ones(length), np.zeros(N - (2*length)), np.ones(length)])
    filtered_audio = np.fft.fft(data) * fil
    return np.fft.ifft(filtered_audio).real

def generate_highpass_filter(fs, data, cutoff):
    N = len(data)
    length = round(cutoff/fs * N)
    fil = np.concatenate([np.zeros(length), np.ones(N - (2*length)), np.zeros(length)])
    filtered_audio = np.fft.fft(data) * fil
    return np.fft.ifft(filtered_audio).real

def generate_bandpass_filter(fs, data, start, stop):
    N = len(data)
    length_start = round(start/fs * N)
    length_stop = round(stop/fs * N)
    fil = np.concatenate([np.zeros(length_start), np.ones(length_stop - length_start), np.zeros(N - 2*length_stop), np.ones(length_stop - length_start), np.zeros(length_start)])
    filtered_audio = np.fft.fft(data) * fil
    return np.fft.ifft(filtered_audio).real
    
#def generate_highpass_filter(cutoff):
    

TheFile = 'audio.wav'
[fs, x] = wavfile.read(TheFile)
[fs2, x2] = wavfile.read('church.wav')
x = prepareWav(x)
x2 = prepareWav(x2)
x = np.concatenate([x, np.zeros(len(x2)-len(x))])

a = np.fft.fft(x)
b = np.fft.fft(x2)

c = generate_lowpass_filter(fs, x, 1500)
myrecording = sd.rec(int(5 * fs), dtype='float64', samplerate = 44100, channels = 1)[:,0]
sd.wait()  # Wait until recording is finished
