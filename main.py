# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:13:01 2020

@author: Lucas English
"""
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import queue
import threading
import matplotlib.animation as animation
import sounddevice as sd

import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import time

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# Imports audio as wave from filename as returns [sample rate, data] where data is in flolat format
def import_audio_wav(filename):
    [fs, x] = wavfile.read(filename)
    x = x.astype(float)
    x = x/np.amax(x)
    return [fs, x]

def generate_lowpass_filter(fs, data, cutoff):
    N = len(data)//2 + 1
    length = round(cutoff/fs * N * 2)
    fil = np.concatenate([np.ones(length), np.zeros(N - (length))])
    return fil
    
def generate_highpass_filter(fs, data, cutoff):
     N = len(data)//2 + 1
     length = round(cutoff/fs * N * 2)
     fil = np.concatenate([np.zeros(length), np.ones(N - (length))])
     return fil

def generate_bandpass_filter(fs, data, start, stop):
    N = len(data)//2 + 1
    length_start = round(start/fs * N * 2)
    length_stop = round(stop/fs * N * 2)
    fil = np.concatenate([np.zeros(length_start), np.ones(length_stop - length_start), np.zeros(N - length_stop)])
    return fil

def apply_filter(fs, data, filter_value):
    filtered_audio = np.fft.rfft(data) * filter_value
    return np.fft.irfft(filtered_audio).real

def combine_filter(filter_value1, filter_value2):
    return filter_value1 * filter_value2

def apply_impulse_response(data, response):
    tmp = signal.convolve(data, response)
    tmp = tmp * 1/(np.max(abs(tmp)))
    return tmp

def frequency_transform(fs, data):
    return [np.fft.rfftfreq(len(data), d=1/fs), np.fft.rfft(data)]

def frequency_shift(fs, data, delta):
    [freq, n] = frequency_transform(fs, data)
    N = len(n)
    shift = abs(int(delta/fs * N * 2))
    if (delta > 0):
        result = np.concatenate([np.zeros(shift), n[0:N-shift]])
    else:
        result = np.concatenate([n[shift:N], np.zeros(shift)])
    return time_transform(result)

def frequency_scale(fs, data, delta):
    [freq, n] = frequency_transform(fs, data)
    return time_transform(signal.resample(data, delta*n))

def time_transform(data):
    return np.fft.irfft(data)

def resample(data, n):
    return signal.resample(data, n)

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.l=Label(top,text="Length of Recording (Seconds)")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Enter',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class MainApplication(tk.Frame):
    
    def draw_fig(self):
        self.lna.set_xdata(np.r_[1:len(self.audio_file)+1]*1/self.fs)
        self.lna.set_ydata(self.audio_file)
        self.plot_time.set_xlim(0,len(self.audio_file)/self.fs)
        self.plot_time.set_ylim(-2,2)
        self.plot_time.set_xlabel('Time (sec)')
        plt.pause(0.0001)
        self.fig1.canvas.draw()
        self.canvasA.draw()
        #self.canvasA.get_tk_widget().grid(column=0, row=3, columnspan=3)
    
    def draw_freq(self):
        [freq, n] = frequency_transform(self.fs, self.audio_file)
        self.lnb.set_xdata(freq)
        self.lnb.set_ydata(20*np.log10(abs(n)))
        self.plot_freq.set_xlim(freq[0],freq[len(freq)-1])
        self.plot_freq.set_ylim(-80, 80)
        self.plot_freq.set_xlabel('Frequency (Hz)')
        print("drawing now")
        print(self.lnb,)
        self.fig2.canvas.draw()
        self.canvasB.draw()
        #self.canvasB.get_tk_widget().grid(column=0, row=4, columnspan=3)
        
    def open_wav(self):
        file = filedialog.askopenfilename(filetypes = (("Audio files","*.wav"),("all files","*.*")))
        [self.fs, self.audio_file] = import_audio_wav(file)
        self.original_audio = self.audio_file
        self.stop = 0
        self.draw_fig() 
        self.draw_freq()
    
    def updateGraphsB(self, i):
        if (i == 0):
            self.canvasB.draw()
        if self.stop == 1:
            self.stop = 0
            self.aniB.event_source.stop()
            self.draw_freq()
            #return self.lnb,
        self.player.write(self.audio_file[i*self.CHUNK:(i+1)*self.CHUNK].astype(np.float32).tostring(), self.CHUNK)
        [freq, n] = frequency_transform(self.fs, self.audio_file[i*self.CHUNK:(i+1)*self.CHUNK])
        self.lnb.set_xdata(freq)
        self.lnb.set_ydata(20*np.log10(abs(n)))
        if (i == int(len(self.audio_file)/self.CHUNK)-1):
            self.draw_freq()
        return self.lnb,

    def play_wav(self):
        self.p = pyaudio.PyAudio()
        self.player = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.RATE, output=True, frames_per_buffer=self.CHUNK)
        self.stop = 0
        self.aniB = animation.FuncAnimation(self.fig2, self.updateGraphsB, frames=int(len(self.audio_file)/self.CHUNK), interval=self.CHUNK/self.fs, blit=True, repeat=False)
        #self.draw_freq()
        
    def stop_wav(self):
        self.stop = 1
        
    def close_wav(self):
        self.player.stop_stream()
        self.player.close()
        
    def reset(self):
        self.audio_file = self.original_audio
        self.draw_freq()
        self.draw_fig()
        
    def record(self):
        self.fs = 44100
        self.window=popupWindow(self.master)
        self.master.wait_window(self.window.top)
        self.audio_file = sd.rec(int(self.window.value) * self.fs, samplerate=self.fs, channels=1)[:,0]
        self.original_audio = self.audio_file
        sd.wait()
        self.audio_file = self.audio_file * 1/np.max(abs(self.audio_file))
        self.original_audio = self.audio_file
        self.stop = 0
        self.draw_fig()
        self.draw_freq()
        
    def adjust_speed(self):
        self.audio_file = resample(self.audio_file, int((1-self.w.get())*len(self.audio_file)))
        self.draw_fig()
        self.draw_freq()
    
    def adjust_pitch(self):
        self.audio_file = frequency_shift(self.fs, self.audio_file, int(self.w2.get() * self.fs/32))
        self.draw_fig()
        self.draw_freq()
         
    def low_pass(self):
        fil = generate_lowpass_filter(self.fs,self.audio_file,int(self.lpfilt_cut.get()))
        self.audio_file = apply_filter(self.fs, self.audio_file, fil)
        self.draw_fig()
        self.draw_freq()
        
    def high_pass(self):
        fil = generate_highpass_filter(self.fs,self.audio_file,int(self.hpfilt_cut.get()))
        self.audio_file = apply_filter(self.fs, self.audio_file, fil)
        self.draw_fig()
        self.draw_freq()
    
    def band_pass(self):
        fil = generate_bandpass_filter(self.fs,self.audio_file,int(self.bpfilt_cutlow.get()), int(self.bpfilt_cuthigh.get()))
        self.audio_file = apply_filter(self.fs, self.audio_file, fil)
        self.draw_fig()
        self.draw_freq()
        
    def get_acoustic_response(self):
        self.sweep_response = sd.playrec(self.sweep, self.sweep_fs, channels=1)[:,0]
        time.sleep(25)
        freq_orig = frequency_transform(self.sweep_fs, self.sweep)[1]
        freq_after = frequency_transform(self.sweep_fs, self.sweep_response)[1]
        freq_impulse = np.divide(freq_after, freq_orig, out=np.zeros_like(freq_orig), where=freq_after!=0)
        self.audio_file = apply_filter(self.fs, self.audio_file, signal.resample(freq_impulse, len(self.audio_file)//2 + 1))
        self.audio_file = self.audio_file * 1/(np.max(abs(self.audio_file)))
        self.draw_fig()
        self.draw_freq()
        
    def sim_church(self):
        self.audio_file = apply_impulse_response(self.audio_file, self.church)
        self.draw_fig()
        self.draw_freq()
    
    def import_sim(self):   
        file = filedialog.askopenfilename(filetypes = (("Audio files","*.wav"),("all files","*.*")))
        [fs, response] = import_audio_wav(file)
        self.audio_file = apply_impulse_response(self.audio_file, response)
        self.draw_fig()
        self.draw_freq()
    
    def __init__(self, master):
        self.CHUNK = 2048
        self.RATE = 44100
        
        self.master = master
        self.frame = tk.Frame(self.master)
        self.master.title("ELEC3305 - HibikiAudio")
        self.master.geometry('800x600')
        
        [fs, x] = wavfile.read('sweep.wav')
        x = x.astype(float)
        x = x/np.amax(x)
        self.sweep = x
        self.sweep_fs = fs
        
        [fs, x] = wavfile.read('church.wav')
        x = x.astype(float)
        x = x/np.amax(x)
        self.church = x
        self.church_fs = fs

        open_btn = Button(master, text="Open Audio", command=self.open_wav)
        open_btn.grid(column=0, row=0)

        close_btn = Button(master, text="Close Audio", command=self.close_wav)
        close_btn.grid(column=1, row=0)
        
        reset_btn = Button(master, text="Reset Audio", command=self.reset)
        reset_btn.grid(column=2, row=0)

        play_btn = Button(master, text="Play", command=self.play_wav)
        play_btn.grid(column=0, row=1)

        pause_btn = Button(master, text="Stop", command=self.stop_wav)
        pause_btn.grid(column=1, row=1)

        stop_btn = Button(master, text="Record", command=self.record)
        stop_btn.grid(column=2, row=1)

        tab_control = ttk.Notebook(master)
        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='Filter')
        tab_control.grid(column = 0, row = 2, columnspan=3)

        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text='Equalizer')

        tab3 = ttk.Frame(tab_control)
        tab_control.add(tab3, text='Effect')
        
        lbl_speed = Label(tab3, text="Speed Up/Down")
        lbl_speed.grid(column = 0, row = 0)
        self.w = Scale(tab3, from_=-1, to=1, resolution=0.01, orient=HORIZONTAL)
        self.w.grid(column = 1, row = 0)
        speed_btn = Button(tab3, text = 'Apply', command=self.adjust_speed)
        speed_btn.grid(column = 2, row = 0)
        
        lbl_pitch = Label(tab3, text="Pitch Shift Up/Down")
        lbl_pitch.grid(column = 0, row = 1)
        self.w2 = Scale(tab3, from_=-1, to=1, resolution=0.01, orient=HORIZONTAL)
        self.w2.grid(column = 1, row = 1)
        pitch_btn = Button(tab3, text = 'Apply', command=self.adjust_pitch)
        pitch_btn.grid(column = 2, row = 1)
        
        lbl_acoustic_sim = Label(tab3, text="Room Simulation")
        lbl_acoustic_sim.grid(column = 0, row = 2)
        acoustic_sim_church_btn = Button(tab3, text = 'Church', command=self.sim_church)
        acoustic_sim_church_btn.grid(column = 1, row = 2)
        
        import_acoustic_sim_btn = Button(tab3, text = 'Import Acoustic Response', command=self.import_sim)
        import_acoustic_sim_btn.grid(column = 2, row = 2)
        
        lbl_lpfilt = Label(tab1, text="Low Pass Filter")
        lbl_lpfilt.grid(column=0, row=0)
        lbl_lpfilt_cut = Label(tab1, text="Cutoff Frequency:")
        lbl_lpfilt_cut.grid(column=1, row=0)
        
        self.lpfilt_cut = tk.Entry(tab1)
        self.lpfilt_cut.grid(column=2, row= 0)
        
        lpfilt_btn = Button(tab1, text='Apply', command=self.low_pass)
        lpfilt_btn.grid(column = 3, row = 0)
        
        lbl_hpfilt = Label(tab1, text="High Pass Filter")
        lbl_hpfilt.grid(column=0, row=1)
        lbl_hpfilt_cut = Label(tab1, text="Cutoff Frequency:")
        lbl_hpfilt_cut.grid(column=1, row=1)
        
        self.hpfilt_cut = tk.Entry(tab1)
        self.hpfilt_cut.grid(column=2, row= 1)
        
        hpfilt_btn = Button(tab1, text='Apply', command=self.high_pass)
        hpfilt_btn.grid(column = 3, row = 1)
        
        lbl_bpfilt = Label(tab1, text="Band Pass Filter")
        lbl_bpfilt.grid(column=0, row=2)
        lbl_bpfilt_cutlow = Label(tab1, text=" Low Cutoff Frequency:")
        lbl_bpfilt_cutlow.grid(column=1, row=2)
        lbl_bpfilt_cuthigh = Label(tab1, text=" High Cutoff Frequency:")
        lbl_bpfilt_cuthigh.grid(column=3, row=2)
        
        self.bpfilt_cutlow = tk.Entry(tab1)
        self.bpfilt_cutlow.grid(column=2, row= 2)
        self.bpfilt_cuthigh = tk.Entry(tab1)
        self.bpfilt_cuthigh.grid(column=4, row= 2)
        
        lbl_auto_eq = Label(tab2, text= "Auto Equalizer (Inverse Room Acoustic Response)")
        lbl_auto_eq.grid(column = 0, row = 0)
        auto_eq_btn = Button(tab2, text = "Apply", command=self.get_acoustic_response)
        auto_eq_btn.grid(column = 1, row = 0)
        
        bpfilt_btn = Button(tab1, text='Apply', command=self.band_pass)
        bpfilt_btn.grid(column = 5, row = 2)      
        
        self.fig1 = plt.figure(1, figsize=(5, 2), dpi=100)
        self.plot_time = self.fig1.add_subplot(111, xlabel = "Time (sec)")
        self.lna, = self.plot_time.plot([], [])
        
        self.fig2 = plt.figure(2, figsize=(5, 2), dpi=100)
        self.plot_freq = self.fig2.add_subplot(111, xlabel="Frequency (Hz)", ylabel="Amplitude (dB)")
        self.lnb, = self.plot_freq.plot([], [])
        self.fig1.tight_layout()
        self.fig2.tight_layout()
        
        self.canvasA = FigureCanvasTkAgg(self.fig1, self.master)
        self.canvasA.draw()
        self.canvasAwid = self.canvasA.get_tk_widget()
        self.canvasAwid.grid(column=0, row=3, columnspan=3)
        
        self.canvasB = FigureCanvasTkAgg(self.fig2, self.master)
        self.canvasB.draw()
        self.canvasBwid = self.canvasB.get_tk_widget()
        self.canvasBwid.grid(column=0, row=4, columnspan=3)

if __name__ == "__main__":        
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()