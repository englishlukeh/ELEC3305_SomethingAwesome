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
import matplotlib.animation as animation
import sounddevice as sd
from tkinter import *

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import time

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# Imports audio as wave from filename and returns [sample rate, data] where data is in float format
def import_audio_wav(filename):
    [fs, x] = wavfile.read(filename)
    x = x.astype(float)
    x = x/np.amax(x)
    return [fs, x]

# Generates low pass filter at cutoff frequency and returns filter in frequnecy domain
def generate_lowpass_filter(fs, data, cutoff):
    N = len(data)//2 + 1
    length = round(cutoff/fs * N * 2)
    fil = np.concatenate([np.ones(length), np.zeros(N - (length))])
    return fil
    
# Generates high pass filter at cutoff requency and returns frequency in frequency domain
def generate_highpass_filter(fs, data, cutoff):
     N = len(data)//2 + 1
     length = round(cutoff/fs * N * 2)
     fil = np.concatenate([np.zeros(length), np.ones(N - (length))])
     return fil

# Generates bandpass filter, passing from start freq to stop freq and returns filter in frequency domain
def generate_bandpass_filter(fs, data, start, stop):
    N = len(data)//2 + 1
    length_start = round(start/fs * N * 2)
    length_stop = round(stop/fs * N * 2)
    fil = np.concatenate([np.zeros(length_start), np.ones(length_stop - length_start), np.zeros(N - length_stop)])
    return fil

# Applies filter to data and returns time domain result
def apply_filter(fs, data, filter_value):
    filtered_audio = np.fft.rfft(data) * filter_value
    return np.fft.irfft(filtered_audio).real

# Convolves an impulse response with data and returns the output
def apply_impulse_response(data, response):
    tmp = signal.convolve(data, response)
    tmp = tmp * 1/(np.max(abs(tmp)))
    return tmp

# Returns the frequency transform of data and the frequency bins associated with the dft
def frequency_transform(fs, data):
    return [np.fft.rfftfreq(len(data), d=1/fs), np.fft.rfft(data)]

# Returns the time domain of data that has been shifted in the frequency domain
def frequency_shift(fs, data, delta):
    [freq, n] = frequency_transform(fs, data)
    N = len(n)
    shift = abs(int(delta/fs * N * 2))
    if (delta > 0):
        result = np.concatenate([np.zeros(shift), n[0:N-shift]])
    else:
        result = np.concatenate([n[shift:N], np.zeros(shift)])
    return time_transform(result)

# Returns the time domain of data that has been resampled in frequency domain
def frequency_scale(fs, data, delta):
    [freq, n] = frequency_transform(fs, data)
    return time_transform(signal.resample(data, delta*n))

# Returns the time domain of frequency domain data
def time_transform(data):
    return np.fft.irfft(data)

# Returns data resampled to n samples
def resample(data, n):
    return signal.resample(data, n)

# Class of the popup window for entering length of recording
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

# Class for the main application
class MainApplication(tk.Frame):
    
    # Draw time domain figure
    def draw_fig(self):
        self.lna.set_xdata(np.r_[1:len(self.audio_file)+1]*1/self.fs)
        self.lna.set_ydata(self.audio_file)
        self.plot_time.set_xlim(0,len(self.audio_file)/self.fs)
        self.plot_time.set_ylim(-2,2)
        self.plot_time.set_xlabel('Time (sec)')
        self.fig1.canvas.draw()
        self.canvasA.draw()
        self.canvasA.get_tk_widget().grid(column=0, row=3, columnspan=3)
    
    # Draw frequency domain figure
    def draw_freq(self):
        # we need to do this because matplotlib's animation doesn't work as it should
        self.plot_freq.clear()
        self.lnb, = self.plot_freq.plot([], [])
        [freq, n] = frequency_transform(self.fs, self.audio_file)
        self.lnb.set_xdata(freq)
        self.lnb.set_ydata(20*np.log10(abs(n)))
        self.plot_freq.set_xlim(freq[0],freq[len(freq)-1])
        self.plot_freq.set_ylim(-80, 80)
        self.plot_freq.set_xlabel('Frequency (Hz)')
        self.plot_freq.set_ylabel('Amplitude (dB)')
        self.fig2.canvas.draw()
        self.canvasB.draw()
        self.canvasB.get_tk_widget().grid(column=0, row=4, columnspan=3)
        
    # Command for opening a wave file
    def open_wav(self):
        file = filedialog.askopenfilename(filetypes = (("Audio files","*.wav"),("all files","*.*")))
        [self.fs, self.audio_file] = import_audio_wav(file)
        self.original_audio = self.audio_file
        self.stop = 0
        self.draw_fig() 
        self.draw_freq()
        self.p = pyaudio.PyAudio()
        self.player = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.RATE, output=True, frames_per_buffer=self.CHUNK)

    # Animation function for animated frequency domain for audio visualizer with dynamic equalization
    def updateGraphsBDyn(self, i):
        [freq, n] = frequency_transform(self.fs, self.audio_file[i*self.CHUNK:(i+1)*self.CHUNK])
        n2 = abs(n)
        filt = []   
        new_sum = np.sum(n2)
        for i in range(0, int(len(n2)), int((len(n2)+1)/1025)):
            tweak = 1 - (np.sum(n2[i:100*( i+int((len(n2)+1)/1025))])/new_sum)/2
            filt = np.concatenate([filt, np.ones(int((len(n2) + 1)/1025)) * tweak])        
        out_freq = np.multiply(n, filt)
        timedomain = time_transform(out_freq)
        self.player.write(timedomain.astype(np.float32).tostring(), self.CHUNK)
        self.lnb.set_xdata(freq)
        self.lnb.set_ydata(20*np.log10(abs(out_freq)))
        if (i == int(len(self.audio_file)/self.CHUNK)-1):
            self.draw_freq()
        return self.lnb,

    # Animation function for animated frequency domain for audio visualizer
    def updateGraphsB(self, i):
        #if (i == 0):
        #    self.canvasB.draw()
        self.player.write(self.audio_file[i*self.CHUNK:(i+1)*self.CHUNK].astype(np.float32).tostring(), self.CHUNK)
        [freq, n] = frequency_transform(self.fs, self.audio_file[i*self.CHUNK:(i+1)*self.CHUNK])
        self.lnb.set_xdata(freq)
        self.lnb.set_ydata(20*np.log10(abs(n)))
        return self.lnb,
    
    # Function to play audio
    def play_wav(self):
        self.aniB = animation.FuncAnimation(self.fig2, self.updateGraphsB, frames=int(len(self.audio_file)/self.CHUNK), interval=self.CHUNK/self.fs, blit=True, repeat=False)
        self.draw_freq()
        
    # Function to stop all audio
    def stop_wav(self):
        self.aniB.event_source.stop()
        self.draw_freq()
        
    # Function to close open file
    def close_wav(self):
        self.player.stop_stream()
        self.player.close()
        
    # Function to reset audio to audio originally opened/recorded
    def reset(self):
        self.audio_file = self.original_audio
        self.draw_freq()
        self.draw_fig()
        
    # Function to record audio
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
        self.p = pyaudio.PyAudio()
        self.player = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.RATE, output=True, frames_per_buffer=self.CHUNK)

    # Function to adjust speed of audio
    def adjust_speed(self):
        self.audio_file = resample(self.audio_file, int((1-self.w.get())*len(self.audio_file)))
        self.draw_fig()
        self.draw_freq()
    
    # Function to adjust pitch of audio by frequency shifting
    def adjust_pitch(self):
        self.audio_file = frequency_shift(self.fs, self.audio_file, int(self.w2.get() * self.fs/32))
        self.draw_fig()
        self.draw_freq()
         
    # Function for applying low pass filter
    def low_pass(self):
        fil = generate_lowpass_filter(self.fs,self.audio_file,int(self.lpfilt_cut.get()))
        self.audio_file = apply_filter(self.fs, self.audio_file, fil)
        self.draw_fig()
        self.draw_freq()
        
    # Function for applying high pass filter
    def high_pass(self):
        fil = generate_highpass_filter(self.fs,self.audio_file,int(self.hpfilt_cut.get()))
        self.audio_file = apply_filter(self.fs, self.audio_file, fil)
        self.draw_fig()
        self.draw_freq()
    
    # Function for applying band pass filter
    def band_pass(self):
        fil = generate_bandpass_filter(self.fs,self.audio_file,int(self.bpfilt_cutlow.get()), int(self.bpfilt_cuthigh.get()))
        self.audio_file = apply_filter(self.fs, self.audio_file, fil)
        self.draw_fig()
        self.draw_freq()
        
    # Function to get acoustic response of room and apply inverse filter
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
        
    # Function to apply equalizer to audio
    def eq(self):
        N = len(self.audio_file)//2 + 1
        band1 = round(40/self.fs * N * 2)
        band2 = round(80/self.fs * N * 2)
        band3 = round(160/self.fs * N * 2)
        band4 = round(300/self.fs * N * 2)
        band5 = round(600/self.fs * N * 2)
        band6 = round(1200/self.fs * N * 2)
        band7 = round(2400/self.fs * N * 2)
        band8 = round(5000/self.fs * N * 2)
        band9 = round(10000/self.fs * N * 2)
        band10 = round(N)
    
        fil = np.concatenate([np.ones(band1)*float(self.eq1.get()), np.ones(band2-band1)*float(self.eq2.get()), np.ones(band3-band2)*float(self.eq3.get()),np.ones(band4-band3)*float(self.eq4.get()),np.ones(band5-band4)*float(self.eq5.get()),np.ones(band6-band5)*float(self.eq6.get()),np.ones(band7-band6)*float(self.eq7.get()),np.ones(band8-band7)*float(self.eq8.get()),np.ones(band9-band8)*float(self.eq9.get()),np.ones(band10-band9)*float(self.eq10.get())])
        print(fil.shape)
        self.audio_file = apply_filter(self.fs, self.original_audio, fil)
        self.draw_fig()
        self.draw_freq()
        return
          
    # Function to convolve audio with sample impulse response (church)
    def sim_church(self):
        self.audio_file = apply_impulse_response(self.audio_file, self.church)
        self.draw_fig()
        self.draw_freq()
    
    # Function to import acoustic impulse response and convolve with audio
    def import_sim(self):   
        file = filedialog.askopenfilename(filetypes = (("Audio files","*.wav"),("all files","*.*")))
        [fs, response] = import_audio_wav(file)
        self.audio_file = apply_impulse_response(self.audio_file, response)
        self.draw_fig()
        self.draw_freq()
        
    # Function to play audio with dynamic equalizer
    def dyn_eq(self):
        self.aniB = animation.FuncAnimation(self.fig2, self.updateGraphsBDyn, frames=int(len(self.audio_file)/self.CHUNK), interval=self.CHUNK/self.fs, blit=True, repeat=False)
        self.draw_freq()
    
    # Function to initialize program
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
        
        lbl_dyn_eq = Label(tab2, text = 'Dynamic Equalizer')
        lbl_dyn_eq.grid(column=0, row = 1)
        dyn_eq_btn = Button(tab2, text = 'Play', command=self.dyn_eq)
        dyn_eq_btn.grid(column = 1, row = 1)

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
        
        lbl_man_eq = Label(tab2, text='Equalizer')
        lbl_man_eq.grid(column=0,row=2)
        
        lbl_eq1 = Label(tab2, text="0-40 Hz")
        lbl_eq1.grid(column=1,row=2)
        self.eq1 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq1.grid(column=2, row = 2)
        
        lbl_eq2 = Label(tab2, text="40-80 Hz")
        lbl_eq2.grid(column=3,row=2)
        self.eq2 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq2.grid(column=4, row = 2)
        
        lbl_eq3 = Label(tab2, text="80-160 Hz")
        lbl_eq3.grid(column=5,row=2)
        self.eq3 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq3.grid(column=6, row = 2)
        
        lbl_eq4 = Label(tab2, text="160-300 Hz")
        lbl_eq4.grid(column=1,row=3)
        self.eq4 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq4.grid(column=2, row = 3)
        
        lbl_eq5 = Label(tab2, text="300-600 Hz")
        lbl_eq5.grid(column=3,row=3)        
        self.eq5 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq5.grid(column=4, row = 3)
        
        lbl_eq6 = Label(tab2, text="600-1.2k Hz")
        lbl_eq6.grid(column=5,row=3)         
        self.eq6 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq6.grid(column=6, row = 3)
        
        lbl_eq7 = Label(tab2, text="1.2k-2.4k Hz")
        lbl_eq7.grid(column=1,row=4) 
        self.eq7 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq7.grid(column=2, row = 4)
        
        lbl_eq8 = Label(tab2, text="2.4k-5k Hz")
        lbl_eq8.grid(column=3,row=4) 
        self.eq8 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq8.grid(column=4, row = 4)
        
        lbl_eq9 = Label(tab2, text="5k-10k Hz")
        lbl_eq9.grid(column=5,row=4) 
        self.eq9 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq9.grid(column=6, row = 4)
        
        lbl_eq10 = Label(tab2, text="10k-20k Hz")
        lbl_eq10.grid(column=1,row=5) 
        self.eq10 = Scale(tab2, from_=0, to=1, resolution=0.01, orient=  HORIZONTAL)
        self.eq10.grid(column=2, row = 5)
        
        eq_btn = Button(tab2, text="Apply", command=self.eq)
        eq_btn.grid(column=3, row=5)
        
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
        
# Function to create and run program
if __name__ == "__main__":        
    root = tk.Tk()
    root.state('zoomed')
    MainApplication(root)
    root.mainloop()