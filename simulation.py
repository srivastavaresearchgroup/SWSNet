import numpy as np
import cmath
from math import pi, sqrt, exp, sin, cos
from scipy.fft import fft, ifft
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import matplotlib as mpl

### setting some parameters for plotting figures
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize']=20
mpl.rcParams['ytick.labelsize']=20
mpl.rcParams['axes.labelsize']=22
mpl.rcParams['axes.titlesize']=24
mpl.rcParams['figure.figsize']=[5,4]

dt=1/50 # sample interval
tmin=0 
tmax=40 # time window length 20s

# define time axis
Nt=int((tmax-tmin)/dt)+1
t=np.linspace(tmin,tmax,Nt)

def ewave_T2(t,T,t0):
    '''   
    normalized derivative of exp-function
    if T is used: exponential term such that distance between extrema = T/2
    here: use effective period T_eff=T/sqrt(2);
    such that maximum spectral amplitude occurs at T
    '''
    T_eff=0.1*2*pi*T
    ewnorm=4/(T_eff*sqrt(exp(1)))
    ew=-16/T_eff**2*np.multiply((t-t0),np.exp(-8/T_eff**2*(t-t0)**2))/ewnorm
    return ew

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def generate(alpha, deltat, period=10, t1=20, nlevel = 0.3, corner_freq = 1.0, 
             tsign=-1, make_plots = False):
    '''
    This function is used to generate synthetic waveforms for given splitting parameters.
    alpha: difference between back-azimuth and fast-axis orientation, given in degrees
    deltat: delay time in seconds
    period: dominant period of the waveform
    nlevel: noise-level to be added in the data
    corner_freq: corner frequency for low-pass filter
    tsign: sign convention for transverse component
    make_plots: when set to true it makes plots for every step of the simulation process
    '''
    
    # get initial waveform 
    up0=ewave_T2(t,period,t1)
    
    if make_plots:
        plt.plot(t, up0)
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Initial waveform')
        
    ### alpha in radians
    alpha=alpha*pi/180

    uf0=cos(alpha)*up0
    us0=-sin(alpha)*up0
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf0, label = 'f-component')
        plt.plot(t, us0, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Initial waveform')
        plt.legend()

    ## adding random noise    
    noisef = np.random.rand(len(uf0))-0.5
    noises = np.random.rand(len(us0))-0.5
    
    noisef = butter_lowpass_filter(noisef, corner_freq, 1/dt, 2)
    noises = butter_lowpass_filter(noises, corner_freq, 1/dt, 2)
    
    noisef=nlevel*2*max(abs(uf0))*noisef/max(abs(noisef))
    noises=nlevel*2*max(abs(us0))*noises/max(abs(noises))
    
    uf=uf0+noisef
    us=us0+noises
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf, label = 'f-component')
        plt.plot(t, us, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Initial waveform contaminated by noise')
        plt.legend()  
        
    # apply time shift (second part of splitting)
    uf1=np.zeros(Nt)
    us1=np.zeros(Nt)
    # be sure that deltat/dt is an even integer
    N_shift=int((deltat/dt)/2)
    # fast wave, advance by deltat/2
    uf1[:Nt-N_shift]=uf[N_shift:Nt]
    uf1[Nt-N_shift:Nt]=uf[:N_shift]
    # slow wave, delay by deltat/2
    us1[N_shift:Nt]=us[:Nt-N_shift]
    us1[:N_shift]=us[Nt-N_shift:Nt]
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf1, label = 'f-component')
        plt.plot(t, us1, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform after splitting')
        plt.legend()
        
    
    # return to r and t coordinates, after splitting
    ur1=(cos(alpha)*uf1-sin(alpha)*us1)
    ut1=tsign * (sin(alpha)*uf1+cos(alpha)*us1)

    # obtain subarray for plotting, cut beginning and end of trace
    icut=150
    ur1_sub=ur1[1+icut:Nt-icut]
    ut1_sub=ut1[1+icut:Nt-icut]
    
    if make_plots:
        plt.figure()
        plt.plot(np.linspace(0,34,1700), ur1_sub, label = 'r-component')
        plt.plot(np.linspace(0,34,1700), ut1_sub, label = 't-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform in RT components')
        plt.legend()
        
        
    return (ur1_sub, ut1_sub)

def generate_deconvolved(alpha, deltat, period=10, t1=20, nlevel = 0.3, corner_freq = 1.0, tsign=-1, wlevel=0.0005, make_plots = False):

    ## this function applies deconvolution to the simulated data
    
    ur, ut = generate(alpha, deltat, period, t1, nlevel, corner_freq, tsign, make_plots = make_plots)
    
    from deconvolution import deconvolve
    
    return deconvolve(ur, ut, period, t1, wlevel, corner_freq, make_plots = make_plots)