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

def generate(alpha1, deltat1, alpha2, deltat2, 
             period=10, t1=20, nlevel = 0.3, corner_freq = 1.0, 
             tsign=-1, make_plots = False):
    '''
    This function is used to generate synthetic waveforms for given splitting parameters, in a 2-layer anisotropy setting
    alpha1: difference between back-azimuth and fast-axis orientation for first anisotropic layer, given in degrees
    deltat1: delay time in seconds for first anisotropic layer
    alpha2: difference between back-azimuth and fast-axis orientation for second anisotropic layer, given in degrees
    deltat2: delay time in seconds for second anisotropic layer
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
        
    ## alphas in radians
    alpha1=alpha1*pi/180
    alpha2=alpha2*pi/180
    
    ## first layer
    uf10=cos(alpha1)*up0
    us10=-sin(alpha1)*up0
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf10, label = 'f-component')
        plt.plot(t, us10, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform before first splitting')
        plt.legend()
        
    noisef = np.random.rand(len(uf10))-0.5
    noises = np.random.rand(len(us10))-0.5
    
    noisef = butter_lowpass_filter(noisef, corner_freq, 1/dt, 2)
    noises = butter_lowpass_filter(noises, corner_freq, 1/dt, 2)
    
    noisef=nlevel*2*max(abs(uf10))*noisef/max(abs(noisef));
    noises=nlevel*2*max(abs(us10))*noises/max(abs(noises));
    
    uf10=uf10+noisef
    us10=us10+noises
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf10, label = 'f-component')
        plt.plot(t, us10, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Initial waveform contaminated by noise')
        plt.legend()  
        
    # apply time shift (second part of splitting)
    uf11=np.zeros(Nt)
    us11=np.zeros(Nt)
    # be sure that deltat/dt is an even integer
    N_shift=int((deltat1/dt)/2)
    # fast wave, advance by deltat/2
    uf11[:Nt-N_shift]=uf10[N_shift:Nt]
    uf11[Nt-N_shift:Nt]=uf10[:N_shift]
    # slow wave, delay by deltat/2
    us11[N_shift:Nt]=us10[:Nt-N_shift]
    us11[:N_shift]=us10[Nt-N_shift:Nt]
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf11, label = 'f-component')
        plt.plot(t, us11, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform after 1st split')
        plt.legend()
        
    
    # return to r and t coordinates, after splitting
    ur1=(cos(alpha1)*uf11-sin(alpha1)*us11)
    ut1=sin(alpha1)*uf11+cos(alpha1)*us11
    
    if make_plots:
        plt.figure()
        plt.plot(t, ur1, label = 'r-component')
        plt.plot(t, ut1, label = 't-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform after 1st split')
        plt.legend()

    ## second layer
    uf20=cos(alpha2)*ur1 + sin(alpha2)*ut1
    us20=-sin(alpha2)*ur1 + cos(alpha2)*ut1

    # apply time shift (second part of splitting)
    uf21=np.zeros(Nt)
    us21=np.zeros(Nt)
    # be sure that deltat/dt is an even integer
    N_shift=int((deltat2/dt)/2)
    # fast wave, advance by deltat/2
    uf21[:Nt-N_shift]=uf20[N_shift:Nt]
    uf21[Nt-N_shift:Nt]=uf20[:N_shift]
    # slow wave, delay by deltat/2
    us21[N_shift:Nt]=us20[:Nt-N_shift]
    us21[:N_shift]=us20[Nt-N_shift:Nt]
    
    if make_plots:
        plt.figure()
        plt.plot(t, uf21, label = 'f-component')
        plt.plot(t, us21, label = 's-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform after 2nd split')
        plt.legend()
        
    
    # return to r and t coordinates, after splitting
    ur2=(cos(alpha2)*uf21-sin(alpha2)*us21)
    ut2=tsign * (sin(alpha2)*uf21+cos(alpha2)*us21)
    
    if make_plots:
        plt.figure()
        plt.plot(t, ur2, label = 'r-component')
        plt.plot(t, ut2, label = 't-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform after 2nd split')
        plt.legend()


    # obtain subarray for plotting, cut beginning and end of trace
    icut=150
    ur2_sub=ur2[1+icut:Nt-icut]
    ut2_sub=ut2[1+icut:Nt-icut]
    
    if make_plots:
        plt.figure()
        plt.plot(np.linspace(0,34,1700), ur2_sub, label = 'r-component')
        plt.plot(np.linspace(0,34,1700), ut2_sub, label = 't-component')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalised amplitude')
        plt.title('Waveform in RT components')
        plt.legend()
        
        
    return (ur2_sub, ut2_sub)

def generate_deconvolved(alpha1, deltat1, alpha2, deltat2, period=10, t1=20, nlevel = 0.3, corner_freq = 1.0, tsign=-1, wlevel=0.0005, make_plots = False):
    
    ## this function applies deconvolution to the simulated data
    
    ur, ut = generate(alpha1, deltat1, alpha2, deltat2, period, t1, nlevel, corner_freq, tsign, make_plots = make_plots)
    
    from deconvolution import deconvolve
    
    return deconvolve(ur, ut, period, t1, wlevel, corner_freq, make_plots = make_plots)