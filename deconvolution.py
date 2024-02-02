import numpy as np
import cmath
from math import pi, sqrt, exp, sin
from scipy.fft import fft, ifft
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import matplotlib as mpl

### setting some parameters for plotting figures
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['xtick.labelsize']=24
mpl.rcParams['ytick.labelsize']=24
mpl.rcParams['axes.labelsize']=26
mpl.rcParams['axes.titlesize']=28
mpl.rcParams['figure.figsize']=[6,4]


Nt=2000  ### number of samples
tmin=0
tmax=40 # time window length 20s

# define time axis
t=np.linspace(tmin,tmax,Nt + 1)
t = t[:-1]
dt=t[1]-t[0];# set time shift for delta-function

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

def deconvolve(ur, ut, period=10, t1=20, wlevel=0.0005, corner_freq = 1.0, make_plots = False):
    '''
    This is where everything comes together to deconvolve the waveform.
    ur: radial component
    ut: transverse component
    period: the dominant period of the data
    t1: time at which the data is centered
    wlevel: water level to avoid division by zero
    corner_freq: corner frequency for butterworth filter
    make_plots: when set to true it makes plots for every step of the deconvolution process
    '''

    if make_plots:
        plt.plot(ur, label = 'radial')
        plt.plot(ut, label = 'transverse')

        plt.xlabel('sample')
        plt.ylabel('amplitude')
#         plt.axhline(0, linestyle = '--', color = 'k')
        plt.title('Input data after\nresampling and mean-removal')
        plt.legend(fontsize = 18)
        plt.show()
        
    # apply Hanning window
    window=np.hanning(len(ur))
    ur=np.multiply(ur,window)
    ut=np.multiply(ut,window)
    
    if make_plots:
        plt.plot(ur, label = 'radial')
        plt.plot(ut, label = 'transverse')

        plt.xlabel('sample')
        plt.ylabel('amplitude')
        plt.title('After applying Hanning Window')
        plt.show()
       
    ## zero padding the data
    ur_pad = np.zeros((Nt,))
    ut_pad = np.zeros((Nt,))

    ur_pad[Nt // 2 - len(ur)//2:Nt // 2 + len(ur) // 2] = ur.copy()
    ut_pad[Nt // 2 - len(ut)//2:Nt // 2 + len(ut) // 2] = ut.copy()
    
    if make_plots:
        plt.plot(t,ur_pad, label = 'radial')
        plt.plot(t,ut_pad, label = 'transverse')

        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Data with zero padding')
        plt.show()
        
        
    #applying lowpass filter
    ur = butter_lowpass_filter(ur_pad, corner_freq, 1/dt, 2)
    ut = butter_lowpass_filter(ut_pad, corner_freq, 1/dt, 2)
        
    up0=ewave_T2(t,period,t1)
    
    if make_plots:
        plt.plot(t,ur, label = 'radial')
        plt.plot(t,ut, label = 'transverse')
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Applying Butterworth filter');
        plt.show()
        
    # FK of initial waveform
    up0f=fft(up0)  
    
    urf = fft(ur)
    utf = fft(ut)
    
    fn = [n / (Nt * dt) for n in range(Nt)]

    if make_plots:
        plt.plot(fn,2*abs(urf), label = 'radial')
        plt.plot(fn,2*abs(utf), label = 'transverse')

        plt.xlabel('frequency (Hz)')
        plt.ylabel('amplitude')
        plt.xlim(0,25)
        plt.title('Fourier coefficients of input data')
        plt.show()

    urf_decon_delta = np.empty(Nt, dtype=complex)
    utf_decon_delta = np.empty(Nt, dtype=complex)
    urf_decon = np.empty(Nt, dtype=complex)
    utf_decon = np.empty(Nt, dtype=complex)


    # apply deconvolution
    for n in range(int(Nt/2)+1):
        # use radial component for deconvolution
        xq=urf[n]
        if (abs(xq) < wlevel):
            xq=wlevel
        # apply time shift
        tshift=cmath.exp(-2j*pi*fn[n]*t1)
        # get delta-function
        urf_decon_delta[n]=(urf[n]/xq)*tshift
        utf_decon_delta[n]=(utf[n]/xq)*tshift
        
        # apply/reconvolve with clean waveform, no tshift (contained in original waveform)
        urf_decon[n]=urf[n]/xq*up0f[n]
        # deconvolution and reconvolution of transverse component
        utf_decon[n]=(utf[n]/xq)*up0f[n]

    # fill up frequencies
    ix=1
    for n in range(int(Nt/2)+1,Nt):
        
        idx=int(Nt/2)-ix
        urf_decon_delta[n]=np.conj(urf_decon_delta[idx]);
        urf_decon[n]=np.conj(urf_decon[idx]);

        utf_decon_delta[n]=np.conj(utf_decon_delta[idx]);
        utf_decon[n]=np.conj(utf_decon[idx])
        ix+=1

    # back into time domain
    ur_decon_delta=ifft(urf_decon_delta);
    ur_decon=ifft(urf_decon);
    ut_decon_delta=ifft(utf_decon_delta);
    ut_decon=ifft(utf_decon);

    if make_plots:
        plt.plot(fn, 2*abs(urf_decon_delta), fn, 2*abs(utf_decon_delta))
        plt.title('Fourier coefficients after deconvolution')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('amplitude')
        plt.xlim(0,25)
        plt.show()

        plt.plot(t,np.real(ur_decon_delta),t,np.real(ut_decon_delta))
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Radial and Transverse\nComponents after deconvolution')
        plt.show()
        
        plt.plot(t,np.real(ur_decon),t,np.real(ut_decon))
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Radial and Transverse Components\nafter convolution with clean waveform')
        plt.show()
    
    ur_decon = np.real(ur_decon)
    ut_decon = np.real(ut_decon)
    
    ## applying Hanning window
    window=np.hanning(len(ur_decon))
    ur_decon=np.multiply(ur_decon,window)
    ut_decon=np.multiply(ut_decon,window)

    if make_plots:
        plt.plot(t,ur_decon,t,ut_decon)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Applying Hanning window')
        plt.show()
        
    ## cropping the data 
    ur_cut = ur_decon[750:1250].copy()
    ut_cut = ut_decon[750:1250].copy()
    
    ##windowing
    window=np.hanning(len(ur_cut))
    ur_cut=np.multiply(ur_cut,window)
    ut_cut=np.multiply(ut_cut,window)
    
    t2=np.linspace(0,10,501)
    t2 = t2[:-1]

    
    if make_plots:
        plt.plot(t2,ur_cut,t2,ut_cut)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Cropping and applying Hanning window')
        plt.show()
    
    ## normalize
    maxima = np.max(ur_cut)
    
    ur_cut = ur_cut / maxima
    ut_cut = ut_cut / maxima
    
    
    if make_plots:
        plt.plot(t2,ur_cut,t2,ut_cut)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('Radial and Transverse\ncomponents after normalization')
        plt.show()
    
    return (ur_cut, ut_cut)