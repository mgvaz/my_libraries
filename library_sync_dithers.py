

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


from ipfnpytools.spline import constrained_spline
from ipfnpytools.has_even_sampling import has_even_sampling
from ipfnpytools.plot import plot_signals, plots, default_specgram_image_show_kw
from ipfnpytools.timing import timing

from ipfnpytools.spec import power_2_ceil

import os

from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy.signal import spectrogram, detrend, find_peaks
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from tqdm import tqdm

import pandas as pd
import seaborn as sns

from ipfnpytools.spline import constrained_spline



def crop_data(time, data, ti, tf):
    """Crop (t, signal) data by initial and final time

    Args:
        time (array): time array
        data (array): data arrat
        ti (_type_): initial time
        tf (_type_): final time

    Returns:
        t, data: cropped time and data
    """
    return time[(time>=ti) & (time<=tf)], data[(time>=ti) & (time<=tf)]

def make_dir(dir_name):
    """Create directory if it does not exist

    Args:
        dir_name (str): directory name

    Returns:
        str: directory name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    directory=dir_name
    return directory

def find_nearest(array, value, to_return='indice'):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if to_return=='indice':
        return idx
    elif to_return=='value':
        return array[idx]
    else:
        print('Not a valid option')


def correct_sampling(t, i, q):
    
    dt = (t - np.roll(t, 1))[1:]

    j=0

    #print(len(np.argwhere(dt > dt[0]*1.5)))
    
    for k in tqdm(np.argwhere(dt > dt[0]*1.5)):
        
        k=int(k[0])
        to_add = int((np.round(dt[k]/dt[0])-1))
        assert to_add==200, 'Number of points added was not 200'

        t=np.insert(t, int(k+j+1), np.arange(t[k+j] + dt[0], t[k+j+1], dt[0]))
        i=np.insert(i, int(k+j+1), np.zeros(to_add))
        q=np.insert(q, int(k+j+1), np.zeros(to_add))
        
        j=int(j+to_add)

    assert has_even_sampling(t, atol=1.e-8), 't does not has even sampling'
    assert len(t)==len(i), 'arrays do not have the same length'
    assert len(t)==len(q), 'arrays do not have the same length'

    print('Sampling Corrected')

    return t, i, q


def correct_sampling_2(t, i, q, dt=1e-7):

    assert t.shape==i.shape, 't and i should have the same shape'
    assert t.shape==q.shape, 't and q should have the same shape'

    t=np.round(t, -int(np.log10(dt)))
    assert np.all(np.diff(t)!=0), 't has repeated values'

    t_corrected = t[0] + np.arange(0, np.round((t[-1]-t[0])/dt)+1, dtype=np.int64) * dt
    t_corrected = np.round(t_corrected, -int(np.log10(dt)))

    _,_,mask=np.intersect1d(np.round(t , -int(np.log10(dt))), t_corrected, assume_unique=True, return_indices=True)

    i_corrected = np.zeros_like(t_corrected)
    q_corrected = np.zeros_like(t_corrected)

    i_corrected[mask] = i
    q_corrected[mask] = q

    assert has_even_sampling(t_corrected, atol=1.e-8), 't does not has even sampling'
    assert len(t_corrected)==len(i_corrected), 'arrays do not have the same length'
    assert len(t_corrected)==len(q_corrected), 'arrays do not have the same length'

    print('Sampling Corrected')

    return t_corrected, i_corrected, q_corrected

def correct_sampling_3(t, i, q, dt=1e-7):

    assert t.shape==i.shape, 't and i should have the same shape'
    assert t.shape==q.shape, 't and q should have the same shape'

    t=np.round(t, -int(np.log10(dt)))
    assert np.all(np.diff(t)!=0), 't has repeated values'

    t_corrected = t[0] + np.arange(0, np.round((t[-1]-t[0])/dt)+1, dtype=np.int64) * dt
    t_corrected = np.round(t_corrected, -int(np.log10(dt)))

    i_corrected=interp1d(t, i, kind='previous')(t_corrected)
    q_corrected=interp1d(t, q, kind='previous')(t_corrected)

 
    assert has_even_sampling(t_corrected, atol=1.e-8), 't does not has even sampling'
    assert len(t_corrected)==len(i_corrected), 'arrays do not have the same length'
    assert len(t_corrected)==len(q_corrected), 'arrays do not have the same length'

    print('Sampling Corrected')

    return t_corrected, i_corrected, q_corrected


def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def jac_gauss(x, a, x0, sigma):
    return np.array([np.exp(-(x-x0)**2/(2*sigma**2)), (x-x0)/sigma**2 * a*np.exp(-(x-x0)**2/(2*sigma**2)), (x-x0)**2 / (sigma**3) * a*np.exp(-(x-x0)**2/(2*sigma**2))]).T

def my_fft(t, i, q, t0, nfft,
            plot=True, plot_kw={}, return_plot=False,
            fmin=-400, fmax=400,
            remove_detrend=True, remove_detrend_args=dict(type='constant'), 
            **fft_args):
    
    assert has_even_sampling(t), 't does not have even sampling'

    fs = 1/np.mean(np.diff(t))

    start = find_nearest(t, t0, to_return='indice') 
    indices = np.arange(start, start+nfft)
    t=t[indices]
    i=i[indices]
    q=q[indices]

    if remove_detrend:
        i=detrend(i, **remove_detrend_args)
        q=detrend(q, **remove_detrend_args)
    
    fft_data=fft(i+1j*q, norm='forward', **fft_args)
    fft_freq=fftfreq(len(fft_data), 1/fs)
    fft_data=fftshift(fft_data)
    fft_freq=fftshift(fft_freq)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fft_freq/1000, np.abs(fft_data)**2, c='k', label='Fourier', **plot_kw)
        ax.set_xlim(left=fmin, right=fmax)
        ax.set_xlabel('f (kHz)')
        ax.set_ylabel('S(f)')
        ax.grid()
        if t is not None:
            ax.set_title('t = '+str(np.round(t[int(nfft/2)],3))+ '\n'+'delta_t = ' + str(np.round((t[-1]-t[0])*1000, 1)) + ' ms')
        else:
            ax.set_title('delta_t = ' + str(np.round(()*1000, 1)))

        fig.legend()
        if return_plot:
            return fft_freq, fft_data, fig, ax
    
    return fft_freq, fft_data

def fit_fft(freq, S_f, freq_range=None, 
            fun=gauss, jac=jac_gauss, bounds=([0, -400e3, 0], [np.inf, 400e3, np.inf]), p0='auto',
            return_fluctuations=False, return_popt=False,
            **curve_fit_args):

    if freq_range:
        indices_to_eval_on=((freq<=(-freq_range[0]*1000)) & (freq>=(-freq_range[1]*1000))) | ((freq>=(freq_range[0]*1000)) & (freq<=(freq_range[1]*1000)))
        S_f = S_f[indices_to_eval_on]
        freq = freq[indices_to_eval_on]
        
    if p0=='auto':
        CM = np.sum(freq * S_f)/np.sum(S_f)
        #sigma = np.sqrt(np.sum((freq-CM)**2 * S_f) / np.sum(S_f))
        sigma = np.sqrt(np.sum((freq-CM)**2 * S_f) / np.sum(S_f))
        dfreq = freq[1]-freq[0]
        amp = np.sum(S_f * dfreq)/(sigma*np.sqrt(2*np.pi))
        p0 = (amp, CM, sigma)

    
    
    popt, pcov, info, mesg, ier = curve_fit(fun, freq, S_f, p0=p0, full_output=True, jac=jac, bounds=bounds, **curve_fit_args)

    if return_popt:
        to_return = (popt,)
    else:
        to_return = (popt[1],)

    if return_fluctuations:
        to_return = to_return + (np.sum(fun(freq, popt[0], popt[1], popt[2])) / len(S_f),)
    
    return to_return

    
def center_of_mass(freq, S_f, freq_range=None, 
                return_fluctuations=False):

    if freq_range:
        assert freq_range[0]<freq_range[1], 'freq_range[0] should be smaller than freq_range[1]'

    if len(S_f.shape)==1:
        if freq_range:
            indices_to_eval_on=((freq<=(-freq_range[0]*1000)) & (freq>=(-freq_range[1]*1000))) | ((freq>=(freq_range[0]*1000)) & (freq<=(freq_range[1]*1000)))
            S_f = S_f[indices_to_eval_on]
            freq = freq[indices_to_eval_on] 

        CM = np.sum(freq * S_f)/np.sum(S_f)

        to_return = (CM,)

        if return_fluctuations:
            if freq_range:
                to_return = to_return + (np.mean(S_f) * 2 * (freq_range[1]-freq_range[0]),)
            else:
                to_return = to_return + (np.mean(S_f) * (np.max(freq)-np.min(freq)),)

        return to_return
    

    if len(S_f.shape)==2:
        if freq_range:
            indices_to_eval_on=((freq<=(-freq_range[0]*1000)) & (freq>=(-freq_range[1]*1000))) | ((freq>=(freq_range[0]*1000)) & (freq<=(freq_range[1]*1000)))
            S_f = S_f[indices_to_eval_on,:]
            freq = freq[indices_to_eval_on] 

        CM = np.dot(S_f.T, freq)/np.sum(S_f, axis=0)
        
        to_return = (CM,)

        if return_fluctuations:
            if freq_range:
                to_return = to_return + (np.mean(S_f, axis=0) * 2 * (freq_range[1]-freq_range[0]),)
            else:
                to_return = to_return + (np.mean(S_f, axis=0) * (np.max(freq)-np.min(freq)),)
        
        return to_return  

def index_at(t, t_look, fs):
    if fs is None:
        fs=1/(t[1]-t[0])
    return int((t_look-t[0])*fs)

def get_indices(fs, t, t_begin, t_end=None, delta_t=None, n_fft=2048, nearest_2_power=True):
    
    assert len([x for x in [t_end, delta_t, n_fft] if x is not None])==1, 'one and only one of [t_end, delta_t, n_fft] should be not none'

    if n_fft:
        return np.arange(int((t_begin-t[0])*fs), int((t_begin-t[0])*fs) + n_fft)

    if delta_t:
        if nearest_2_power:
            dx = power_2_ceil(int(delta_t*1.e-3*fs))
        else:
            dx = int(delta_t*1.e-3*fs)
        return np.arange(int((t_begin-t[0])*fs), int((t_begin-t[0])*fs) + dx)

    if t_end:
        if nearest_2_power:
            dx = power_2_ceil(int((t_end-t_begin)*fs))
            return np.arange(int((t_begin-t[0])*fs), int((t_begin-t[0])*fs) + dx)
        else:
            return np.arange(int((t_begin-t[0])*fs), int((t_end-t[0])*fs))

def normalize(x, ti=None, tf=None, t=None):
    if (ti is not None) and (tf is not None) and (t is not None):
        x=x[(t>=ti) & (t<=tf)]
        t=t[(t>=ti) & (t<=tf)]
        x=x-np.min(x)
        x=x/(np.max(x)-np.min(x))
        return t, x
    else:
        x=x-np.min(x)
        x=x/(np.max(x)-np.min(x))
        return x

def find_ditter(data, fs, fmin, fmax):

    f, t, Sxx = spectrogram(x = data, fs=fs, nperseg=264)
    print('delta t : %.1f ms'%((t[1]-t[0])*1000))
    print('delta f : %.5f Hz'%((f[1]-f[0])))

    mag_200 = np.mean(Sxx[(f>0) & (f<200), :], axis=0)

def get_peaks_in_interval(t, x, peaks, ti, tf):
    assert len(t)==len(x), 't and x shoud be the same length' 
    
    return t[peaks][(t[peaks] >= ti) & (t[peaks] <= tf)], x[peaks][(t[peaks] >= ti) & (t[peaks] <= tf)]
    
def sort_arrays(x, y):
    assert x.size==y.size, 'x and y must be the same size'
    x, y = np.array(x), np.array(y)
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    return x, y

def make_it_array(my_list):
    if len(my_list)==0:
        return np.array([])
    n_cols = len(max(my_list, key=len))
    n_rows = len(my_list)

    M = np.ones((n_rows, n_cols)) * np.NaN

    for i in range(n_rows):
        M[i,0:len(my_list[i])]=np.array(my_list[i])
    return M


def moving_average_non_unif(x, y, dx=None, dx_overlap=0, bins=None, n=None, x_begin=None, x_stop=None, return_counts=False):

    #print(dx is not None, n is not None, bins is not None)
    assert (dx is not None) or (n is not None) or (bins is not None), 'dx, n or bins must be provided'

    if x_stop is None:
        x_stop=np.nanmax(x)
    if x_begin is None:
        x_begin=np.nanmin(x)
    
    mask=(x>=x_begin) & (x<=x_stop)
    x=x[mask]
    y=y[mask]

    if bins is not None:
        stats, bins, _ = binned_statistic(x, [x,y], bins=bins)
        x_ret=stats[0]
        y_ret=stats[1]

    if dx:
        j=0
        xi=x_begin
        x_ret=np.zeros( int((x_stop-x_begin)/(dx-dx_overlap)) + 1 )
        y_ret=np.zeros_like(x_ret)

        while xi < x_stop:
            mask=(x>=xi) & (x<(xi+dx))
            if mask.sum()>2:
                x_ret[j] = np.mean(x[mask])
                y_ret[j] = np.mean(y[mask])
            else:
                x_ret[j] = np.nan
                y_ret[j] = np.nan
            j=j+1
            
            xi=xi+dx-dx_overlap

    
    if n:
        j=0
        x, y = sort_arrays(x, y)

        x_ret=np.zeros( int(len(x)/n)+10 )
        y_ret=np.zeros_like(x_ret)
        k=0

        while (k+n) < len(x):

            x_ret[j] = np.mean(x[k:k+n])
            y_ret[j] = np.mean(y[k:k+n])

            j=j+1
            
            k=k+n
   
    return x_ret, y_ret


def annotate_plot(pulse, t_ditters, axes, y_pos_frac=0.7):
    x=(np.array(t_ditters[pulse]['ti'])+np.array(t_ditters[pulse]['tf']))/2
    
    dx = (np.array(t_ditters[pulse]['tf']) - np.array(t_ditters[pulse]['tf']))

    for ax in axes:
        
        for i in range(len(t_ditters[pulse]['ti'])):
            if i==0:
                y_pos = ax.get_ylim()[0] + 0.5*(y_pos_frac)*(ax.get_ylim()[1]-ax.get_ylim()[0])
            else:
                y_pos = ax.get_ylim()[0] + y_pos_frac*(ax.get_ylim()[1]-ax.get_ylim()[0])

            ax.text(s=str(i), x=x[i], y=y_pos, ha='center', va='bottom')
            ax.annotate(text='', xy=(t_ditters[pulse]['ti'][i], y_pos), xytext=(t_ditters[pulse]['tf'][i], y_pos),
                        arrowprops=dict(arrowstyle="<->"))

            ax.vlines(t_ditters[pulse]['ti'][i], ax.get_ylim()[0], ax.get_ylim()[1], linestyles='--', color='k')
            ax.vlines(t_ditters[pulse]['tf'][i], ax.get_ylim()[0], ax.get_ylim()[1], linestyles='--', color='k')

def get_ditters_indices(begin, end):
    ditters_list=[]

    cycle_begin=begin[0]
    cycle_end=end[0]
    (x,y) = (begin[0], begin[1])

    while (x,y) != (end[0], end[1]):
        ditters_list.append((x,y))

        y=y+1
        if y==14:
            y=0
            x=x+1

    return ditters_list

def sync_signals(t, t_stamps, t_shift=0, normalize=False):
    if type(t) is float:
        t = np.array([t])
        
    t_sync = np.full(t.shape, np.nan)
    if normalize:
        for i in range(len(t_stamps) - 1):
            mask = (t>=(t_stamps[i]-t_shift)) & (t<=(t_stamps[i+1]-t_shift))
            t_sync[mask] = (t[mask] - t_stamps[i] ) / (t_stamps[i+1] - t_stamps[i])
    else:
        for i in range(len(t_stamps) - 1):
            mask = (t>=(t_stamps[i]-t_shift)) & (t<=(t_stamps[i+1]-t_shift))
            t_sync[mask] = t[mask] - t_stamps[i] 

    return t_sync

def filter_sig(t_sig, sig, f_cut_off):
    assert has_even_sampling(t_sig), 't_sig does not have even sampling'
    
    fs = 1 / np.mean(np.diff(t_sig))

    fft_data=fft(sig, n=len(sig))
    fft_freq=fftfreq(len(fft_data), 1/fs)

    fft_data[fft_freq>f_cut_off]=0
    fft_data[fft_freq<-f_cut_off]=0
    smooth_sig = ifft(fft_data).real

    return smooth_sig


def peak_find_args(pulse, fs, sig):
    param_dict_tdao = {
                        '96292': dict(height=-1.05e16, prominence=0.4e15),
                        '96294': dict(height=-1.07e16, prominence=0.3e15),
                        '94120': dict(height=-2e16, prominence=0.3e16),
                        '105439': dict(height=-2e16, prominence=0.3e16),
                        '100759': dict(height=-3.18e15, prominence=0.2e15),
                        '100844': dict(height=-3.3e15, prominence=0.2e15),
                        '100841': dict(height=-3.3e15, prominence=0.2e15),
                        '99472': dict(height=-2.5e15, prominence=0.1e15)                        
    }
    param_dict_dtdai_pos = {
                            '96292': dict(distance=int(0.006*fs), height=0.62e19, prominence=0.4e19),
                            '94120': dict(distance=int(0.005*fs), height=2e19, prominence=0.4e19),
                            '96294': dict(distance=int(0.005*fs), height=0.4e19, prominence=0.4e19 ),
                            '105439': dict(distance=int(0.005*fs), height=2e19,),
                            '100759': dict(distance=int(0.005*fs), height=1e19,),
                            '100841': dict(distance=int(0.005*fs), height=1e19,),
                            '100844': dict( distance=int(0.005*fs), height=1e19,),
                            '99472': dict(distance=int(0.005*fs), height=0.5e19,)
                            
                            }
    param_dict_dtdai_neg = {
                            '96292': dict(height = 5e18, distance=int(5e-3 * fs)),
                            '94120': dict(height = 2e19, distance=int(5e-3 * fs)),
                            '96294': dict(height = 0.4e19, distance = int(5e-3 * fs) ),
                            '105439': dict(distance=int(0.006*fs), height=0.4e19),
                            '100759': dict(distance=int(0.002*fs), height=0., prominence=1e19),
                            '100841': dict(distance=int(0.006*fs)),
                            '100844': dict(distance=int(0.008*fs), height=1e19),
                            '99472': dict(distance=int(0.005*fs), height=0.7e19)
                            }
    
    param_dict_d135 = {
                        '96292': dict(height = 0, distance=int(5e-3 * fs), width = int(0.0005 * fs), rel_height=.5),
                        '94120': dict(height = 0, distance=int(5e-3 * fs), width=int(0.0005*fs), rel_height=.5 ),
                        '96294': dict(height = 0, distance = int(5e-3 * fs), width=int(0.00005*fs), rel_height=.3  ),
                        '100759': dict(height = 0.02, distance = int(4e-3 * fs), width=int(0.00011*fs), rel_height=.3, prominence=0.05  ),
                        '100841': dict(height = 0.0, distance = int(4e-3 * fs), width=int(0.00011*fs), rel_height=.3  ),
                        '100844': dict(height = 0.0, distance = int(4e-3 * fs), width=int(0.00011*fs), rel_height=.3, prominence=0.05  ),
                        '105439': dict(height = 0.0, distance = int(4e-3 * fs), width=int(0.00011*fs), rel_height=.3, prominence=0.05  ),
                        '99472': dict(height=0.03, distance=0.005*fs, width=0, rel_height=.3)
                        }

    param_dict_d254 = {
                        '96292': dict(height = 0, distance=int(4e-3 * fs), width = int(0.0001 * fs), rel_height=.4),
                        '94120': dict(height = 0, distance=int(7e-3 * fs), width=int(0.0005*fs), rel_height=.5 ),
                        '96294': dict(height = 0, distance = int(5e-3 * fs), width=int(0.00005*fs), rel_height=.3  ),
                        '100759': dict(height = 0.02, distance = int(4e-3 * fs), width=int(0.00005*fs), rel_height=.3, prominence=0.05  ),
                        '100841': dict(height = 0, distance = int(4e-3 * fs), width=int(0.00004*fs), rel_height=.3  ),
                        '100844': dict(height = 0, distance = int(4e-3 * fs), width=int(0.00004*fs), rel_height=.3, prominence=0.05  ),
                        '105439': dict(height = 0, distance = int(4e-3 * fs), width=int(0.00004*fs), rel_height=.3, prominence=0.05  ),
                        '99472': dict(height=0.025, distance=0.005*fs, width=0, rel_height=.3)
                        }
    to_return = {
                'tdao':param_dict_tdao,
                'dtai_max':param_dict_dtdai_pos,
                'dtai_min':param_dict_dtdai_neg,
                'd135':param_dict_d135,
                'd254':param_dict_d254
                }
    try:
        return to_return[sig][pulse]
    except:
        return dict(height = 0, distance = int(4e-3 * fs) )
    
def dithers_trigger_max_H_alpha_inner(t, H_alpha, ti, tf, filter='moving_average_gaussan',smoothing_window=3, f_cut_off=500):

    ttdai_cropped, tdai_cropped = crop_data(t, H_alpha, ti=ti, tf=tf)

    fs_tdai = 1 / np.mean(np.diff(ttdai_cropped))

    # Smoth tdai data before finding peaks
    if filter == 'moving_average':
        smoth_tdai=uniform_filter1d(tdai_cropped, smoothing_window)

    if filter == 'moving_average_gaussan':
        smoth_tdai=gaussian_filter1d(tdai_cropped, smoothing_window)

    elif filter == 'step_function':
        smoth_tdai=filter_sig(t, H_alpha, f_cut_off)
    height_requirement = np.max(tdai_cropped ) * 0.5
    prominence_requirement = np.max(tdai_cropped) * 0.09

    peaks_pos, _ = find_peaks(smoth_tdai, distance=int(0.0045*fs_tdai), height=height_requirement, prominence=prominence_requirement)
    t_peaks, y_peaks = ttdai_cropped[peaks_pos], tdai_cropped[peaks_pos]

    

    return t_peaks, y_peaks

def clean_peaks(t_peaks, y_peaks, distance=3.5e-3):
    t_peaks_clean=[]
    y_peaks_clean=[]
    t_peaks_0=t_peaks[0]
    y_peaks_0=y_peaks[0]

    t_peaks_clean.append(t_peaks[0])
    y_peaks_clean.append(y_peaks[0])
    for peak, y_peak in zip(t_peaks, y_peaks):
            
        if peak - t_peaks_0 > distance:
            t_peaks_clean.append(peak)
            y_peaks_clean.append(y_peak)
        t_peaks_0=peak
        y_peaks_0=y_peak

        

    t_peaks_clean=np.array(t_peaks_clean)
    y_peaks_clean=np.array(y_peaks_clean)
    return t_peaks_clean, y_peaks_clean

def clean_peaks_2(t_peaks, y_peaks, distance=3.5e-3):
    t_peaks_clean=[]
    y_peaks_clean=[]
    t_peaks_0=t_peaks[0]
    y_peaks_0=y_peaks[0]

    t_peaks_clean.append(t_peaks[0])
    y_peaks_clean.append(y_peaks[0])
    for peak, y_peak in zip(t_peaks, y_peaks):
            
        if peak - t_peaks_0 > distance:
            t_peaks_clean.append(peak)
            y_peaks_clean.append(y_peak)
            t_peaks_0=peak
            y_peaks_0=y_peak

        
    
    t_peaks_clean=np.array(t_peaks_clean)
    y_peaks_clean=np.array(y_peaks_clean)
    return t_peaks_clean, y_peaks_clean

def dithers_trigger_H_alpha_inner_derivative(pulse, t , H_alpha, ti, tf, smoothing_window=3):
    # Smoth tdai data before finding peaks

    smoth_tdai=gaussian_filter1d(H_alpha, smoothing_window)

    t_tdai_deriv = (t[0:-1]+t[1:])/2

    dt_tdai = np.diff(t)
    dy_tdai = np.diff(smoth_tdai)

    tdai_deriv = dy_tdai/dt_tdai

    fs_d_tdai=1/np.mean(np.diff(t))

    peaks_pos, _ = find_peaks(tdai_deriv, **peak_find_args(pulse, fs_d_tdai, 'dtai_max') )
    t_peaks_pos, y_peaks_pos = get_peaks_in_interval(t_tdai_deriv, tdai_deriv, peaks_pos, ti=ti, tf=tf)


    peaks_neg, _ = find_peaks(-tdai_deriv, **peak_find_args(pulse, fs_d_tdai, 'dtai_min'))
    t_peaks_neg, y_peaks_neg = get_peaks_in_interval(t_tdai_deriv, tdai_deriv, peaks_neg, ti=ti, tf=tf)

    if pulse in ['100759', '100844', '99472']:
        t_peaks_neg, y_peaks_neg = clean_peaks_2(t_peaks_neg, y_peaks_neg, 6e-3)


    return t_peaks_pos, y_peaks_pos, t_peaks_neg, y_peaks_neg

def dithers_trigger_H_alpha_outter(pulse, t, H_alpha, ti, tf):
    
    ttdao_cropped, tdao_cropped = crop_data(t, H_alpha, ti=ti, tf=tf)
    fs_tdao=np.mean(np.diff(ttdao_cropped))
    peaks, _ = find_peaks(-tdao_cropped, **peak_find_args(pulse, fs_tdao, 'tdao'))

    t_peaks, y_peaks = get_peaks_in_interval(ttdao_cropped, tdao_cropped, peaks, ti=ti, tf=tf)

    t_peaks_clean=[]
    y_peaks_clean=[]
    t_peaks_0=t_peaks[0]
    y_peaks_0=y_peaks[0]

    t_peaks_clean.append(t_peaks[0])
    y_peaks_clean.append(y_peaks[0])
    for peak, y_peak in zip(t_peaks, y_peaks):
            
        if peak - t_peaks_0 > 3.5e-3:
            t_peaks_clean.append(peak)
            y_peaks_clean.append(y_peak)
        t_peaks_0=peak
        y_peaks_0=y_peak

    t_peaks_clean=np.array(t_peaks_clean)
    y_peaks_clean=np.array(y_peaks_clean)

    if pulse in ['100759']:
        t_peaks_clean, y_peaks_clean = clean_peaks_2(t_peaks_clean, y_peaks_clean, 6e-3)
    
    return t_peaks_clean, y_peaks_clean

def dithers_trigger_magnetic_signals(pulse, t, mag_sig, ti, tf, sig='d135', filter='moving_average_gaussian', smoothing_window=31, f_cut_off=1000):


    fs = 1 / np.mean(np.diff(t))
    if filter == 'moving_average':
        smooth = uniform_filter1d(mag_sig, smoothing_window)
    if filter == 'moving_average_gaussian':
        smooth = gaussian_filter1d(mag_sig, smoothing_window)
    elif filter == 'step_function':
        smooth = filter_sig(t, mag_sig, f_cut_off)
        
    peaks_position, properties = find_peaks(smooth, **peak_find_args(pulse, fs, sig))

    t_peaks, y_peaks = get_peaks_in_interval(t, smooth, peaks_position, ti=ti, tf=tf)


    return t_peaks, y_peaks

def t_shift_calculator(t_stamps_df, key, ref_key='H_alpha_inner', t_shift=0.):
    #return 0
    return np.mean(t_stamps_df[key]-t_stamps_df[ref_key])+t_shift


def best_sync_key(t_stamps_df, ref_signal, n_bins=70, bin_width=None, use_t_shift_calculator=True, t_shift=0.,
                  plot_all=False, plot_bins=False, plot_kws=dict(),
                  return_std = False,
                  return_all_std=False,
                  **use_t_shift_calculator_kwargs):
    assert len(ref_signal)==2, 'ref_signal should be a tuple with (t, signal)'
    assert (n_bins is None) or (bin_width is None), 'n_bins and bin_width cannot be provided at the same time'
    dict_std={}

    for key in t_stamps_df.keys():
        if use_t_shift_calculator:
            t_shift = t_shift_calculator(t_stamps_df, key, **use_t_shift_calculator_kwargs)
        
        t_sync = sync_signals(ref_signal[0], t_stamps_df[key], t_shift=t_shift)

        df_sync = pd.DataFrame()
        df_sync['t_sync'] = t_sync[~np.isnan(t_sync)]
        df_sync['signal'] = ref_signal[1][~np.isnan(t_sync)]
        df_sync['time'] = ref_signal[0][~np.isnan(t_sync)]

        if n_bins:
            map_time, bins_time = pd.qcut(df_sync['t_sync'], n_bins, precision=50, retbins=True, duplicates='drop')
        elif bin_width:
            bins_time = np.arange(df_sync['t_sync'].min(), df_sync['t_sync'].max(), bin_width)
            map_time = pd.cut(df_sync['t_sync'], bins_time, precision=50, duplicates='drop')

        std=df_sync.groupby([map_time], observed=False).std().sum()['signal']

        dict_std[key] = std

        if plot_all:
            fig, ax = plots(x=df_sync['t_sync'], y=df_sync['signal'], z=df_sync['time'],
                            title=f"Sync. by {key} - Avg STD = {std}",
                            x_labels='t_sync', y_labels='signal', z_labels='time',
                            markersizes=5,
                            **plot_kws)
            if plot_bins:
                for bin_time in bins_time:
                    ax.axvline(bin_time, c='r', ls='--')
    
    if return_all_std:
        return min(dict_std, key=dict_std.get), dict_std
    if return_std:
        return min(dict_std, key=dict_std.get), dict_std[min(dict_std, key=dict_std.get)]
    else:
        return min(dict_std, key=dict_std.get)

def automatic_bins(t_s, dt, n=None):
    """
    Automatic number of bins for a given time array
    
    Args:
        t_s (array): time array
        dt (float): time interval
        n (int): number of points per bin
    """
    assert (dt is None) or (n is None), 'dt and n cannot be provided at the same time'
    t_s = t_s[~np.isnan(t_s)]
    if dt:
        t0 = np.min(t_s)
        return int(len(t_s)/len(t_s[(t_s>=t0) & (t_s<=t0+dt)]))
    elif n:
        return int(len(t_s)/n)
