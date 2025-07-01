import numpy as np
from scipy.signal import find_peaks

def find_max_peak(x, max_key='prominences', **find_peaks_kw):

    default_find_peaks_kw = dict(height=(None, None), prominence=(None, None))
    default_find_peaks_kw.update(**find_peaks_kw)

    peaks, properties = find_peaks(x, **default_find_peaks_kw)

    j = np.argmax(properties[max_key])

    for key in properties.keys():
        properties[key]=properties[key][j]

    return peaks[j], properties