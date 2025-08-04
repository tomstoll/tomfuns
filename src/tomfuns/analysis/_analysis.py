# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:00:35 2024

@author: tomstoll
"""
import numpy as np
import scipy.signal as sig


def butter_filter(data, fs, l_freq=1, h_freq=None, order=1,
                  return_mne_params=False):
    """
    Create a butterworth filter and either apply it causally or return the
    parameters to apply it causally using MNE's ``filter_data`` function.

    Parameters
    ----------
    data : array | None
        The data to be filtered. The filter will be applied on the last axis.
        Can be None if using this function to generate filter parameters
        for mne filtering.
    fs : float
        The sample rate of the data.
    l_freq : float | None
        The lower filter cutoff frequency. The default is 1. If None the
        data will not be high-passed.
    h_freq : float | None, optional
        The upper filter cutoff frequency. If None (default) the data will
        not be low-passed.
    order : int
        The order of the filter. The default is 1.
    return_mne_params : bool, optional
        Option to return parameters necessary to use MNE's ``filter_data``
        function. When True, will only return filter parameters and will not
        filter ``data``. The default is False.

    Returns
    -------
    data : array
        The filtered data. Only returned if ``return_mne_params=False``.
    params : dict
        A dictionary containing the parameters to perform causal iir filtering
        using MNE's ``filter_data`` function (e.g., through raw.filter().
        Only returned if ``return_mne_params=True``.

    """
    from warnings import warn
    if l_freq is not None and h_freq is not None:  # bandpass
        sos = sig.butter(order, [l_freq, h_freq], btype='band', fs=fs,
                         output='sos')
    elif l_freq is not None and h_freq is None:  # highpass
        sos = sig.butter(order, l_freq, btype='high', fs=fs, output='sos')
    elif l_freq is None and h_freq is not None:  # lowpass
        sos = sig.butter(order, h_freq, btype='low', fs=fs, output='sos')
    elif l_freq is None and h_freq is None:  # no filtering
        warn('Lowcut and highcut are None. No filtering is done.')
        if return_mne_params:
            sos = sig.tf2sos(1, 1)  # do no filtering
        else:
            return data

    if return_mne_params or data is None:
        params = {'method': 'iir', 'phase': 'forward',
                  'l_freq': l_freq, 'h_freq': h_freq,
                  'iir_params': {'sos': sos}}
        return params
    else:
        data = sig.sosfilt(sos, data)
        return data


def notch_filt(data, fs, freqs, notch_width):
    """
    A function to notch filter your data. Can be applied to MNE raw data
    using raw.apply_function(notch_filt, **params), where params is a dict
    containing fs, freqs, and notch_width.

    Parameters
    ----------
    data : ndarray
        The data to be filtered. Must have time on the last axis.
    fs : float
        The sampling rate of data.
    freqs : int or list or ndarray
        The frequencies which should be filtered out using a notch filter.
    notch_width : float
        The width of the notch.

    Returns
    -------
    data : ndarray
        The notch-filtered data.

    """

    if np.size(freqs) == 1:
        freqs = [freqs]
    for f in freqs:
        b, a = sig.iirnotch(f, f/notch_width, fs=fs)
        data = sig.lfilter(b, a, data, axis=-1)
    return data


def drift_correct_epochs(eps, start_trigs, drift_trigs, trial_times):
    """ Modifies epochs in place to correct for clock drift.
    eps : Epochs object
    start_trigs : array
        samples each epoch started
    drift_trigs : array
        samples each drift trigger was stamped
    trial_times : float or array
        the actual time between the start trigger and drift trigger.
        can be a scalar if the time was the same for all trials.
    """
    from mne import resample
    eps_shape = eps._data.shape
    n_trials = eps_shape[0]
    n_samps = eps_shape[-1]
    if np.isscalar(trial_times):
        trial_times = np.ones(n_trials)*trial_times
    for ti in range(n_trials):
        fs_actual = (drift_trigs[ti]-start_trigs[ti])/trial_times[ti]
        tmp = eps._data[ti].copy()
        tmp = resample(tmp, eps.info['sfreq'], fs_actual,
                       n_jobs=-1, axis=-1)[..., :n_samps]
        eps._data[ti] = np.pad(tmp, [[0, 0], [0, n_samps-tmp.shape[-1]]],
                               mode='constant')
