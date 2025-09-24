# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:51:35 2025

@author: tomstoll
"""
import numpy as np
import scipy.signal as sig
from warnings import warn
try:
    import pyzbc2014 as zbc
except ImportError as e:
    raise ImportError(
        "The tomfuns.models submodule requires pyzbc2014. "
        "Install it with 'pip install tomfuns[models]'."
    ) from e


# %% From https://github.com/HearingTechnology/Verhulstetal2018Model/blob/master/ic_cn2018.py
def cochlearNuclei(anfH, anfM, anfL, numH, numM, numL, fs):
    Acn = 1.5
    Scn = 0.6
    inhibition_delay = int(round(1e-3*fs))
    Tex = 0.5e-3
    Tin = 2e-3

    summedAN = numL*anfL + numM*anfM + numH*anfH

    delayed_inhibition = np.zeros_like(summedAN)
    delayed_inhibition[inhibition_delay:,
                       :] = summedAN[0:len(summedAN)-inhibition_delay, :]

    # filters obtained with bilinear transform
    # # Excitatory filter:
    m = (2*Tex*fs)
    a = (m-1)/(m+1)
    bEx = 1.0/(m+1)**2*np.array([1, 2, 1])  # numerator
    aEx = np.array([1, -2*a, a**2])  # denominator

    # # Inhibitory filter:
    m = (2*Tin*fs)
    a = (m-1)/(m+1)
    bIncn = 1.0/(m+1)**2*np.array([1, 2, 1])  # numerator
    aIncn = np.array([1, -2*a, a**2])  # denominator

    cn = Acn*(sig.lfilter(bEx, aEx, summedAN, axis=0)-Scn *
              sig.lfilter(bIncn, aIncn, delayed_inhibition, axis=0))
    return cn, summedAN


def inferiorColliculus(cn, fs):
    Tex = 0.5e-3
    Tin = 2e-3
    Aic = 1
    Sic = 1.5
    inhibition_delay = int(round(2e-3*fs))

    delayed_inhibition = np.zeros_like(cn)
    delayed_inhibition[inhibition_delay:,
                       :] = cn[0:len(cn)-inhibition_delay, :]

    # # Excitatory filter:
    m = (2*Tex*fs)
    a = (m-1)/(m+1)
    bEx = 1.0/(m+1)**2*np.array([1, 2, 1])  # numerator
    aEx = np.array([1, -2*a, a**2])  # denominator

    # # Inhibitory filter:
    m = (2*Tin*fs)
    a = (m-1)/(m+1)
    bIncn = 1.0/(m+1)**2*np.array([1, 2, 1])  # numerator
    aIncn = np.array([1, -2*a, a**2])  # denominator

    ic = Aic*(sig.lfilter(bEx, aEx, cn, axis=0)-Sic *
              sig.lfilter(bIncn, aIncn, delayed_inhibition, axis=0))
    return ic


# %% Define functions to get AN rates and model ABR
def run_zbc(x, cf, fs, fibertype='hsr'):
    if fs < 100e3:
        raise RuntimeError(
            f"Sample rate must be at least 100e3, got {fs}. Upsample before "
            "calling this function."
            )
    ihc_potential = zbc.sim_ihc_zbc2014(x, cf, fs=fs)
    anrate = zbc.sim_anrate_zbc2014(ihc_potential, cf=cf, fs=fs,
                                    fibertype=fibertype)
    return anrate


def model_abr(x, fs, cfs=None, n_hsf=13, n_msf=3, n_lsf=3,
              n_jobs=1):
    """
    Model an ABR waveform based on the 2014 Carney AN model and 2004 Nelson
    and Carney IC/CN model (as implemented in Verhulst 2018).

    Parameters
    ----------
    x : ndarray
        The pressure waveform of the sound for which the ABR should be modeled.
        It should be a one-dimensional numpy array that is scaled to be in
        units of Pascals.
    fs : float
        The sampling frequency of the stimulus, x. If not over 100e3 kHz, the
        stimulus will be upsampled using mne's resample function before being
        input to the model.
    cfs : int, list, ndarray, optional
        The characteristic frequencies of the auditory nerve fibers to be
        simulated. If None, 43 fibers will be simulated with the cf ranging
        from 125 to 16000 Hz in 1/6th octave steps. The default is None.
    n_hsf : int, optional
        The number of high spontaneous rate auditory nerve fibers that
        contribute to the response. Note that only one fiber is actually
        simulated. This number is used to scale that fiber's contribution. The
        default is 13.
    n_msf : TYPE, optional
        The number of medium spontaneous rate auditory nerve fibers that
        contribute to the response. Note that only one fiber is actually
        simulated. This number is used to scale that fiber's contribution. The
        default is 3.
    n_lsf : TYPE, optional
        The number of low spontaneous rate auditory nerve fibers that
        contribute to the response. Note that only one fiber is actually
        simulated. This number is used to scale that fiber's contribution. The
        default is 3.
    n_jobs : int, optional
        The maximum number of jobs to be run in parallel. This will be passed
        to joblib as well as mne's resample function. If 1, the code will be
        run serially (using for loops). The default is 1.

    Returns
    -------
    abr : ndarray
        The ABR waveform, with (hopefully) reasonable scaling and timing of
        response components (waves I, III, and V).

    """
    # check inputs, set defaults
    if not isinstance(x, np.ndarray):
        x = np.array(x).squeeze()  # in case the input is a list
    # make sure the input stimulus is 1d
    if not x.ndim == 1:
        raise ValueError(
            f"The input stimulus must be one dimensional, but has shape "
            f"{x.shape}."
            )
    if n_jobs != 1:  # make sure we can use joblib if n_jobs isn't 1
        try:
            from joblib import Parallel, delayed
            run_parallel = True
        except ModuleNotFoundError:
            warn(
                "Could not import joblib, falling back to serial "
                "implementation, which will be slower."
                )
            run_parallel = False
    else:
        run_parallel = False

    if cfs is None:  # default cfs
        cfs = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1/6)
    else:
        cfs = np.atleast_1d(cfs)
    # make sure the number of fibers makes sense
    if n_hsf + n_msf + n_lsf <= 0:
        raise ValueError(
            "The number of fibers for at least one fiber type must be greater "
            f"than zero, got n_hsf={n_hsf}, n_msf={n_msf}, n_lsf={n_lsf}."
            )
    check_ints = [n_hsf, n_msf, n_lsf]
    check_ints_str = ['n_hsf', 'n_msf', 'n_lsf']
    for ci, cis in zip(check_ints, check_ints_str):
        if not isinstance(ci, int) or ci < 0:
            raise ValueError(
                f"{cis} must be a non-negative integer, got {ci}"
                )
    # upsample if needed
    fs_up = 100e3
    downsample = False
    if not fs >= fs_up:
        from mne.filter import resample
        downsample = True
        x = resample(x, fs_up, fs, n_jobs=n_jobs)
    else:  # if fs is higher than fs_up, use it as is
        fs_up = fs

    # simulate the responses for different types of AN fibers
    # initialize so unsimulated types won't give an error
    h_rate = []
    m_rate = []
    l_rate = []
    if run_parallel:
        if n_hsf > 0:
            h_rate = Parallel(n_jobs=n_jobs)([delayed(run_zbc)(x, cf, fs_up,
                                                               fibertype="hsr")
                                              for cf in cfs])
        if n_msf > 0:
            m_rate = Parallel(n_jobs=n_jobs)([delayed(run_zbc)(x, cf, fs_up,
                                                               fibertype='msr')
                                              for cf in cfs])
        if n_lsf > 0:
            l_rate = Parallel(n_jobs=n_jobs)([delayed(run_zbc)(x, cf, fs_up,
                                                               fibertype='lsr')
                                              for cf in cfs])
    else:
        for cf in cfs:
            if n_hsf > 0:
                h_rate += [run_zbc(x, cf, fs_up, fibertype='hsr')]
            if n_msf > 0:
                m_rate += [run_zbc(x, cf, fs_up, fibertype='msr')]
            if n_lsf > 0:
                l_rate += [run_zbc(x, cf, fs_up, fibertype='lsr')]

    # set rate to scalar 0 if n_fibers is 0, to prevent dimension mismatches
    h_rate = 0 if n_hsf == 0 else h_rate
    m_rate = 0 if n_msf == 0 else m_rate
    l_rate = 0 if n_lsf == 0 else l_rate

    h_rate = np.array(h_rate)
    m_rate = np.array(m_rate)
    l_rate = np.array(l_rate)

    # sum the responses from different fibers together, simulate CN and IC
    w3, w1 = cochlearNuclei(h_rate.T, m_rate.T, l_rate.T,
                            n_hsf, n_msf, n_lsf, fs_up)
    w5 = inferiorColliculus(w3, fs_up)

    # shift, scale, and sum responses to model abr waveform
    w1_shift = int(fs_up*1e-3)
    w3_shift = int(fs_up*2.25e-3)
    w5_shift = int(fs_up*3.5e-3)

    # use scaling from Verhulst code, modify based on number of cfs used here
    # (Verhulst used 401 in their code)
    M1 = 4.2767e-14
    M3 = 5.1435e-14
    M5 = 13.3093e-14
    correction_factor = 401/len(cfs)
    w1 = np.roll(np.sum(w1, axis=1)*M1*correction_factor, w1_shift)
    w3 = np.roll(np.sum(w3, axis=1)*M3*correction_factor, w3_shift)
    w5 = np.roll(np.sum(w5, axis=1)*M5*correction_factor, w5_shift)

    # clean up the roll (get rid of response that rolled over to start)
    w1[:w1_shift] = w1[w1_shift]
    w3[:w3_shift] = w3[w3_shift]
    w5[:w5_shift] = w5[w5_shift]

    abr = w1+w3+w5

    if downsample:  # make the output match the input sampling rate
        abr = resample(abr, fs, fs_up, n_jobs=n_jobs)
    return abr


def gen_anm(x, fs_in, cfs=None, fs_out=None, shift_ms=2.75, n_jobs=1):
    """
    Function to generate the ANM regressor described in Shan et al. 2024.
    Parameters
    ----------
    x : ndarray
        The pressure waveform of the sound for which the ANM regressor should
        be calculated. Must be a one-dimensional numpy array that is scaled to
        be in units of Pascals.
    fs_in : float
        The sampling frequency of the stimulus, x. If not over 100e3 kHz,
        the stimulus will be upsampled using mne's resample function before
        being input to the model.
    cfs : int, list, ndarray, optional
        The characteristic frequencies of the auditory nerve fibers to be
        simulated. If None, 43 fibers will be simulated with the cf ranging
        from 125 to 16000 Hz in 1/6th octave steps. The default is None.
    shift_ms : float
        How much to shift the final output, in miliseconds. The default
        is 2.75 ms, which matches Shan et al. 2024.
    n_jobs : int, optional
        The maximum number of jobs to be run in parallel. This will be passed
        to joblib as well as mne's resample function. If 1, the code will be
        run serially (using for loops). The default is 1.
    """
    # Check inputs, set defaults
    if not isinstance(x, np.ndarray):
        x = np.array(x).squeeze()  # in case the input is a list
    # make sure the input stimulus is 1d
    if not x.ndim == 1:
        raise ValueError(
            f"The input stimulus must be one dimensional, but has shape "
            f"{x.shape}."
            )
    if n_jobs != 1:  # make sure we can use joblib if n_jobs isn't 1
        try:
            from joblib import Parallel, delayed
            run_parallel = True
        except ModuleNotFoundError:
            warn(
                "Could not import joblib, falling back to serial "
                "implementation, which will be slower."
                )
            run_parallel = False
    else:
        run_parallel = False

    if cfs is None:  # default cfs
        cfs = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1/6)
    else:
        cfs = np.atleast_1d(cfs)

    # Upsample if needed
    fs_up = int(100e3)
    downsample = False
    if fs_in < 100e3:
        from mne.filter import resample
        downsample = True
        x = resample(x, fs_up, fs_in, n_jobs=n_jobs)
    else:
        fs_up = fs_in
    if fs_out is None:
        fs_out = fs_in
    if fs_out != fs_in:
        from mne.filter import resample
        downsample = True

    float_params = {'fs_in': fs_in, 'fs_out': fs_out, 'shift_ms': shift_ms}
    for name, val in float_params.items():
        if not isinstance(val, (float, int)):
            raise TypeError(f"{name} must be a float or int, got '{val}'.")

    if run_parallel:
        anf_rates = Parallel(n_jobs=n_jobs)([delayed(run_zbc)
                                             (x, cf, fs_up)
                                             for cf in cfs])
    else:
        anf_rates = []
        for cf in cfs:
            anf_rates += [run_zbc(x, cf, fs_up)]
    anf_rates = np.array(anf_rates)

    # Downsample to the output sampling frequency
    if downsample:
        anf_rates = resample(anf_rates, fs_out, fs_up, npad='auto',
                             n_jobs=n_jobs)

    # Scale, sum, and shift
    M1 = 4.2767e-14 * 401/len(cfs)
    anm = M1*anf_rates.sum(0)
    final_shift = int(fs_out*shift_ms/1000)
    anm = np.roll(anm, final_shift)
    anm[:final_shift] = anm[final_shift]
    return anm


def scale_stim(x, stim_db, stim_gen_rms=0.01):
    """
    Helper function to scale a stimulus so that it is in units of pascals

    Parameters
    ----------
    x : ndarray
        The waveform of the stimulus to be scaled. Must be a numpy array.
    stim_db : float
        The desired sound level of the stimulus.
    stim_gen_rms : float, optional
        The reference RMS used to generate the stimulus. The default is 0.01.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
    if not isinstance(x, np.ndarray):
        raise TypeError(
            f"x must be a numpy array, got {type(x)}."
            )
    sine_rms_at_0db = 20e-6
    scale_factor = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_db / 20.))
    x *= scale_factor
    return x
