# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:00:35 2024

@author: tomstoll
"""
import numpy as np
import scipy.signal as sig


def gen_pip(freq, fs, n_per=5, stim_gen_rms=0.01):
    """
    Simple function to generate a single pip (short toneburst).

    Parameters
    ----------
    freq : float
        The frequency of the pip.
    fs : TYPE
        The sampling rate of the pip.
    n_per : float, optional
        How many periods should be in the pip. The default is 5.
    stim_gen_rms : float, optional
        The RMS of a reference sine wave, used to scale the pip to a
        peak-equivalent amplitude. The default is 0.01.

    Returns
    -------
    pip : ndarray
        A single tone pip.

    """

    t = np.arange(int(fs * n_per / freq)) / fs
    pip = (np.cos(2 * np.pi * freq * t) *
           (-0.5 + 0.5 * np.cos(2 * np.pi * freq / n_per * t)))
    pip *= stim_gen_rms / np.sqrt(0.5)
    return pip


def gen_click(fs, dur_trial=None, click_dur=100e-6, stim_gen_rms=0.01):
    """
    Simple function to generate a single condensation click.

    Parameters
    ----------
    fs : float
        The sampling rate of the click.
    dur_trial : float or None, optional
        How long the output stimulus should be (in seconds). Must be greater
        than the click duration. If None (default), it will be one second.
    click_dur : float, optional
        The duration of the click, in seconds. The default is 100e-6.
    stim_gen_rms : float, optional
        The RMS of a reference sine wave, used to scale the pip to a
        peak-equivalent amplitude. The default is 0.01.

    Returns
    -------
    click : ndarray
        A single click.

    """

    dur_trial = int(fs*2*click_dur) if dur_trial is None else int(dur_trial*fs)
    assert dur_trial > int(fs*click_dur)
    click = np.zeros(dur_trial)
    click[1:int(fs*click_dur)+1] = 1
    click *= stim_gen_rms / np.sqrt(0.5)
    return click


def gen_rand_impluse_train(fs, rate, dur_trial=1, stim_dur=100e-6, seed=None,
                           flip_half=True):
    """
    Generate a randomly-timed impulse train. Useful to generate stimuli.

    Parameters
    ----------
    fs : float
        The sampling rate of the impulse train.
    rate : float
        How many pulses should be present in one second.
    dur_trial : float, optional
        How long the output impulse train should be (in seconds).
        The default is 1.
    stim_dur : float, optional
        The duration of stimuli that will be convolved with the impulse train.
        Used to prevent stimuli from being cut off at the beginning/end of the
        resulting toneburst/click train. Can be 0 if not needed.
        The default is 100e-6.
    seed : float or None, optional
        The seed used for the random generator See np.random.default_rng.
        The default is None.
    flip_half : bool, optional
        If True, invert half of the impulses. The default is True.

    Returns
    -------
    x_pulse : ndarray
        An impulse train with random timing.

    """

    x_pulse = np.zeros(int(dur_trial*fs))
    n_pulse = 2 * (np.round(rate * dur_trial -  # number of impulses
                            stim_dur).astype(int) // 2)
    rng = np.random.default_rng(seed)
    len_trial = int(dur_trial * fs)
    pulse_inds = np.array(rng.permutation(len_trial - int(fs*stim_dur))
                          + int(fs*stim_dur/2))[:n_pulse]
    x_pulse[pulse_inds] = -1
    if flip_half:
        inv_inds = pulse_inds[rng.permutation(n_pulse)[:n_pulse//2]]
        x_pulse[inv_inds] *= -1
    return x_pulse


def gen_fixed_timing_train(fs, n_pulses, dt, dur_trial=1,
                           alternate_polarity=False):
    """
    Generate a simple impulse train, with pulses spaced at a user-defined
    interval

    Parameters
    ----------
    fs : float
        The sampling rate of the impulse train.
    n_pulses : int
        How many impulses should be in the train.
    dt : float
        The amount of time between the impulses, in seconds.
    dur_trial : float, optional
        The length of the impulse train, in seconds. The default is 1.
    alternate_polarity : bool, optional
        If True, alternate the polarity of the impulses. The default is False.

    Returns
    -------
    x_pulse : ndarray
        An impulse train with the first impulse at 0 and folloiwng impulses
        (if any) following with time dt between them.

    """
    assert isinstance(n_pulses, int), ('The number of pulses must be an '
                                       'integer.')
    length = int(fs*dur_trial)
    spacing = int(np.round(dt*fs))
    x_pulse = np.zeros(length)
    pulse_indcs = [spacing*pulse_i for pulse_i in range(n_pulses)]
    x_pulse[pulse_indcs[::2]] = 1
    x_pulse[pulse_indcs[1::2]] = -1
    return x_pulse


def gen_pip_train(fs, freq, rate, stim_gen_rms=0.01, n_per=5, dur_trial=1,
                  seed=None, flip_half=True, return_timing_train=True):
    """
    A function to generate a randomly timed pip train.

    Parameters
    ----------
    fs : float
        The sampling rate of the pip train.
    freq : float
        The frequency of the pips.
    rate : float
        How many pips there should be in one second.
    stim_gen_rms : float, optional
        The RMS of a reference sine wave. Useful to later scale the pip train
        to a peak-equivalent amplitude. The default is 0.01.
    n_per : float, optional
        The number of periods in a pip. The default is 5.
    dur_trial : float, optional
        The desired duration (in seconds) of the pip train. The default is 1.
    seed : float or None, optional
        The seed used for the random generator See np.random.default_rng.
        The default is None.
    flip_half : bool, optional
        if True, invert half of the pips to generate a pip train with both
        rarefaction and condensation pips. The default is True.
    return_timing_train : bool, optional
        If True, return the timing train used to generate the pip train.
        The default is True.

    Returns
    -------
    x_pulse : ndarray
        The timing train used to generate the pip train. The timing train will
        have unit impulses at the start of each pip, with polarity matching
        the corresponding pip. Only returned if return_timing_train is True.
    x : ndarray
        The resulting pip train.

    """

    pip = gen_pip(freq, fs, n_per, stim_gen_rms)
    stim_dur = len(pip)/fs
    x_pulse = gen_rand_impluse_train(fs, rate, dur_trial, stim_dur, seed,
                                     flip_half)
    pip_train = sig.fftconvolve(x_pulse, pip)[:int(fs*dur_trial)]
    if return_timing_train:
        return x_pulse, pip_train
    else:
        return pip_train


def gen_click_train(fs, rate=None, click_dur=100e-6, stim_gen_rms=0.01,
                    dur_trial=1, seed=None, flip_half=None, n_pulses=None,
                    dt=None, return_timing_train=True,
                    timing_fn=gen_rand_impluse_train):
    """
    A function to generate a click train.

    Parameters
    ----------
    fs : float
        The sampling rate of the click train.
    rate : float or None, optional
        How many pips there should be in one second. Should be None if not
        used in the funciton used to generate the timing train.
    click_dur : float, optional
        The duration of the click, in seconds. The default is 100e-6.
    stim_gen_rms : float, optional
        The RMS of a reference sine wave. Useful to later scale the click train
        to a peak-equivalent amplitude. The default is 0.01.
    dur_trial : float, optional
        The desired duration (in seconds) of the click train. The default is 1.
    seed : float or None, optional
        The seed used for the random generator See np.random.default_rng.
        The default is None.
    flip_half : bool or None, optional
        if True, invert half of the clicks to generate a click train with both
        rarefaction and condensation clicks. The default is None.
    return_timing_train : bool, optional
        If True, return the timing train used to generate the click train.
        The default is True.
    timing_fn : function, optional
        The function used to place the clicks. Can be used to control if the
        click train has random (default) or uniform timing.

    Returns
    -------
    x_pulse : ndarray
        The timing train used to generate the click train. The timing train
        will have unit impulses at the start of each click, with polarity
        matching  the corresponding click.
        Only returned if return_timing_train is True.
    x : ndarray
        The resulting click train.

    """

    click = gen_click(fs, dur_trial, click_dur, stim_gen_rms)
    timing_fn_args = {'fs': fs, 'dur_trial': dur_trial, 'click_dur': click_dur}
    for key, value in zip(['rate', 'seed', 'flip_half', 'n_pulses', 'dt'],
                          [rate, seed, flip_half, n_pulses, dt]):
        if value is not None:
            timing_fn_args[key] = value
    x_pulse = timing_fn(**timing_fn_args)
    click_train = sig.fftconvolve(x_pulse, click)[:int(fs*dur_trial)]
    if return_timing_train:
        return x_pulse, click_train
    else:
        return click_train
