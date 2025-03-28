# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:00:35 2024

@author: tomstoll
"""
import numpy as np
import os


def checkpath(path):
    """
    Checks if a path exists and makes it if it doesn't.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    None.

    """
    if not os.path.isdir(path):
        os.makedirs(path)


def bin_4s8s_to_dec(bin_vals):
    """
    Simple function to convert a binary number encoded with 4 and 8 as 0 and 1
    to its decimal representation.

    Parameters
    ----------
    bin_vals : str or int or array-like
        A string, integer, or list or array of strings or integers, containing
        only 4s and 8s, representing a binary number where 4=0 and 8=1.
        If a numpy array, the first dimension must be the number of values.

    Returns
    -------
    dec_vals.

    """
    return_single_num = False
    return_ndarray = False
    if isinstance(bin_vals, (str, int)):
        bin_vals = [bin_vals]
        return_single_num = True
    if isinstance(bin_vals, np.ndarray):
        bin_vals = [''.join(b) for b in bin_vals.astype(int).astype(str)]
        return_ndarray = True

    dec_vals = []
    for bv in bin_vals:
        if not isinstance(bv, str):
            bv = str(bv)
        if not all([b in ['4', '8'] for b in bv]):
            raise RuntimeError("Input must contain only 4s and 8s.")
        dec_vals += [int(''.join([str(int(b) >> 3) for b in bv]), 2)]

    if return_ndarray:
        dec_vals = np.array(dec_vals)
    elif return_single_num:
        dec_vals = dec_vals[0]
    return dec_vals
