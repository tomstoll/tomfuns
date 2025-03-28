# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:00:35 2024

@author: tomstoll
"""
import numpy as np
import matplotlib.pyplot as plt


def plotting_defaults(fontsize=10, figsize=(9, 6)):
    """
    Set some default plotting parameters to try and make plots look nicer.
    I recommend using matplotlib stylesheets when making figures for a final
    paper, as it's easy to set/chagne formating to match journal guidelines
    that way.
    """
    # set font size and family
    font_size = fontsize
    font = {'family': 'Arial',
            'size': font_size}
    plt.matplotlib.rcParams['pdf.fonttype'] = 42  # embed fonts in pdfs
    plt.rc('font', **font)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

    plt.rcParams.update({"axes.grid": True})  # grid on by default

    # set default figure size
    plt.rc('figure', figsize=figsize)


def get_greek_unicode(letter):
    """
    Table lookup function to get the unicode string corresponding to a
    greek letter.

    Parameters
    ----------
    letter : str
        The greek letter for which the unicode string is desired. This is
        case sensitive, where the first letter indicates if the capital or
        lowercase letter should be returned.

    Returns
    -------
    str
        The unicode corresponding to the desired greek letter.

    """
    if letter[0].isupper():
        letter = letter.capitalize()

    # handle a couple modern vs ancient greek spellings
    aliases = {'Lamda': 'Lambda', 'lamda': 'lambda', 'Ni': 'Nu', 'ni': 'nu'}
    if letter in aliases.keys():
        letter = aliases[letter]

    greek_uni = {
        'Alpha': '\u0391',
        'alpha': '\u03b1',
        'Beta': '\u0392',
        'beta': '\u03b2',
        'Gamma': '\u0393',
        'gamma': '\u03b3',
        'Delta': '\u0394',
        'delta': '\u03b4',
        'Epsilon': '\u0395',
        'epsilon': '\u03b5',
        'Zeta': '\u0396',
        'zeta': '\u03b6',
        'Eta': '\u0397',
        'eta': '\u03b7',
        'Theta': '\u0398',
        'theta': '\u03b8',
        'Iota': '\u0399',
        'iota': '\u03b9',
        'Kappa': '\u039a',
        'kappa': '\u03ba',
        'Lambda': '\u039b',
        'lambda': '\u03bb',
        'Mu': '\u039c',
        'mu': '\u03bc',
        'Nu': '\u039d',
        'nu': '\u03bd',
        'Xi': '\u039e',
        'xi': '\u03be',
        'Omicron': '\u039f',
        'omicron': '\u03bf',
        'Pi': '\u03a0',
        'pi': '\u03c0',
        'Rho': '\u03a1',
        'rho': '\u03c1',
        'Sigma': '\u03a3',
        'sigma': '\u03c3',
        'Tau': '\u03a4',
        'tau': '\u03c4',
        'Upsilon': '\u03a5',
        'upsilon': '\u03c5',
        'Phi': '\u03a6',
        'phi': '\u03c6',
        'Chi': '\u03a7',
        'chi': '\u03c7',
        'Psi': '\u03a8',
        'psi': '\u03c8',
        'Omega': '\u03a9',
        'omega': '\u03c9',
        }
    try:
        return greek_uni[letter]
    except KeyError:
        raise KeyError("The letter '%s' is not in the table." % letter)


def get_pabr_colors():
    """
    Return a 5x4 numpy array with the colors we typically use to plot pABR
    responses. Each row corresponds to a different frequency (with index 0
    corresponding to 500 Hz, 1 to 1 kHz, etc.).
    """
    import cmocean
    cm_lines, cmlb, cmub = cmocean.cm.phase, 1.0, 0.2
    return cm_lines(np.linspace(cmlb, cmub, 5))


class um_cols:
    """
    Stores the official UM colors (for which there are hex codes).
    From https://brand.umich.edu/design-resources/colors/
    """
    # primary colors
    maize = '#FFCB05'
    blue = '#00274C'

    # secondary colors
    red = '#9A3324'
    orange = '#D86018'
    r_green = '#75988d'
    w_green = '#A5A508'
    teal = '#00B2A9'
    a_blue = '#2F65A7'
    amethyst = '#702082'
    violet = '#575294'
    tan = '#CFC096'
    beige = '#9B9A6D'
    gray = '#989C97'
    stone = '#655A52'
    black = '#131516'
