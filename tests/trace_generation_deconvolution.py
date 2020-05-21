#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:26:45 2020
This file gives examples of generation of calcium traces and deconvolution 
using OASIS
@author: @caichangjia adapted based on @j-friedrich code 
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from oasis.functions import deconvolve
import scipy
from scipy.interpolate import interp1d

#%%
def gen_data(g=[.95], sn=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13, trueSpikes=None, nonlinearity=False):
    """
    Generate data from homogenous Poisson Process

    Parameters
    ----------
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : int, optional, default .5
        Neural firing rate.
    b : int, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    seed : int, optional, default 13
        Seed of random number generator.
    trueSpikes: array, optional , default None
        input spike train
    nonlinearity: boolean, default False
        whether add nonlinearity or not

    Returns
    -------
    y : array, shape (T,)
        Noisy fluorescence data.
    c : array, shape (T,)
        Calcium traces (without sn).
    s : array, shape (T,)
        Spike trains.
    """
    
    if trueSpikes is None:
        np.random.seed(seed)
        Y = np.zeros((N, T))
        trueSpikes = np.random.rand(N, T) < firerate / float(framerate)
    else:
        N, T = trueSpikes.shape
        Y = np.zeros((N, T))
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    if nonlinearity is not False:
        maximum = truth.max(axis=1)
        f = nonlinear_transformation()
        truth = f(truth)
        truth = truth / truth.max(axis=1)[:, np.newaxis] * maximum[:, np.newaxis]
    Y = b + truth + sn * np.random.randn(N, T)            
    return Y, truth, trueSpikes
    
def nonlinear_transformation():
    """
    Output function for nonlinear transformation. The transformation is based on
    the property of GCAMP6f
    
    Returns:
    -------
    f : function
        Nonlinear transformation
    """
    x = np.array(range(-2,9))
    x = np.array([0.0, 1.0, 2.0, 3.0,  4.0 ])
    y = np.array([0.0, 0.5, 0.8, 1.5, 2.2])    
    f = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
    xnew = np.arange(-2, 10, 0.01)
    ynew = f(xnew)
    plt.figure()
    plt.title('Nonlinear transformation')
    plt.plot(xnew, ynew)
    plt.xlabel('# AP')
    plt.ylabel('deltaF/F')
    return f

#%% Calcium trace generation
Y, truth, trueSpikes = gen_data(firerate=2, N=1000)
Y_nl, _, _ = gen_data(trueSpikes=trueSpikes, nonlinearity=True)

#%% Deconvolution using OASIS
index = 0
c, s, b, g, lam = deconvolve(Y[index])
c_nl, s_nl, b_nl, g_nl, lam_nl = deconvolve(Y_nl[index])

#%% Show without nonlinear transformation result
framerate=30
plt.figure()
tt = np.arange(0, len(c) * 1 / 30, 1 / 30)
plt.plot(tt, Y[index], label='trace')
plt.plot(tt, c, label='deconvolve')
plt.plot(tt, truth[index], label='ground truth signal')
plt.plot(tt, trueSpikes[index],  label='ground truth spikes')
plt.plot(tt, s, label='deconvolved spikes')
plt.legend()
np.corrcoef(s, trueSpikes[index])


#%% Result with nonlinear transformation
plt.figure()
plt.plot(tt, Y_nl[index], label='trace')
plt.plot(tt, c_nl, label='deconvolve')
plt.plot(tt, truth[index], label='ground truth signal')
plt.plot(tt, trueSpikes[index],  label='ground truth spikes')
plt.plot(tt, s_nl, label='deconvolved spikes')
plt.legend()
np.corrcoef(s_nl, trueSpikes[index])

#%% Correlation coefficients between spike train and deconvolved spike train
coef = []
coef_nl = []
for index in range(1000):
    c, s, b, g, lam = deconvolve(Y[index])
    c_nl, s_nl, b_nl, g_nl, lam_nl = deconvolve(Y_nl[index])
    coef.append(np.corrcoef(s, trueSpikes[index])[0, 1])
    coef_nl.append(np.corrcoef(s_nl, trueSpikes[index])[0, 1])
    
#%%
print(np.array(coef).mean())
print(np.array(coef_nl).mean())
    
    
    
    
    
    
    
