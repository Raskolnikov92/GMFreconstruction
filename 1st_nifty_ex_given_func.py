#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:04:14 2022

@author: atsouros
"""
import sys
import numpy as np
import nifty7 as ift
import random as rn
import itertools
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 0.01  #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N))
    for i,j in itertools.product(range(N),range(N)):
        mat[i][j] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def rms(arr):
    """
    Returns RMS of array elements.
    """
    return np.sqrt(np.mean(np.square(arr)))

def divisors(n):
    """
    Returns an array of all the divisors of the integer n.
    """
    if not(isinstance(n,int)):
        raise ValueError('The input of the divisors function must be an integer.')
        
    ret = []
    for i in range(1,n-1):
        if (n % (i+1)) == 0:
            ret.append(i+1)
    if not ret:
        raise ValueError('Number of pixels per axis is prime. Choose a more composite number.')
    else:
        return np.array(ret)
    
def plots(mat1, mat2, mat3, mat4):
    cmap = 'viridis'
    
    fig, axes = plt.subplots(2, 2, figsize=(15,15))
    
    pl1 = axes[0,0].imshow(mat1, cmap=cmap)
    axes[0,0].set_title('Mock Signal', fontsize=20)
    axes[0,0].set(ylabel='Pixel number')
        
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl1, cax=cax, orientation='vertical')
    
    pl2 = axes[0,1].imshow(mat2, cmap=cmap)
    axes[0,1].set_title('Data',fontsize=20)
    
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl2, cax=cax, orientation='vertical')
    
    
    pl3 = axes[1,0].imshow(mat3, cmap=cmap)
    axes[1,0].set_title('Wiener filter reconstruction',fontsize=20)
    axes[1,0].set(ylabel='Pixel number', xlabel='Pixel number')
    
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl3, cax=cax, orientation='vertical')
    
    pl4 = axes[1,1].imshow(mat4, cmap=cmap)
    axes[1,1].set_title('Residuals',fontsize=20)
    axes[1,1].set(xlabel='Pixel number')
    
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl4, cax=cax, orientation='vertical')
    filename = 'nifty_results_given_func.png'
    plt.savefig(filename)
    plt.show()
    
    return  

N_pixels = 360

x = y =  np.linspace(0, 1, N_pixels)
X, Y = np.meshgrid(x, y)
Z = np.exp(-((X-0.5)**2+(Y-0.5)**2)/0.01)

position_space = ift.RGSpace([N_pixels, N_pixels])
harmonic_space = position_space.get_default_codomain()

mask = make_perc_mask(N_pixels)
mask = ift.Field.from_raw(position_space, mask)
Mask = ift.MaskOperator(mask)

HT1 = ift.HartleyOperator(harmonic_space, target=position_space)
HT2 = ift.HartleyOperator(position_space, target=harmonic_space)

def power_spectrum(k):
        s = 7
        output = []
        kc = 1
        c = kc**(-s)
        if isinstance(k,np.ndarray):
            for ki in k:
                if ki<= kc:
                    output.append(c)
                else:
                    output.append(ki**(-s))
            output = np.asarray(output)
        else:
            if k<=kc:
                return c
            else:
                return k**(-s)       
        return output

power_space = ift.PowerSpace(harmonic_space)
PD = ift.PowerDistributor(harmonic_space, power_space)
prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
S = ift.DiagonalOperator(prior_correlation_structure)

R = Mask(HT1)

MOCK_SIGNAL = HT2(ift.Field.from_raw(position_space, Z))

#Let's make our data noisy uwu

noise = 0.05 #noise covariance
data_space = R.target
N = ift.ScalingOperator(data_space, noise)
MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)



data = R(MOCK_SIGNAL) + MOCK_NOISE

# Build inverse propagator D and information source j
D_inv = R.adjoint @ R + S.inverse
j = R.adjoint_times(data)
# Make D_inv invertible (via Conjugate Gradient)
IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse

# Calculate Wiener filter solution
m = D(j)

#plot

average_pixels = False #True if you want to plot the result of sequentially averaging pixels and comparing results. Also shows change of RMS(Residual)/RMS(signal) wrt to length scale

if average_pixels:    
    mat1 = HT1(MOCK_SIGNAL).val
    mat2 = Mask.adjoint(data).val
    mat3 = HT1(m).val
    mat4 = HT1(MOCK_SIGNAL).val - HT1(m).val

    N = N_pixels
    x = [1/N]
    y = [rms(mat1.flatten(order='C'))/rms(mat2.flatten(order='C'))]  
    plots(mat1, mat2, mat3, mat4)
    
    for i in divisors(N):
        x.append(i/N)
        reduced_mat1 = np.zeros((N//i,N//i))
        reduced_mat2 = np.zeros((N//i,N//i))
        reduced_mat3 = np.zeros((N//i,N//i))
        reduced_mat4 = np.zeros((N//i,N//i))
        s = 0
        for j, k in itertools.product(range(N//i), range(N//i)):
            reduced_mat1[j][k] = np.mean(blockshaped(mat1, i, i)[s].flatten(order='C'))
            reduced_mat2[j][k] = np.mean(blockshaped(mat2, i, i)[s].flatten(order='C'))
            reduced_mat3[j][k] = np.mean(blockshaped(mat3, i, i)[s].flatten(order='C'))
            reduced_mat4[j][k] = np.mean(blockshaped(mat4, i, i)[s].flatten(order='C'))
            s += 1
        y.append(rms(reduced_mat1.flatten(order='C'))/rms(reduced_mat2.flatten(order='C')))
        plots(reduced_mat1, reduced_mat2, reduced_mat3, reduced_mat4)


plot = ift.Plot()
cmap = None
filename = "nifty_test_given_func.png"

plot.add(HT1(MOCK_SIGNAL), title='Mock Signal', cmap = cmap)
plot.add(Mask.adjoint(data), title='Data', cmap = cmap)
plot.add(HT1(m), title='Reconstruction', cmap = cmap)
plot.add(HT1(MOCK_SIGNAL) - HT1(m), title='Residuals', cmap = cmap) #x = RMS of residuals/RMS of map (single number) -> smoothing  at different length scales, and   
plot.output(nx=2, ny=2, xsize=10, ysize=10, name=filename)
print("Saved results as '{}'.".format(filename))     