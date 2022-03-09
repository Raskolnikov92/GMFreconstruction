#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:18:01 2022

@author: atsouros
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:16:41 2022

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
import os
import os
import imageio
from natsort import natsorted
from scipy import fft, fftpack

def make_random_mask(domain):
    mask = ift.from_random(domain, 'pm1') #This creates a field whose values are either -1 or 1 on each pixel of the input domain
    #'pm1' stands for plus-minus 1
    mask = (mask + 1)/2 #this maps -1->0 and 1->1
    return mask.val

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 1e-2 #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N))
    for i,j in itertools.product(range(N),range(N)):
        mat[i][j] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat

# Set prior correlation covariance with a power spectrum leading to
# homogeneous and isotropic statistics
def power_spectrum(k):
    s = 4
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

def div_clean_2d(dft_mat1, dft_mat2):
    N = len(dft_mat1)
    fft_mat1_div_cleaned = np.ones([N, N], dtype=np.complex_) 
    fft_mat2_div_cleaned = np.ones([N, N], dtype=np.complex_)
    dft_mat1 = fftpack.fftshift(dft_mat1)
    dft_mat2 = fftpack.fftshift(dft_mat2)
    for i, j in itertools.product(range(N), range(N)):
        k = np.array([i - N/2, j - N/2])
        k_norm_sq = k[0]**2+k[1]**2
        inner_product = dft_mat1[i][j]*i + dft_mat2[i][j]*j
        if k_norm_sq >= 1e-10:
            fft_mat1_div_cleaned[i][j] = dft_mat1[i][j] - k[0]*inner_product/k_norm_sq
            fft_mat2_div_cleaned[i][j] = dft_mat2[i][j] - k[1]*inner_product/k_norm_sq
            fft_mat1_div_cleaned[i][j] *= 1.5
            fft_mat2_div_cleaned[i][j] *= 1.5
        else:
            fft_mat1_div_cleaned[i][j] = dft_mat1[i][j]
            fft_mat2_div_cleaned[i][j] = dft_mat2[i][j]
    fft_mat1_div_cleaned = fftpack.ifftshift(fft_mat1_div_cleaned)
    fft_mat2_div_cleaned = fftpack.ifftshift(fft_mat2_div_cleaned)
    return fft_mat1_div_cleaned, fft_mat2_div_cleaned

def plots(mat1, mat2, mat3, mat4, max_val = None, min_val = None): #Tries to do the same as plots, but with a single colorbar
    cmap = 'viridis'
    
    mats = [mat1, mat2, mat3, mat4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15,15))
    
    titles = ['Mock Signal', 
              'Data', 
              'Reconstruction', 
              'Residuals']
    
    i = 0
    for ax in axes.flat:
        if max_val == None and min_val == None:
            im = ax.imshow(mats[i], cmap=cmap, vmin = -10, vmax = +10)
        else:
            im = ax.imshow(mats[i], cmap=cmap, vmin = min_val, vmax = max_val)
        ax.set_title(titles[i], fontsize=20)
        i += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return

def main():
    N_pixels = int(5e2)
    
    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])
    
    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()
    
    # Harmonic transform from harmonic space to position space
    #HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
    
    L = 1
    x = y = np.linspace(0, L, N_pixels)
    X, Y  = np.meshgrid(x, y)
    
    given_func = False
    if given_func:

        R = np.sqrt(X**2+Y**2)
        Bx = X**2*Y
        By = -X*Y**2
    
    position_space = ift.RGSpace([N_pixels, N_pixels])
    harmonic_space = position_space.get_default_codomain()

    mask = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)

    HT1 = ift.HartleyOperator(harmonic_space, target=position_space)
    HT2 = ift.HartleyOperator(position_space, target=harmonic_space)
    
    power_space = ift.PowerSpace(harmonic_space)
    PD = ift.PowerDistributor(harmonic_space, power_space)
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
    S = ift.DiagonalOperator(prior_correlation_structure)

    R = Mask(HT1)
    
    MOCK_SIGNALx = S.draw_sample_with_dtype(dtype=np.float64)
    MOCK_SIGNALy = S.draw_sample_with_dtype(dtype=np.float64)

    Bx = MOCK_SIGNALx.val
    By = MOCK_SIGNALy.val
    

    #dft_Bx, dft_By = div_clean_2d(MOCK_SIGNALx.val, MOCK_SIGNALy.val)
    #Bx = np.real(fftpack.ifft2(dft_Bx))
    #By = np.real(fftpack.ifft2(dft_By))

    
    MOCK_SIGNALx = HT2(ift.Field.from_raw(position_space, Bx))
    MOCK_SIGNALy = HT2(ift.Field.from_raw(position_space, By))
    
    #Let's make our data noisy
    noise = np.max(MOCK_SIGNALx.val.flatten(order='C'))

    data_space = R.target
    N = ift.ScalingOperator(data_space, noise)
    MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)

    datax = R(MOCK_SIGNALx) + MOCK_NOISE 
    datay = R(MOCK_SIGNALy) + MOCK_NOISE
    
    # Build inverse propagator D and information source j
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    jx = R.adjoint_times(N.inverse_times(datax))
    jy = R.adjoint_times(N.inverse_times(datay))
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    
    # Calculate Wiener filter solution
    mx = D(jx)
    my = D(jy)
    
    #Get ndarray containing values of reconstruction in each voxel
    mat_x = HT1(mx).val 
    mat_y = HT1(my).val 
        
    #Get DFT matrix for each component 
    dft_mat_x = fftpack.fftn(mat_x)
    dft_mat_y = fftpack.fftn(mat_y)
        
    #Apply the divergence cleaning function
    dft_mat_x, dft_mat_y = div_clean_2d(dft_mat_x, dft_mat_y)
    
    #invert the dft matrix after application of the divergence cleaning
    div_cleaned_mat_x = np.real(fftpack.ifft2(dft_mat_x))
    div_cleaned_mat_y = np.real(fftpack.ifft2(dft_mat_y))
    
    ############################################
    #Main part ends here
    ############################################
    
    B_mag = np.sqrt(Bx**2 + By**2)
    B_x_dir = Bx/B_mag
    B_y_dir = By/B_mag
    
    L = 1    
    
    # Create Meshgrid
    x = np.linspace(0, L, N_pixels)
    y = np.linspace(0, L, N_pixels)
    xx, yy = np.meshgrid(x, y)

    cmap = 'viridis'
        
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))    
    

    plt.subplot(131)
    plt.contourf(X, Y, np.log10(B_mag), cmap=cmap)
    plt.colorbar()
    plt.streamplot(X, Y, B_x_dir, B_y_dir, color="black")
    plt.title('Original Field')
    plt.axis("image")
    
    Bx = mat_x
    By = mat_y
    B_mag = np.sqrt(Bx**2 + By**2)
    B_x_dir = Bx/B_mag
    B_y_dir = By/B_mag

    plt.subplot(132)
    plt.contourf(xx, yy, np.log10(B_mag), cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, B_x_dir, B_y_dir, color="black")
    plt.title('Reconstruction (no div cleaning)')
    plt.axis("image")
    
    Bx = div_cleaned_mat_x
    By = div_cleaned_mat_y
    B_mag = np.sqrt(Bx**2 + By**2)
    B_x_dir = Bx/B_mag
    B_y_dir = By/B_mag


    plt.subplot(133)
    plt.contourf(xx, yy, np.log10(B_mag), cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, B_x_dir, B_y_dir, color="black")
    plt.title('Reconstruction (div cleaned)')
    plt.axis("image")
    
if __name__ == '__main__':
    main()