#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:50:26 2022

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



def random_los(n_los): 
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends 

def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends

def make_random_mask(domain):
    mask = ift.from_random(domain, 'pm1') #This creates a field whose values are either -1 or 1 on each pixel of the input domain
    #'pm1' stands for plus-minus 1
    mask = (mask + 1)/2 #this maps -1->0 and 1->1
    return mask.val

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 1e-3 #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N,N))
    for i,j,k in itertools.product(range(N),range(N),range(N)):
        mat[i][j][k] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat

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
    cmap = 'plasma'
    
    fig, axes = plt.subplots(2, 2, figsize=(15,15))
    
    pl1 = axes[0,0].imshow(mat1, cmap=cmap)
    axes[0,0].set_title('Mock Signal', fontsize=20)
    axes[0,0].set(ylabel='Pixel number')
        
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl1, cax=cax, orientation='vertical')
    
    pl2 = axes[0,1].imshow(mat2, cmap=cmap)
    axes[0,1].set_title('Data (averaged per axis)',fontsize=20)
    
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
    
    plt.show()
    return 

def plots2(mat1, mat2, mat3, mat4, max_val, min_val): #Tries to do the same as plots, but with a single colorbar
    cmap = 'plasma'
    
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
     

#In order to input point data, perhaps check from_raw instead of from_random    

def main():    
    #mode = 0,1,2 for random LoS, radial LoS, and masked pixels respectively. Default is radial LoS
    
    #Here I define the signal space ()
    N_pixels = 100 #number of pixels in N_pixelsxN_pixels grid    
        
    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])
    
    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()
    
    # Harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

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
    
    # 1D spectral space on which the power spectrum is defined
    power_space = ift.PowerSpace(harmonic_space)
    
    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(harmonic_space, power_space)
    
    # Apply the mapping
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
    
    # Insert the result into the diagonal of an harmonic space operator
    S = ift.DiagonalOperator(prior_correlation_structure)
    #S is the prior field covariance. It is diagonal (assuming homogeneity and isotropy).
    
    #Now we build the response operator
    
    # Masking operator to model that parts of the field have not been observed
    #mask = make_random_mask(position_space)
        
    mask = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)
    
    R = Mask(HT)
    
    data_space = R.target
    
    # Create mock data
    MOCK_SIGNAL = S.draw_sample_with_dtype(dtype=np.float64)
    
    x = HT(MOCK_SIGNAL).val.flatten(order='C')
    max_val = max(x)
    min_val = min(x)  
    mean_val = np.mean(x)
    
    noise = 1*max_val #noise covariance as a fraction of 
    
    N = ift.ScalingOperator(data_space, noise)
        
    MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)
    data = R(MOCK_SIGNAL) + MOCK_NOISE
    
    # Build inverse propagator D and information source j
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    
    # Calculate Wiener filter solution
    m = D(j)
    
    make_tomographic_gif = True
    if make_tomographic_gif:
        for i in range(0,100):
            mat1 = HT(MOCK_SIGNAL).val[i]
            mat2 = Mask.adjoint(data).val[i]
            mat3 = HT(m).val[i]
            mat4 = HT(MOCK_SIGNAL).val[i] - HT(m).val[i]
            
            args = [
                mat1,
                mat2,
                mat3,
                mat4,
                max_val,
                min_val]
            
            plots2(*args)

            gif_dir = 'gif_pics'
            if not os.path.isdir(gif_dir):
                os.mkdir(gif_dir)     
            filename = 'gif_frame'
            #plt.savefig(gif_dir + '/' + filename + '%d'%i)
            filename += str(i)
            plt.savefig(os.path.join(gif_dir, filename)) #save individual frames of gif
            
            images = []
            for file_name in natsorted(os.listdir(gif_dir)):
                if file_name.endswith('.png'):
                    file_path = os.path.join(gif_dir, file_name)
                    images.append(imageio.imread(file_path))
            imageio.mimsave(os.path.join(gif_dir, 'tomography.gif'), images, fps = 7)
       
    
    for axis in range(0,3):
        mat1 = HT(MOCK_SIGNAL).val
        mat2 = Mask.adjoint(data).val
        mat3 = HT(m).val 
        mat4 = HT(MOCK_SIGNAL).val - HT(m).val
        
        mat1 = np.mean(mat1,axis=axis)
        mat2 = np.mean(mat2,axis=axis)
        mat3 = np.mean(mat3,axis=axis)
        mat4 = np.mean(mat4,axis=axis)
        
        args = [
            mat1,
            mat2,
            mat3,
            mat4,
            max_val,
            min_val]
           
        plots2(*args)

if __name__ == '__main__':
    main()