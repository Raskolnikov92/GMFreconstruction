import sys
import numpy as np
import nifty7 as ift
import random as rn
import itertools
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import imageio
from natsort import natsorted
from scipy import fft, fftpack
import numpy.linalg as linalg
import time
import numpy.ma as ma

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 1e-2 #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N,N))
    for i,j,k in itertools.product(range(N),range(N),range(N)):
        mat[i][j][k] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat


#Function that divergence cleans ndarrays. This is the 3d version
def div_clean(dft_mat):
    print('div cleaning procedure starts. May take a while...')
    t = time.time()
    
    N = len(dft_mat[0])
    fft_mat_div_cleaned = np.ones([3, N, N, N], dtype=np.complex_)
    dft_mat = np.array([fftpack.fftshift(dft_mat[0]), fftpack.fftshift(dft_mat[1]), fftpack.fftshift(dft_mat[2])])
    for ii, jj, kk in itertools.product(range(N), range(N), range(N)):
        k = np.array([ii - N/2, jj - N/2,  kk - N/2])
        k_norm_sq = linalg.norm(k)**2
        inner_product = dft_mat[0][ii][jj][kk]*ii + dft_mat[1][ii][jj][kk]*jj + dft_mat[2][ii][jj][kk]*kk
        if k_norm_sq >= 1e-10:
            c = 1 #rescaling factor
            for index in range(0,3):
                fft_mat_div_cleaned[index][ii][jj][kk] = c*(dft_mat[index][ii][jj][kk]  - k[index]*inner_product/k_norm_sq)       
        else:
            for index in range(0,3):
                fft_mat_div_cleaned[index][ii][jj][kk] = dft_mat[index][ii][jj][kk]
    for index in range(0,3):
        fft_mat_div_cleaned[index] = fftpack.ifftshift(fft_mat_div_cleaned[index])
    t = time.time() - t
    print('div cleaning took %.2f seconds' %t)
    return fft_mat_div_cleaned

def plots(mat1, mat2, mat3, mat4): #Tries to do the same as plots, but with a single colorbar
    arr1 = mat1.flatten(order='C')
    arr2 = mat3.flatten(order='C')
    
    arr = arr1 + arr2
    
    max_val = np.max(arr)
    min_val = np.min(arr)    
    
    cmap = 'viridis'
        
    mats = [mat1, mat2, mat3, mat4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15,15))
    
    titles = ['Mock Signal', 
              'Data', 
              'Reconstruction', 
              'Residuals']
    
    i = 0
    for ax in axes.flat:
        im = ax.imshow(mats[i], cmap=cmap, vmin = min_val, vmax = max_val)
        ax.set_title(titles[i], fontsize=20)
        i += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return

def power_spectrum(k):
    s = 4
    output = []
    kc = 1
    c = kc**(-s)
    for ki in k:
        if ki<= kc:
            output.append(c)
        else:
            output.append(ki**(-s))
    output = np.asarray(output)      
    return output


def main():   
    # Number of pixels in N_pixelsxN_pixels grid   
    N_pixels = 50
    
    # Define the configuration space. Here a square grid.   
    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])
            
    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()
    
    # Harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
    # Same (and vice versa) for fields defined from given field values
    HT1 = ift.HartleyOperator(harmonic_space, target=position_space)
    HT2 = ift.HartleyOperator(position_space, target=harmonic_space)

    # 1D spectral space on which the power spectrum is defined
    power_space = ift.PowerSpace(harmonic_space)
    
    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(harmonic_space, power_space)
        
    # Apply the mapping
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
    
    # Insert the result into the diagonal of an harmonic space operator
    S = ift.DiagonalOperator(prior_correlation_structure)

    # Create two Gaussian random fields that have the covariance S specified above
    mock_signal = np.array([S.draw_sample_with_dtype(dtype=np.float64), 
                            S.draw_sample_with_dtype(dtype=np.float64), 
                            S.draw_sample_with_dtype(dtype=np.float64)])
    
    # Extract the GRF field values as ndarrays
    mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    div_cleaned_mat = np.zeros([3, N_pixels, N_pixels, N_pixels])
    
    for index in range(0,3):
        mat[index] = HT(mock_signal[index]).val  
        dft_mat[index] = fftpack.fftn(mat[index])
    
    #divergence clean
    dft_mat = div_clean(dft_mat)
    
    for index in range(0,3):
        div_cleaned_mat[index] = np.real(fftpack.ifftn(dft_mat[index]))
        mock_signal[index] = HT2(ift.Field.from_raw(position_space, div_cleaned_mat[index]))

    # Define response operator
    mask_mat = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask_mat)
    Mask = ift.MaskOperator(mask)
    R = Mask(HT1)
    
    # Define noise covariance
    noise = np.max(mock_signal[0].val.flatten(order='C'))   
    data_space = R.target
    N = ift.ScalingOperator(data_space, noise)
    mock_noise = N.draw_sample_with_dtype(dtype=np.float64)
    
     # Build inverse propagator D and information source j
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    
    data = []
    j = []
    m = []
    for index in range(0,3):
        data.append(R(mock_signal[index]) + mock_noise)
        #Define the source
        j.append(R.adjoint_times(N.inverse_times(data[index])))
        # Calculate Wiener filter solution
        m.append(D(j[index]))
     
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    

    # Initialize matrices
    rec_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    dft_rec_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    div_cleaned_rec_mat = np.zeros([3, N_pixels, N_pixels, N_pixels])
    
    for index in range(0,3):
        rec_mat[index] = HT(m[index]).val  
        dft_rec_mat[index] = fftpack.fftn(rec_mat[index])
    
    #divergence clean the reconstruction
    dft_rec_mat = div_clean(dft_rec_mat)
    
    for index in range(0,3):
        div_cleaned_rec_mat[index] = np.real(fftpack.ifftn(dft_rec_mat[index]))
        
    
    #######################
    #Main part ends here. Below we plot results. 
    #######################

    
    axis = 0
    
    for index in range(0,3):
        mat1 = HT(mock_signal[index]).val
        mat2 = Mask.adjoint(data[index]).val
        mat2 = ma.masked_values(mat2, 0.)
        
        mat3 = div_cleaned_rec_mat[index]
        mat4 = mat1 - mat3
        
        mat1 = np.mean(mat1,axis=axis)
        mat2 = np.mean(mat2,axis=axis)
        mat2 = ma.masked_values(mat2, 0.)
        
        mat3 = np.mean(mat3,axis=axis)
        mat4 = np.mean(mat4,axis=axis)
        
        args = [mat1,
                mat2,
                mat3,
                mat4]
        
           
        plots(*args)


                  
if __name__ == '__main__':
    t = time.time()
    main() 
    t = time.time() - t
    print('Total running time: %.2f seconds' %t)
    

