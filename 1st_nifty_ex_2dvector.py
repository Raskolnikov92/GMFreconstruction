#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:31:22 2022

@author: atsouros
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 18:59:12 2021

@author: atsouros
"""

import sys
import numpy as np
import nifty7 as ift
import random as rn
import itertools
import matplotlib.pyplot as plt

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 0.1 #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N))
    xm = []
    for i,j in itertools.product(range(N),range(N)):
        mat[i][j] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
        xm.append([i/N + 1/(2*N),j/N + 1/(2*N)])
    return mat, xm

#In order to input point data, perhaps check from_raw instead of from_random    

def main():
    #Here I define the signal space ()
    N_pixels = 200 #number of pixels in NxN grid
    position_space = ift.RGSpace([N_pixels, N_pixels])
        
    
    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()
    
    # Harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
    
    # Set prior correlation covariance with a power spectrum leading to
    # homogeneous and isotropic statistics
    def power_spectrum(k):
        return 1./(1+k**2)
    
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
    mask, xm = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)
    
    
    R = Mask(HT)
    
    data_space = R.target
    
    # Set the noise covariance N
    noise = 50.
    N = ift.ScalingOperator(data_space, noise)
    
    # Create mock data
    MOCK_SIGNALx = S.draw_sample_with_dtype(dtype=np.float64)
    MOCK_NOISEx = N.draw_sample_with_dtype(dtype=np.float64)
    datax = R(MOCK_SIGNALx) + MOCK_NOISEx
    
    MOCK_SIGNALy = S.draw_sample_with_dtype(dtype=np.float64)
    MOCK_NOISEy = N.draw_sample_with_dtype(dtype=np.float64)
    datay = R(MOCK_SIGNALy) + MOCK_NOISEy

    
    # Build inverse propagator D and information source j
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    
    
    jx = R.adjoint_times(N.inverse_times(datax))
    jy = R.adjoint_times(N.inverse_times(datay))
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    
    # Calculate WIENER FILTER solution
    mx = D(jx)
    my = D(jy)

    # Plotting
    rg = isinstance(position_space, ift.RGSpace)
    plot = ift.Plot()
    
    plot_config = True
    
    cmap = None
    
    histogram = False #True draws histogram of residuals 
    
    signalx = HT(MOCK_SIGNALx).val
    signaly = HT(MOCK_SIGNALy).val
    signal_mag = np.sqrt(signalx**2 + signaly**2)
    signal_dir_x = signalx/signal_mag
    signal_dir_y = signaly/signal_mag
    
    reconstructionx = HT(mx).val
    reconstructiony = HT(my).val
    reconstruction_mag = np.sqrt(reconstructionx**2 + reconstructiony**2)
    reconstruction_dir_x = reconstructionx/reconstruction_mag
    reconstruction_dir_y = reconstructiony/reconstruction_mag
    
    L = 1
    y, x = np.mgrid[0:L:200j, 0:L:200j]
    
    cmap = 'plasma'
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))    
    
    plt.subplot(121)
    plt.contourf(x, y, np.log10(signal_mag), cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, signal_dir_x, signal_dir_y, color="black")
    plt.title(r'$ \mathbf{B}(\mathbf{x}) $')
    plt.axis("image")


    plt.subplot(122)
    plt.contourf(x, y, np.log10(reconstruction_mag), cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, reconstruction_dir_x, reconstruction_dir_y, color="black")
    plt.title(r'$ \langle \mathbf{B}(\mathbf{x}) \rangle$')
    plt.axis("image")
    plt.show()
    
    
if __name__ == '__main__':
    main()