#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:59:03 2022

@author: atsouros
"""

from sys import exit
import numpy as np
import nifty7 as ift
import itertools
import random as rn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy.ma as ma
import time
import numpy.linalg as la
from scipy.ndimage import gaussian_filter

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

class div_cleaning(ift.LinearOperator):
    """
    Div cleaning operator operator
    ----------
    domain : real space domain on which the input field lives

    """
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

# call it in main body as O = Operator(correlated_field.target)
    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            mat = x.val
            dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
            mat_div_cleaned = np.zeros([3, N_pixels, N_pixels, N_pixels])


            for index in range(0,3):
                dft_mat[index] = fftpack.fftn(mat[index])
                dft_mat[index] = fftpack.fftshift(dft_mat[index])
                                
            k = np.mgrid[:N_pixels,:N_pixels,:N_pixels] - N_pixels/2
            k_norm_sq = np.multiply(k[0],k[0]) + np.multiply(k[1],k[1]) + np.multiply(k[2],k[2])
            inner_product = np.multiply(dft_mat[0],k[0]) + np.multiply(dft_mat[1],k[1]) + np.multiply(dft_mat[2],k[2])
            dft_mat_div_cleaned = np.where(k_norm_sq >= 1e-10, 3/2*(dft_mat - k*inner_product/k_norm_sq), dft_mat)

            for index in range(0,3):
                dft_mat_div_cleaned[index] = fftpack.ifftshift(dft_mat_div_cleaned[index])
                mat_div_cleaned[index] = np.real(fftpack.ifftn(dft_mat_div_cleaned[index]))
                
            return ift.sugar.makeField(self.target, mat_div_cleaned)
        if mode == self.ADJOINT_TIMES:
            mat = x.val
            dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
            mat_div_cleaned = np.zeros([3, N_pixels, N_pixels, N_pixels])
            
            
            for index in range(0,3):
                dft_mat[index] = fftpack.fftn(mat[index])
                dft_mat[index] = fftpack.fftshift(dft_mat[index])
                                
            k = np.mgrid[:N_pixels,:N_pixels,:N_pixels] - N_pixels/2
            k_norm_sq = np.multiply(k[0],k[0]) + np.multiply(k[1],k[1]) + np.multiply(k[2],k[2])
            inner_product = np.multiply(dft_mat[0],k[0]) + np.multiply(dft_mat[1],k[1]) + np.multiply(dft_mat[2],k[2])
            dft_mat_div_cleaned = np.where(k_norm_sq >= 1e-10, 3/2*(dft_mat - k*inner_product/k_norm_sq), dft_mat)
            
            for index in range(0,3):
                dft_mat_div_cleaned[index] = fftpack.ifftshift(dft_mat_div_cleaned[index])
                mat_div_cleaned[index] = np.real(fftpack.ifftn(dft_mat_div_cleaned[index]))
            return ift.sugar.makeField(self.target, mat_div_cleaned)


N_pixels = 50
N_data_points = 1000
frac = N_data_points/N_pixels**3

def make_perc_mask(N):
    # this function does the same as make_random_mask, but you get to chose the 
    # percentage of pixels you want to keep.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N,N))
    for i,j,k in itertools.product(range(N),range(N),range(N)):
        mat[i][j][k] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac])))
    return mat

def relative_norm_fun(mat1, mat2):
    signal_norm = la.norm(mat1.flatten(order='C')).astype(float)
    res_norm = la.norm(mat2.flatten(order='C')).astype(float)

    return 1.-res_norm/signal_norm

def gaussian_smoothing_kernel(signal_array, reconstruction_array):
    relative_norm = []
    sigma_array = np.linspace(0, 100, num = 50)
    for sigma in sigma_array:
        smoothed_array = gaussian_filter(signal_array, sigma = sigma)
        val = relative_norm_fun(smoothed_array, smoothed_array-reconstruction_array)
        relative_norm.append(val)
    max_value = np.max(relative_norm)
    max_index = relative_norm.index(max_value)
    return max_value, sigma_array[max_index]

def plots(mat1, mat2, mat3, mat4, relative_norm = None, sigma = None, noise = None):

    #max_val = np.max(arr)
    #min_val = np.min(arr)

    cmap = 'viridis'

    mats = [mat1, mat2, mat3, mat4]

    fig, axes = plt.subplots(2, 2, figsize=(15,15))

    if sigma is None:
        titles = ['Mock Signal',
                  'Data',
                  'Reconstruction',
                  'Absolute Relative Error']
    else:
        titles = [r'Signal smoothed with $\sigma = %.2f$'%sigma,
                  'Data',
                  'Reconstruction',
                  'Absolute Relative Error']


    i = 0
    for ax in axes.flat:
        im = ax.imshow(mats[i], cmap=cmap)
        ax.set_title(titles[i], fontsize=20)
        i += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # if relative_norm is not None:
    #    fig.suptitle('Relative Norm: %.2f, Fraction of voxels kept: %.1f %%, noise covariance = %.2f'%(relative_norm, frac*100., noise), fontsize = 20)
    fig.show()
    return

def main():

    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])

    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.
    args = {
        # Amplitude of field fluctuations
        'fluctuations': (1e0, 1e0),

        # Exponent of power law power spectrum component. Choose a Kolmogorov slope 
        # as a mean, and allow for a ± 1 std
        'loglogavgslope': (-11/3, 1e0),

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1e-16,1e0),

        # How ragged the integrated Wiener process component is
        'asperity': None
    }

    cfmaker = ift.CorrelatedFieldMaker(prefix='', total_N=3)
    cfmaker.add_fluctuations(position_space, **args)
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-6))
    pspec = cfmaker.power_spectrum
    signal = cfmaker.finalize()
    
    
    # Change model parameters for each component using dofdex

    #Define div cleaning operator.
    DC = div_cleaning(signal.target)

    # Define response operator
    mask = make_perc_mask(N_pixels)
    mask = np.array([mask, mask, mask])
    mask = ift.Field.from_raw(signal.target, mask)
    # mask = ift.Field.from_raw(position_space, mask)
    R = ift.MaskOperator(mask)

    signal_response = R(DC(signal))

    mock_position = ift.from_random(signal_response.domain, 'normal')


    mat = signal(mock_position).val    
    mock_signal = ift.Field.from_raw(signal.target, mat)
    DCmock_signal = DC(mock_signal)
    
    # Specify noise
    data_space = R.target
    noise = np.max(DCmock_signal.val)
    N = ift.ScalingOperator(data_space, noise)
    
    data = R(DCmock_signal) + N.draw_sample_with_dtype(dtype=np.float64)

    # Notice that R(signal) & DC(signal) are of _OpChain type
    # While R(mock_signal) & DC(mock_signal) are of Field type

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
            deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
            convergence_level=2, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
            deltaE=0.5, iteration_limit=15, convergence_level=2)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                         R(DC(signal)))
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    # number of samples used to estimate the KL
    N_samples = 6

    # Draw new samples to approximate the KL six times
    for i in range(6):
        if i==5:
            # Double the number of samples in the last step for better statistics
            N_samples = 2*N_samples
        # Draw new samples and minimize KL
        KL = ift.GeoMetricKL(mean, H, N_samples, minimizer_sampling, True)
        KL, convergence = minimizer(KL)
        mean = KL.position
        ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                             KL.position, KL.samples)

    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    ##########MAIN PART ENDS HERE#####################
    
    powers = [pspec.force(s + KL.position).val[0] for s in KL.samples]
    k = range(len(pspec.force(mock_position).val[0]))
    k = [2*np.pi*i/64 for i in k]
        
    for i in range(len(powers)):
        plt.loglog(k, powers[int(i)],color='gray',alpha=0.3)
    plt.loglog(k, pspec.force(mock_position).val[0], color='blue', label = 'Ground truth')
    plt.loglog(k, pspec.force(KL.position).val[0],color='magenta', label = 'Posterior mean')
    #plt.loglog(k, sc.mean.exp().val[index], color='red', label = 'Posterior mean')
    plt.legend()
    plt.ylabel(r'$P(k)$', fontsize = 15)
    plt.xlabel(r'$k$', fontsize = 15)
    plt.show()
    
    axis = 0
    #plot results
    for index in range(0,3):
        signal_arr = DCmock_signal.val[index]
        data_arr = R.adjoint_times(data).val[index]
        data_arr = ma.masked_values(data_arr, 0.)

        reconstruction_arr = sc.mean.val[index]
        residual_arr = np.sqrt((signal_arr - reconstruction_arr)**2)
        # std_arr = ift.sqrt(sc.var).val[index]


        signal_norm = la.norm(signal_arr.flatten(order='C')).astype(float)
        res_norm = la.norm(residual_arr.flatten(order='C')).astype(float)

        ratio_norm = 1.-res_norm/signal_norm

        print('Relative norm is %.2f'%ratio_norm)

        signal_arr_PoS = np.mean(signal_arr,axis=axis)
        data_arr_PoS = np.mean(data_arr,axis=axis)
        data_arr_PoS = ma.masked_values(data_arr_PoS, 0.)

        reconstruction_arr_PoS = np.mean(reconstruction_arr,axis=axis)
        residual_arr_PoS = np.mean(residual_arr,axis=axis)
        # std_arr_PoS = np.mean(std_arr,axis=axis)


        args = [signal_arr_PoS,
                data_arr_PoS,
                reconstruction_arr_PoS,
                residual_arr_PoS]

        kwargs = {
            'relative_norm' : ratio_norm,
            'noise' : noise
        }

        plots(*args, **kwargs)

        max_rel_norm, sigma = gaussian_smoothing_kernel(signal_arr, reconstruction_arr)
        print('Maximum relative norm after smoothing: %.2f'%max_rel_norm)
        print('Maximum norm lengthscale: %.2f pixels'%(2*np.pi*sigma))


        sig_smoothed_PoS = gaussian_filter(signal_arr, sigma = sigma)
        # rec_PoS_smoothed = gaussian_filter(reconstruction_arr, sigma = sigma)
        
        error_arr_smoothed = np.sqrt((sig_smoothed_PoS-reconstruction_arr)**2)

        axis = 0
        sig_PoS_smoothed = np.mean(sig_smoothed_PoS, axis = axis)
        # rec_PoS_smoothed = np.mean(rec_PoS_smoothed, axis = axis)
        error_arr_smoothed_PoS = np.mean(error_arr_smoothed, axis = axis)

        kwargs = {'relative_norm' : max_rel_norm,
                  'noise' : noise,
                  'sigma' : sigma
        }

        args = [sig_PoS_smoothed,
                data_arr_PoS,
                reconstruction_arr_PoS,
                error_arr_smoothed_PoS]
        
        plots(*args, **kwargs)
        plt.show()
        
        print('Total number of data points: %d'%N_data_points)

        

if __name__ == '__main__':
    t = time.time()
    main()
    t = time.time() - t
    t *= 1/60 #time in minutes
    print('Total calculation time: %.2f minutes'%t)