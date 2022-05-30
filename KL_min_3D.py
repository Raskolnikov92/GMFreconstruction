port sys
import numpy as np
import nifty7 as ift
import sys
import itertools
import random as rn
import matplotlib.pyplot as plt
from scipy.linalg import dft
from scipy import fft, fftpack
import numpy.ma as ma
import numpy.linalg as la
#ift.random.push_sseq_from_seed(27)
from scipy.ndimage import gaussian_filter
import numpy.linalg as linalg
import time

class identity(ift.LinearOperator):
    """
    Identity operator
    ----------
    domain : real space domain on which the input field lives

    """
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

# call it in main body as H = harmonicSpaceDeriv(correlated_field.target)
    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            mat = x.val
            return ift.sugar.makeField(self.target, mat)
        if mode == self.ADJOINT_TIMES:
            mat = x.val
            return ift.sugar.makeField(self.domain, mat)


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

# call it in main body as H = harmonicSpaceDeriv(correlated_field.target)
    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            mat = x.val
            dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
            div_cleaned_mat = np.zeros([3, N_pixels, N_pixels, N_pixels])

            for index in range(0,3):
                dft_mat[index] = fftpack.fftn(mat[index])
                
            k = np.mgrid[:N_pixels,:N_pixels,:N_pixels] - N_pixels/2
            k_norm_sq = np.multiply(k[0],k[0]) + np.multiply(k[1],k[1]) + + np.multiply(k[2],k[2])
            inner_product = np.multiply(dft_mat[0],k[0]) + np.multiply(dft_mat[1],k[1]) + np.multiply(dft_mat[2],k[2])
            fft_mat_div_cleaned = np.where(k_norm_sq >= 1e-10, 3/2*(dft_mat - k*inner_product/k_norm_sq), dft_mat)

            for index in range(0,3):
                fft_mat_div_cleaned[index] = fftpack.ifftshift(fft_mat_div_cleaned[index])

            for index in range(0,3):
                div_cleaned_mat[index] = np.real(fftpack.ifftn(dft_mat[index]))

            return ift.sugar.makeField(self.target, div_cleaned_mat)
        if mode == self.ADJOINT_TIMES:
            mat = x.val
            dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
            div_cleaned_mat = np.zeros([3, N_pixels, N_pixels, N_pixels])

            for index in range(0,3):
                dft_mat[index] = fftpack.fftn(mat[index])
                
            k = np.mgrid[:N_pixels,:N_pixels,:N_pixels] - N_pixels/2
            k_norm_sq = np.multiply(k[0],k[0]) + np.multiply(k[1],k[1]) + + np.multiply(k[2],k[2])
            inner_product = np.multiply(dft_mat[0],k[0]) + np.multiply(dft_mat[1],k[1]) + np.multiply(dft_mat[2],k[2])
            fft_mat_div_cleaned = np.where(k_norm_sq >= 1e-10, 3/2*(dft_mat - k*inner_product/k_norm_sq), dft_mat)

            for index in range(0,3):
                fft_mat_div_cleaned[index] = fftpack.ifftshift(fft_mat_div_cleaned[index])

            for index in range(0,3):
                div_cleaned_mat[index] = np.real(fftpack.ifftn(dft_mat[index]))


            return ift.sugar.makeField(self.target, div_cleaned_mat)


N_pixels = 64
frac = 0.1
def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    #frac = 1e-0 #percentage of pixels kept
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N,N))
    for i,j,k in itertools.product(range(N),range(N),range(N)):
        mat[i][j][k] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac])))
    return mat

def plots(mat1, mat2, mat3, mat4, relative_norm = None, sigma = None, noise = None):
    arr1 = mat1.flatten(order='C')
    arr2 = mat3.flatten(order='C')

    arr = arr1 + arr2

    #max_val = np.max(arr)
    #min_val = np.min(arr)

    cmap = 'viridis'

    mats = [mat1, mat2, mat3, mat4]

    fig, axes = plt.subplots(2, 2, figsize=(15,15))

    if sigma is None:
        titles = ['Mock Signal',
                  'Data',
                  'Reconstruction',
                  'Residuals']
    else:
        titles = ['Signal smoothed with sigma = %.2f'%sigma,
                  'Data',
                  'Reconstruction',
                  'Residuals']


    i = 0
    for ax in axes.flat:
        im = ax.imshow(mats[i], cmap=cmap)
        ax.set_title(titles[i], fontsize=20)
        i += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if relative_norm is not None:
        fig.suptitle('Relative Norm: %.2f, Fraction of voxels kept: %.1f %%, noise covariance = %.2f'%(relative_norm, frac*100., noise), fontsize = 20)
    fig.show()
    return

def main():

    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])

    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.
    args = {
        # Amplitude of field fluctuations
        'fluctuations': (1e0, 1e-2),

        # Exponent of power law power spectrum component
        'loglogavgslope': (-11/3, 1e0),

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': None,

        # How ragged the integrated Wiener process component is
        'asperity': None
    }

    cfmaker = ift.CorrelatedFieldMaker(prefix='', total_N=3)
    cfmaker.add_fluctuations(position_space, **args)
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-6))
    signal = cfmaker.finalize()

    # Change model parameters for each component using dofdex

    #Define unity operator. Eventually this will be the div cleaning one
    Id = identity(signal.target)

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

    # Specify noise
    data_space = R.target
    # noise = np.max(signal(mock_position).val)
    noise = 1e0
    N = ift.ScalingOperator(data_space, noise)

    mat = signal(mock_position).val
    mock_signal = ift.Field.from_raw(signal.target, mat)
    data = R(DC(mock_signal)) + N.draw_sample_with_dtype(dtype=np.float64)

    # Notice that R(signal) & Id(signal) are of _OpChain type
    # While R(mock_signal) & Id(mock_signal) are of Field type

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
                         signal_response)
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

    #plot results
    axis = 0

    for index in range(0,3):
        signal_arr = signal(mock_position).val[index]
        data_arr = R.adjoint_times(data).val[index]
        data_arr = ma.masked_values(data_arr, 0.)

        reconstruction_arr = sc.mean.val[index]
        residual_arr = signal_arr - reconstruction_arr

        signal_arr_PoS = np.mean(signal_arr,axis=axis)
        data_arr_PoS = np.mean(data_arr,axis=axis)
        data_arr_PoS = ma.masked_values(data_arr_PoS, 0.)

        reconstruction_arr_PoS = np.mean(reconstruction_arr,axis=axis)
        residual_arr_PoS = np.mean(residual_arr,axis=axis)


        args = [signal_arr_PoS,
                data_arr_PoS,
                reconstruction_arr_PoS,
                residual_arr_PoS]

        plots(*args)

if __name__ == '__main__':
    main()
