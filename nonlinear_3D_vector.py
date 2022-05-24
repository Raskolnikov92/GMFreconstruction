import sys
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


N_pixels = 50
N_data_points = 100
frac = N_data_points/N_pixels**3
#frac = 1e-2

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

    max_val = np.max(arr)
    min_val = np.min(arr)

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
        im = ax.imshow(mats[i], cmap=cmap, vmin = min_val, vmax = max_val)
        ax.set_title(titles[i], fontsize=20)
        i += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    #if relative_norm is not None:
    #    fig.suptitle('Relative Norm: %.2f, Fraction of voxels kept: %.1f %%, noise covariance = %.2f'%(relative_norm, frac*100., noise), fontsize = 20)
    fig.show()
    return


def relative_norm_fun(mat1, mat2):
    signal_norm = la.norm(mat1.flatten(order='C')).astype(float)
    res_norm = la.norm(mat2.flatten(order='C')).astype(float)

    return 1.-res_norm/signal_norm

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
            c = 3/2 #rescaling factor
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

def gaussian_smoothing_kernel(signal_array, reconstruction_array):
    relative_norm = []
    sigma_array = np.linspace(0, 100, num = 50)
    for sigma in sigma_array:
        smoothed_array1 = gaussian_filter(signal_array, sigma = sigma)
        smoothed_array2 = gaussian_filter(reconstruction_array, sigma = sigma)
        val = relative_norm_fun(smoothed_array1, smoothed_array1-smoothed_array2)
        relative_norm.append(val)
    max_value = np.max(relative_norm)
    max_index = relative_norm.index(max_value)
    return max_value, sigma_array[max_index]

def gaussian_smoothing_kernel2(signal_array, reconstruction_array):
    relative_norm = []
    sigma_array = np.linspace(0, 100, num = 50)
    for sigma in sigma_array:
        smoothed_array1 = gaussian_filter(signal_array, sigma = sigma)
        val = relative_norm_fun(smoothed_array1, smoothed_array1-reconstruction_array)
        relative_norm.append(val)
    max_value = np.max(relative_norm)
    max_index = relative_norm.index(max_value)
    return max_value, sigma_array[max_index]


def main():

    # N_pixels = 100
    slope = -4.

    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])

    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.
    args = {
        'offset_mean': 0,
        'offset_std': (1e-5, 1e-16),

        # Amplitude of field fluctuations
        'fluctuations': (1e0, 1e-16),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (slope, 1e-16),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1e-3, 1e-16),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (1e-3, 1e-16)  # 0.1, 0.5
    }

    correlated_field = ift.SimpleCorrelatedField(position_space, **args)

    pspec = correlated_field.power_spectrum

    signal = [correlated_field,
              correlated_field,
              correlated_field
              ]

    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()


    # Define response operator
    #mask = make_random_mask(position_space)
    mask = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)
    R = Mask

    signal_response = [R(signal[0]),
                       R(signal[1]),
                       R(signal[2])
                       ]

    # Generate mock signal and data
    mock_position = [ift.from_random(signal_response[0].domain, 'normal'),
                     ift.from_random(signal_response[1].domain, 'normal'),
                     ift.from_random(signal_response[2].domain, 'normal')
                     ]

    # Specify noise
    data_space = R.target
    noise = 1.
    # noise = np.max([signal[0](mock_position[0]).val, signal[1](mock_position[1]).val, signal[2](mock_position[2]).val])
    N = ift.ScalingOperator(data_space, noise)

    """
    HOW TO DIV CLEAN
    Here we can define that matrix of values of the field as
    mat = signal(mock_position).val (for each component)
    div clean
    and then define data from the resulting fields which will be made from_raw
    """
    signal_mat = [signal[0](mock_position[0]).val,
                           signal[1](mock_position[1]).val,
                           signal[2](mock_position[2]).val]

    #Since mat here is an ndarray with shape (N_pixels,N_pixels,N_pixels) then it
    #can also be used to import raw data.
    mock_signal = [ift.Field.from_raw(position_space, signal_mat[0]),
                   ift.Field.from_raw(position_space, signal_mat[1]),
                   ift.Field.from_raw(position_space, signal_mat[2])
                   ]
    #########signal div cleaning procedure starts here#########
    # Extract the GRF field values as ndarrays
    mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    div_cleaned_mat = np.zeros([3, N_pixels, N_pixels, N_pixels])

    for index in range(0,3):
        mat[index] = mock_signal[index].val
        dft_mat[index] = fftpack.fftn(mat[index])

    #divergence clean
    dft_mat = div_clean(dft_mat)

    mock_signal_DC = []

    for index in range(0,3):
        div_cleaned_mat[index] = np.real(fftpack.ifftn(dft_mat[index]))
        #mock_signal[index] = ift.Field.from_raw(position_space, div_cleaned_mat[index])
        mock_signal_DC.append(ift.Field.from_raw(position_space, div_cleaned_mat[index]))

    #########signal div cleaning procedure ends here#########

    #data = signal_response(mock_position) + N.draw_sample_with_dtype(dtype=np.float64)
    data = [R(mock_signal_DC[0]) + N.draw_sample_with_dtype(dtype=np.float64),
            R(mock_signal_DC[1]) + N.draw_sample_with_dtype(dtype=np.float64),
            R(mock_signal_DC[2]) + N.draw_sample_with_dtype(dtype=np.float64)
            ]

    """
    #IMPORTING RAW DATA

    #if the data is given as dat_mat (ndarray with shape (Npixels, Npixels, Npixels)),
    #and you know the positions of the voxels with data, (so you know R) then
    #the following works
    dat_mat = R.adjoint_times(data).val
    data = ift.Field.from_raw(position_space, dat_mat)
    data = R(data)
    """

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
            deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
            convergence_level=2, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
            deltaE=0.5, iteration_limit=15, convergence_level=2)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    latent_mean = []
    for index in range(0,3):
        # Set up likelihood energy and information Hamiltonian
        likelihood_energy = (ift.GaussianEnergy(mean=data[index], inverse_covariance=N.inverse) @
                             signal_response[index])
        H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

        mean = ift.MultiField.full(H.domain, 0.)

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
            ift.extra.minisanity(data[index], lambda x: N.inverse, signal_response[index],
                                 KL.position, KL.samples)

        sc = ift.StatCalculator()
        for sample in KL.samples:
            sc.add(signal[index](sample + KL.position))
        latent_mean.append(sc.mean.val)


    #########signal div cleaning procedure starts here#########
    # Extract the GRF field values as ndarrays
    mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    dft_mat = np.zeros([3, N_pixels, N_pixels, N_pixels], dtype=np.complex_)
    latent_mean_mat_DC = np.zeros([3, N_pixels, N_pixels, N_pixels])

    for index in range(0,3):
        mat[index] = latent_mean[index]
        dft_mat[index] = fftpack.fftn(mat[index])

    #divergence clean
    dft_mat = div_clean(dft_mat)

    for index in range(0,3):
        latent_mean_mat_DC[index] = np.real(fftpack.ifftn(dft_mat[index]))
        #latent_mean_DC.append(ift.Field.from_raw(position_space, div_cleaned_mat[index]))

    #########signal div cleaning procedure ends here#########

    ##########MAIN PART ENDS HERE#####################
    axis = 0
    #plot results
    for index in range(0,3):
        signal_arr = div_cleaned_mat[index]
        data_arr = R.adjoint_times(data[index]).val
        data_arr = ma.masked_values(data_arr, 0.)

        reconstruction_arr = latent_mean_mat_DC[index]
        residual_arr = signal_arr - reconstruction_arr

        signal_norm = la.norm(signal_arr.flatten(order='C')).astype(float)
        res_norm = la.norm(residual_arr.flatten(order='C')).astype(float)

        ratio_norm = 1.-res_norm/signal_norm

        print('Relative norm is %.2f'%ratio_norm)

        signal_arr_PoS = np.mean(signal_arr,axis=axis)
        data_arr_PoS = np.mean(data_arr,axis=axis)
        data_arr_PoS = ma.masked_values(data_arr_PoS, 0.)

        reconstruction_arr_PoS = np.mean(reconstruction_arr,axis=axis)
        residual_arr_PoS = np.mean(residual_arr,axis=axis)


        args = [signal_arr_PoS,
                data_arr_PoS,
                reconstruction_arr_PoS,
                residual_arr_PoS]

        kwargs = {
            'relative_norm' : ratio_norm,
            'noise' : noise
        }

        plots(*args, **kwargs)

        max_rel_norm, sigma = gaussian_smoothing_kernel2(signal_arr, reconstruction_arr)
        print('Maximum relative norm after smoothing: %.2f'%max_rel_norm)
        print('Maximum norm lengthscale: %.2f pixels'%(2*np.pi*sigma))


        sig_PoS_smoothed = gaussian_filter(signal_arr, sigma = sigma)
        rec_PoS_smoothed = gaussian_filter(reconstruction_arr, sigma = sigma)

        axis = 0
        sig_PoS_smoothed = np.mean(sig_PoS_smoothed, axis = axis)
        rec_PoS_smoothed = np.mean(rec_PoS_smoothed, axis = axis)

        kwargs = {'relative_norm' : max_rel_norm,
                  'noise' : noise,
                  'sigma' : sigma
        }

        args = [sig_PoS_smoothed,
                data_arr_PoS,
                rec_PoS_smoothed,
                sig_PoS_smoothed-rec_PoS_smoothed]

        plots(*args, **kwargs)


if __name__ == '__main__':
    main()
