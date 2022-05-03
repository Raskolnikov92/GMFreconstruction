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
#ift.random.push_sseq_from_seed(27)

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 1e-2 #percentage of pixels kept
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N,N))
    for i,j,k in itertools.product(range(N),range(N),range(N)):
        mat[i][j][k] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat

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


def main():
    
    N_pixels = 50
            
    filename = "nonlinear_div_cleaned_3D.png"
    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])
    
    #  For a detailed showcase of the effects the parameters
    #  of the CorrelatedField model have on the generated fields,
    #  see 'getting_started_4_CorrelatedFields.ipynb'.
    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (1., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.4)  # 0.1, 0.5
    }

    correlated_field = ift.SimpleCorrelatedField(position_space, **args)
    
    pspec = correlated_field.power_spectrum

    # Apply a nonlinearity
    #signal = ift.sigmoid(correlated_field)
    signal = correlated_field
    

    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()
    
    # Harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
    #HT1 = ift.HartleyOperator(harmonic_space, target=position_space)
    HT2 = ift.HartleyOperator(position_space, target=harmonic_space)
     
    # Define response operator
    #mask = make_random_mask(position_space)
    mask = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)
    R = Mask
    signal_response = R(signal)
        
    # Specify noise
    data_space = R.target
    noise = .1
    N = ift.ScalingOperator(data_space, noise)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')   
    
    """
    HOW TO DIV CLEAN
    Here we can define that matrix of values of the field as
    mat = signal(mock_position).val (for each component)
    div clean
    and then define data from the resulting fields which will be made from_raw 
    """
    mat = signal(mock_position).val
    mock_signal = ift.Field.from_raw(position_space, mat)

    #data = signal_response(mock_position) + N.draw_sample_with_dtype(dtype=np.float64)
    data = R(mock_signal) + N.draw_sample_with_dtype(dtype=np.float64)
            
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
    N_samples = 10

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
        
    
    axis = 0
    #plot results
    mat1 = signal(mock_position).val
    mat2 = R.adjoint_times(data).val
    mat2 = ma.masked_values(mat2, 0.)
    
    mat3 = sc.mean.val
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
    main()
