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

ift.random.push_sseq_from_seed(27)

def random_los(n_los): 
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends


def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 1e-1  #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N))
    for i,j in itertools.product(range(N),range(N)):
        mat[i][j] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat

def make_random_mask(domain):
    # Random mask for spherical mode
    mask = ift.from_random(domain, 'pm1')
    mask = (mask + 1)/2
    return mask.val

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
    
    N_pixels = 72
    
    # Choose between random line-of-sight response (mode=0) and radial lines
    # of sight (mode=1)
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 0
        
    mode = 1
    filename = "getting_started_3_mode_{}_".format(mode) + "{}.png"
    position_space = ift.RGSpace([N_pixels, N_pixels])
    
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
     
    local_sampling = True
    if local_sampling:
        # Define response operator
        #mask = make_random_mask(position_space)
        mask = make_perc_mask(N_pixels)
        mask = ift.Field.from_raw(position_space, mask)
        Mask = ift.MaskOperator(mask)
        R = Mask
        signal_response = R(signal)
    else:
        # Build the line-of-sight response and define signal response
        LOS_starts, LOS_ends = random_los(100) if mode == 0 else radial_los(100)
        R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        signal_response = R(signal)


    # Specify noise
    data_space = R.target
    noise = .1
    N = ift.ScalingOperator(data_space, noise)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')

    data = signal_response(mock_position) + N.draw_sample_with_dtype(dtype=np.float64)
            
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

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth', zmin = 0, zmax = 1)
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec.force(mock_position)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))
    
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

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(KL.position), title="Latent mean", zmin = 0, zmax = 1)
        plot.add([pspec.force(KL.position + ss) for ss in KL.samples],
                 title="Samples power spectrum")
        plot.output(ny=1, ysize=6, xsize=16,
                    name=filename.format("loop_{:02d}".format(i)))
    

    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean", zmin = 0, zmax = 1)
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")
    
    mat1 = signal(mock_position).val
    mat2 = R.adjoint_times(data).val
    mat2 = ma.masked_values(mat2, 0.)
    
    mat3 = sc.mean.val
    mat4 = mat1 - mat3
    
    
    args = [mat1,
            mat2,
            mat3,
            mat4]
    
    plots(*args)

    powers = [pspec.force(s + KL.position) for s in KL.samples]
    sc = ift.StatCalculator()
    for pp in powers:
        sc.add(pp.log())
    plot.add(
        powers + [pspec.force(mock_position),
                  pspec.force(KL.position), sc.mean.exp()],
        title="Sampled Posterior Power Spectrum",
        linewidth=[1.]*len(powers) + [3., 3., 3.],
        label=[None]*len(powers) + ['Ground truth', 'Posterior latent mean', 'Posterior mean'])
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()
