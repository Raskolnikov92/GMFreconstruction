import sys
import numpy as np
import nifty7 as ift
import random as rn
import itertools
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import dft
from scipy import fft, fftpack

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 1e-2 #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N))
    for i,j in itertools.product(range(N),range(N)):
        mat[i][j] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    N_nonzero = np.count_nonzero(mat)
    if not N_nonzero > 0:
        raise ValueError('You must have some datapoints.')
    return mat

def vector_plot(div_cleaned_matx, div_cleaned_maty, div_cleaned_rec_matx, div_cleaned_rec_maty):
    N_pixels = len(div_cleaned_matx)
    L = 1    
    
    # Create Meshgrid
    x = np.linspace(0, L, N_pixels)
    y = np.linspace(0, L, N_pixels)
    xx, yy = np.meshgrid(x, y)

    cmap = 'viridis'
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))    
         
    Bx = div_cleaned_matx
    By = div_cleaned_maty
    B_mag = np.sqrt(Bx**2 + By**2)
    Bx_dir = Bx/B_mag
    By_dir = By/B_mag

    plt.subplot(121)
    plt.contourf(xx, yy, B_mag, cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, Bx_dir, By_dir, color="black")
    plt.title('Original Vector Field')
    plt.axis("image")
    
    rec_Bx = np.real(div_cleaned_rec_matx)
    rec_By = np.real(div_cleaned_rec_maty)
    rec_B_mag = np.sqrt(rec_Bx**2 + rec_By**2)
    rec_Bx_dir = rec_Bx/rec_B_mag
    rec_By_dir = rec_By/rec_B_mag


    plt.subplot(122)
    plt.contourf(xx, yy, rec_B_mag, cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, rec_Bx_dir, rec_By_dir, color="black")
    plt.title('Reconstructed field (div cleaned)')
    plt.axis("image") 
    
    """
    Bx = rec_matx
    By = rec_maty
    B_mag = np.sqrt(Bx**2 + By**2)
    Bx_dir = Bx/B_mag
    By_dir = By/B_mag

    plt.subplot(133)
    plt.contourf(xx, yy, B_mag, cmap=cmap)
    plt.colorbar()
    plt.streamplot(x, y, Bx_dir, By_dir, color="black")
    plt.title('Not div cleaned reconstruction')
    plt.axis("image")
    """
    plt.show()
    

    
def scalar_plot(mat1, mat2, mat3, mat4):
    cmap = 'viridis'
    
    fig, axes = plt.subplots(2, 2, figsize=(15,15))
    
    pl1 = axes[0,0].imshow(mat1, cmap=cmap)
    axes[0,0].set_title('Mock Signal', fontsize=20)
    axes[0,0].set(ylabel='Pixel number')
        
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl1, cax=cax, orientation='vertical')
    
    pl2 = axes[0,1].imshow(mat2, cmap=cmap)
    axes[0,1].set_title('Data (averaged per pixel )',fontsize=20)
    
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
        
# Set prior correlation covariance with a power spectrum leading to
# homogeneous and isotropic statistics
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


#Function that divergence cleans ndarrays. This is the 2d version
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

# Divergence function
def divergence(f,sp):
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])
    
def main():    

    # Number of pixels in N_pixelsxN_pixels grid   
    N_pixels = 1000
    
    # Define the configuration space. Here a square grid.   
    position_space = ift.RGSpace([N_pixels, N_pixels])
            
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
    mock_signalx = S.draw_sample_with_dtype(dtype=np.float64)   
    mock_signaly = S.draw_sample_with_dtype(dtype=np.float64)
    
    # Extract the GRF field values as ndarrays
    matx = HT(mock_signalx).val  
    maty = HT(mock_signaly).val
    
    dft_matx = fftpack.fftn(matx)
    dft_maty = fftpack.fftn(maty)
        
    dft_matx, dft_maty = div_clean_2d(dft_matx, dft_maty)
    
    div_cleaned_matx = np.real(fftpack.ifftn(dft_matx))
    div_cleaned_maty = np.real(fftpack.ifftn(dft_maty))
    
    # the mock signal is the div cleaned version of the original GRF
    mock_signalx = HT2(ift.Field.from_raw(position_space, div_cleaned_matx))
    mock_signaly = HT2(ift.Field.from_raw(position_space, div_cleaned_maty))
    
    
    # Define response operator
    mask = make_perc_mask(N_pixels)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)
    R = Mask(HT1)
    
    # Define noise covariance
    noise = np.max(mock_signalx.val.flatten(order='C'))     
    data_space = R.target
    N = ift.ScalingOperator(data_space, noise)
    mock_noise = N.draw_sample_with_dtype(dtype=np.float64)

    # Our linear measurement equation
    datax = R(mock_signalx) + mock_noise 
    datay = R(mock_signaly) + mock_noise
    
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
    
    # Get ndarray containing values of reconstruction in each pixel
    rec_matx = HT1(mx).val 
    rec_maty = HT1(my).val 
        
    # Get DFT matrix for each component 
    dft_rec_matx = fftpack.fftn(rec_matx)
    dft_rec_maty = fftpack.fftn(rec_maty)
    
    # Apply the divergence cleaning function
    dft_rec_matx, dft_rec_maty = div_clean_2d(dft_rec_matx, dft_rec_maty)
    
    # Invert the dft matrix after application of the divergence cleaning
    div_cleaned_rec_matx = np.real(fftpack.ifft2(dft_rec_matx))
    div_cleaned_rec_maty = np.real(fftpack.ifft2(dft_rec_maty))    
    
    #######################
    #Main part ends here. Below we plot results. 
    #######################
    
    #Vector plot 
    
    
    args = [div_cleaned_matx, 
            div_cleaned_maty, 
            div_cleaned_rec_matx, 
            div_cleaned_rec_maty]
    
    #vector_plot(*args)
    
    L = 1
    
    # Create Meshgrid
    x = np.linspace(0, L, N_pixels)
    y = np.linspace(0, L, N_pixels)
    xx, yy = np.meshgrid(x, y)


    
    cmap = 'viridis'
    
    F = [matx,maty]
    
    # Compute Divergence
    sp_x = np.diff(x)[0]
    sp_y = np.diff(y)[0]
    sp = [sp_x, sp_y]
    
    mat1 = divergence(F, sp)
    
    fig, axes = plt.subplots(1, 2, figsize=(18,5))
    
    pl1 = axes[0].imshow(mat1, cmap=cmap)
    axes[0].set_title('Divergence (before div cleaning)', fontsize=20)
    axes[0].set(ylabel='Pixel number')
    axes[0].set(xlabel='Pixel number')
        
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl1, cax=cax, orientation='vertical')
    
    F = [div_cleaned_matx, div_cleaned_maty]
    
    mat2 = divergence(F, sp)
    
    pl2 = axes[1].imshow(mat2, cmap=cmap)
    axes[1].set_title('Divergence (after div cleaning)',fontsize=20)
    axes[1].set(ylabel='Pixel number')
    axes[1].set(xlabel='Pixel number')
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl2, cax=cax, orientation='vertical')
    plt.show()
    
      
if __name__ == '__main__':
    main()
    
