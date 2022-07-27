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
import numpy as np
from math import *

# cosmic ray parameters
c = 2.9979258 #* in units of 10^8 m/s
pc = 3.085677581 #* in units of 10^16 m
side_length = 3000 #pc
N_pixels = 2**6
cdt = side_length/N_pixels #* in units where cdt = 1pc
ell = cdt
energy = 0.5 #* in units of 10^20 eV
bb = 1. #* in units of 1μG = 10^(-10) T
charge = 1 #* in units where q = Ze
# position = [-8300, 0, 0]#[x,y,z] in pc in Galactocentric cartesian system
position = [side_length/2, side_length/2, 0]
velocity = [0,0,-1]#[-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)] #[vx,vy,vz] such that norm is 1 (equal to c)
energy = 0.5
charge = 1


class Cosmic_ray:
    def __init__(self, position, velocity, energy, charge):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.energy = energy
        self.charge = charge
        self.X_position = [self.position[0]]
        self.Y_position = [self.position[1]]
        self.Z_position = [self.position[2]]

    def update_velocity(self, new_velocity):
        self.velocity = np.array(new_velocity)

    def update_position(self, new_position):
        self.position = np.array(new_position)

    def backtracking_cubic_grid(self, mat):
        while abs(self.position[0]) < side_length and abs(self.position[1]) < side_length and abs(self.position[2]) < side_length:
            magnetic_field, magnetic_field_strength = Bfield(mat, self.position)
            D = (cdt * pc * c * self.charge * 1e-6) / self.energy
            A =  magnetic_field_strength * D
            previous_velocity = self.velocity - A * np.cross(self.velocity, magnetic_field)
            previous_position = self.position - self.velocity * cdt
            self.X_position.append(previous_position[0])
            self.Y_position.append(previous_position[1])
            self.Z_position.append(previous_position[2])

            self.update_position(previous_position)
            self.update_velocity(previous_velocity)
            self.velocity /= np.linalg.norm(self.velocity)

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

def deflection(mat, source_position):
    cr_cubic_grid = Cosmic_ray(position, velocity, energy, charge)
    pos_bef_cubic_grid = galactic_coordinates_from_velocity(cr_cubic_grid.velocity)
    cr_cubic_grid.backtracking_cubic_grid(mat)
    pos_aft_cubic_grid = galactic_coordinates_from_velocity(cr_cubic_grid.velocity)

    ang_distance = angular_distance(source_position, pos_aft_cubic_grid)
    return ang_distance, cr_cubic_grid.X_position, cr_cubic_grid.Y_position, cr_cubic_grid.Z_position

# def Bfield(mat, position): #measured in μG
#     """
#     Returns the unit vector and strength at position.
#     """
#     Bx = mat[0][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
#     By = mat[1][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
#     Bz = mat[2][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
#     B = np.sqrt(Bx**2 + By**2 + Bz**2)
#     unit_vector = np.array([Bx/B, By/B, Bz/B])
#     strength = B
#     return unit_vector, strength

def Bfield(mat, position): #measured in μG
    """
    Returns the unit vector and strength at position.
    """
    Bx = mat[0][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
    By = mat[1][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
    Bz = mat[2][int(position[0]/ell), int(position[1]/ell), int(position[2]/ell)]
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    unit_vector = np.array([Bx/B, By/B, Bz/B])
    strength = B
    return unit_vector, strength


def velocity_from_galactic_coords(longitude, latitude):
    b = latitude
    l = longitude
    v_z, v_x, v_y = 0, 0, 0

    v_z = np.sin(b)
    v_x = np.cos(b) * np.cos(l)
    v_y = np.cos(b) * np.sin(l)

    return np.array([v_x, v_y, v_z])

def galactic_coordinates_from_velocity(velocity):
    v_x = velocity[0]
    v_y = velocity[1]
    v_z = velocity[2]

    b = np.arcsin(v_z)
    l = np.sign(v_y) * np.arccos(v_x / np.cos(b))
    if l < 0:
        l += 2 * np.pi
    return np.array([l, b])

def angular_distance(position_A, position_B): #position in galactic coordinates (in rad)
    alpha_A, delta_A = position_A
    alpha_B, delta_B = position_B

    theta = np.arccos(
        np.sin(delta_A) * np.sin(delta_B)
        +
        np.cos(delta_A) * np.cos(delta_B) * np.cos(alpha_A - alpha_B)
    )

    return theta*180/np.pi #convert to degrees


def main():

    # defines domain
    position_space = ift.RGSpace([N_pixels, N_pixels, N_pixels])

    # magnetic field power spectrum parameters
    args = {
        # Amplitude of field fluctuations
        'fluctuations': (1e0, 1e0),

        # Exponent of power law power spectrum component. Choose a Kolmogorov slope
        # as a mean, and allow for a ± 1 std
        'loglogavgslope': (-11/3, 1e0),

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (1e-16, 1e0),

        # How ragged the integrated Wiener process component is
        'asperity': None
    }

    # creates magnetic field

    cfmaker = ift.CorrelatedFieldMaker(prefix='', total_N=3)
    cfmaker.add_fluctuations(position_space, **args)
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-6))
    pspec = cfmaker.power_spectrum
    signal = cfmaker.finalize()


    # Change model parameters for each component using dofdex

    #Define div cleaning operator.
    DC = div_cleaning(signal.target)

    mock_position = ift.from_random(DC(signal).domain, 'normal')

    # true_signal = DC(signal(mock_position))
    true_signal = signal(mock_position)

    # this is a (3,N_pixels,N_pixels,N_pixels) ndarray, containting the three components of the magnetic field
    Bfield_array = true_signal.val

#     arr = np.mean(Bfield_array[0], axis=0)
#     plt.imshow(arr, cmap = 'plasma')
#     plt.show()
    #############################################################################################################

    # So we have a magnetic field defined on a cube, with a specified value at each voxel.

    # set initial position and velocity
    # find the position of each cell in terms of

    cr_cubic_grid = Cosmic_ray(position, velocity, energy, charge)
    pos_bef_cubic_grid = galactic_coordinates_from_velocity(cr_cubic_grid.velocity)
    cr_cubic_grid.backtracking_cubic_grid(Bfield_array)
    pos_aft_cubic_grid = galactic_coordinates_from_velocity(cr_cubic_grid.velocity)

    source_position = pos_aft_cubic_grid
    
    from mpl_toolkits import mplot3d
    %matplotlib inline
    %matplotlib notebook
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot(cr_cubic_grid.X_position,cr_cubic_grid.Y_position,cr_cubic_grid.Z_position, color = 'k', label = 'CR path through true MF')

    rec_array = gaussian_filter(Bfield_array, sigma = 5)
    deltaTheta, Xpos, Ypos, Zpos = deflection(rec_array,source_position)
    print('deflection angle on PoS (in degrees): %.2f'%deltaTheta)
    
    ax.plot(Xpos,Ypos,Zpos, color = 'r', label = 'CR path through smoothed MF')
    ax.set_xlabel(r'$x$ (pc)', fontsize=15)
    ax.set_ylabel(r'$y$ (pc)', fontsize=15)
    ax.set_zlabel(r'$z$ (pc)', fontsize=15)
    ax.legend()

    
#     sigma_array = np.linspace(0, 100, num = 25)
#     deltaTheta = []

#     for sigma in sigma_array:
#         rec_array = gaussian_filter(Bfield_array, sigma = sigma)
#         deltaTheta.append(deflection(rec_array,source_position))

#     plt.plot(sigma_array, deltaTheta)
#     plt.ylabel(r'$\delta \theta$ (deg)')
#     plt.xlabel(r'$\sigma$')
#     plt.axhline(y = 3, linestyle = '--', color = 'k')
#     plt.show()

if __name__ == '__main__':
    t = time.time()
    main()
    t = time.time() - t
    t *= 1 #time in seconds
    print('Total calculation time: %.5f seconds'%t)
