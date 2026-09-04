'''
functions.py

This file contains the functions necessary for running the simulation.

Functions:
    complex_dot: Computes a dot product of complex numbers.
    K_delta_periodic: Computes the k-periodic Cauchy kernel.


Dependencies: numpy, numba
'''

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.animation as animation

@njit
def complex_dot(
    a, 
    b
    ):
    '''
    complex_dot

    This function computes a dot product when given two complex numbers

    Arguments:
        a (complex): a complex number
        b (complex): a complex number

    Returns:
        Dot product of a with b (scalar)
    '''
    return(np.real(a * np.conjugate(b)))

@njit
def K_delta_periodic(
    z, 
    k,
    delta
    ):
    '''
    K_delta_periodic

    This function computes the k-periodic Cauchy kernel.

    Arguments:
        z (complex): the denominator of the unregularized Cauchy kernel, i.e. 
        the spacing between two points on the vortex sheet.
        
        k (scalar): the wavenumber, i.e. 2*pi/wavelength

        delta (scalar): Smoothing term for the Cauchy kernel.

    Returns:
        Value of the regularized Cauchy kernel, interpreted as a velocity
        (complex) 
    '''

    # Note: this function needs division-by-0 checking still.

    wavelength = 2*np.pi / k
    x = np.real(z)
    y = np.imag(z)
    denominator = np.cosh(k*y) - np.cos(k*x) + delta**2
    K_delta = (-np.sinh(k*y) + 1j*np.sin(k*x)) / (2 * wavelength * denominator)
    return(K_delta)

@njit(parallel=True)
def compute_sheet_velocity(
    sheet_z,
    dGamma,
    k,
    delta
    ):
    '''
    compute_sheet_velocity

    This function evaluates the Birkhoff-Rott equation over the vortex sheet;
    i.e. this function computes the velocity of the points on the vortex sheet.

    Args:
        sheet_z (vector, complex): a vector of points discretizing the vortex
        sheet.

        dGamma (vector, real): a vector of circulation differentials, given by
        dGamma = gamma*ds, where gamma is the sheet strength.

        k: The wavenumber, i.e. 2*pi/wavelength

        delta: Smoothing term for the Cauchy kernel.
    
    Returns:
        Vector of sheet velocities (vector, complex)
    '''

    N = np.size(dGamma)
    sheet_dzdt = np.zeros(N, dtype=np.complex128)

    # Check the below code for race conditions; there shouldn't be a problem
    # from what I can tell because each i is a different point in memory.

    for i in prange(N):
        for j in range(N):
            sheet_dzdt[i] = sheet_dzdt[i] + K_delta_periodic(
                sheet_z[i] - sheet_z[j],
                k,
                delta
            ) * dGamma[j]

    return(sheet_dzdt)

@njit(parallel=True)
def integrate_euler(
    sheet_z,
    sheet_dzdt,
    dt
    ):
    '''
    integrate_euler

    This function uses Euler's method to update the sheet position with first-
    order accuracy. This should mainly be used for debugging, since the accuracy
    is insufficient for long-time study, and a more accurate integration scheme
    is cheap.

    Arguments:
        sheet_z (vector, complex): a vector of points discretizing the vortex
        sheet.

        sheet_dzdt (vector, complex): a vector of velocities at the current 
        timestep.

        dt (scalar): length of timestep.
    '''
    N = np.size(sheet_z)
    for i in prange(N):
        sheet_z[i] = sheet_z[i] + dt * sheet_dzdt[i]
    return(sheet_z)

def animate_sheet(
    z_data,
    time,
    filename
    ):
    fig, ax = plt.subplots()

    sheet_line, = ax.plot(
        np.real(z_data[:, 0]),
        np.imag(z_data[:, 0]),
        'k-'
    )
    
    ax.set_xlim(
        np.nanmin(np.real(z_data)),
        np.nanmax(np.real(z_data))
    )
    ax.set_ylim(
        np.nanmin(np.imag(z_data)),
        np.nanmax(np.imag(z_data))
    )
    
    ax.set_aspect('equal')

    def update(frame):
        sheet_line.set_xdata(np.real(z_data[:, frame]))
        sheet_line.set_ydata(np.imag(z_data[:, frame]))

        
        ax.set_title(f"t={time[frame]:.2f}")
        return sheet_line

    desired_time = 10  # seconds
    fps_desired = z_data.shape[1] / desired_time
    interval_desired = desired_time * 1000 / z_data.shape[1]

    print(f"framerate = {fps_desired} fps")

    sheet_animation = animation.FuncAnimation(
        fig,
        update,
        frames=z_data.shape[1],
        interval=interval_desired,
        blit=False
    )

    sheet_animation.save(filename, writer='ffmpeg', fps=fps_desired)