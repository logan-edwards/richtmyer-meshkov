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
def compute_sheet_strength_derivative(
    atwood_number,
    tangent_vector,
    dUdt,
    dgamma2ds,
    acceleration,
    sheet_strength,
    dUds
    ):
    '''
    compute_sheet_strength_derivative

    This function computes the derivative of sheet strength with respect to
    time, dgamma/dt.

    Arguments:
        atwood_number (scalar): the Atwood number, A=(rho- - rho+)/(rho+ + rho-)

        tangent_vector (vector, complex): an array containing the unit tangent
        vector (a complex number with modulus 1) at each point on the sheet.

        dUdt (vector, complex): an array containing the fluid acceleration
        induced by the motion of the sheet itself.

        dgamma2ds (vector, real): an array containing the derivative of
        (gamma^2) with respect to s, relating to the advection of sheet strength
        across the vortex sheet.

        acceleration (vector, complex): an array containing the complex-valued
        reference frame acceleration; a nonzero acceleration term corresponds to
        a coupled Rayleigh-Taylor instability.

        sheet_strength (vector, real): an array containing the real-valued sheet
        strength at each point on the body.

        dUds (vector, complex): a vector containing the complex-valued
        derivative of vortex sheet velocity with respect to arclength. This term
        represents the effect of sheet stretching on sheet strength evolution.

    Returns:
        vector of real-valued time derivatives of sheet strength.
    '''

    N = np.size(tangent_vector)
    tangent_accel = complex_dot(dUdt, tangent_vector) - 0.125 * dgamma2ds
    rt_accel = complex_dot(acceleration, tangent_vector)
    stretch_term = sheet_strength * complex_dot(dUds, tangent_vector)

    return(-2 * atwood_number * (tangent_accel - rt_accel) - stretch_term)

@njit(parallel=True)
def compute_ds(
    z,
    wavelength
    ):
    '''
    compute_ds

    This function computes the differential ds in sheet arclength using a line
    segment approximation. Improvements may involve a polynomial spline or
    similar procedure.

    Arguments:
        z (vector, complex): an array of the complex-valued position of points
        on the vortex sheet.

        wavelength (scalar): the wavelength over which motion is periodic.
    '''

    N = np.size(z)

    ds = np.zeros(N)
    for i in prange(1, N-1):
        ds[i] = 0.5 * np.abs(z[i+1] - z[i-1])
    ds[0] = 0.5 * np.abs(z[1] - (z[N-1] - wavelength))
    ds[N-1] = 0.5 * np.abs((z[0] + wavelength) - z[N-2])

    return(ds)

@njit(parallel=True)
def compute_tangent_vector(
    z,
    ds,
    wavelength
    ):
    '''
    
    '''
    N = np.size(z)

    tangent_vector = np.zeros(N, dtype=np.complex128)
    for i in prange(1, N-1):
        tangent_vector[i] = 0.5 * (z[i+1] - z[i-1]) / ds[i]
    tangent_vector[0] = 0.5 * (z[1] - (z[N-1] - wavelength)) / ds[0]
    tangent_vector[N-1] = 0.5 * ((z[0] + wavelength) - z[N-2]) / ds[N-1]

    return(tangent_vector)


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

@njit(parallel=True)
def integrate_ab2(
    sheet_z,
    sheet_dzdt,
    sheet_dzdt_prev,
    dt
    ):
    '''
    integrate_ab2
    
    This function uses 2nd order Adams-Bashforth to update the sheet position.
    This can be used for all steps beyond the first timestep.

    Arguments:
        sheet_z (vector, complex): a vector points discretizing the vortex
        sheet.

        sheet_dzdt (vector, complex): a vector of velocities at the current
        timestep.

        sheet_dzdt_prev (vector, complex): a vector of velocities at the prior
        timestep.

        dt (scalar): length of timestep.
    '''

    N = np.size(sheet_z)
    for i in prange(N):
        sheet_z[i] = sheet_z[i] + 0.5 * dt * (
            3*sheet_dzdt[i] - sheet_dzdt_prev[i]
        )
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