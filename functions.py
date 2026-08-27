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

    x = np.real(z)
    y = np.imag(z)
    denominator = np.cosh(k*y) - np.cos(k*x) + delta**2
    if(denominator < 1e-12):
        return 0
    return((-np.sinh(k*y) + 1j * np.sin(k*x)) / denominator)

@njit(parallel=True)
def compute_sheet_velocity(
    sheet_z,
    circulation,
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

        k: The wavenumber, i.e. 2*pi/wavelength

        delta: Smoothing term for the Cauchy kernel.
    
    Returns:
        Vector of sheet velocities (vector, complex)
    '''

    N = np.size(sheet_z)

    sheet_velocity = np.zeros(N)
    for i in prange(N):
        for j in range(N):
            sheet_velocity[i] = K_delta_periodic(
                sheet_z[i] - sheet_z[j],
                k,
                delta
            ) * circulation[j]

    return(sheet_velocity)

@njit(parallel=True)
def integrate_ab2(
    sheet_z,
    sheet_dzdt,
    sheet_dzdt_prev,
    dt
    ):
    '''
    integrate_ab2

    This function uses AB2 to compute the updated vortex sheet positions with
    2nd order accuracy. If there is no previous data, one can pass an array
    containing np.nan, in which case the function reverts to 1st order Euler's
    method.

    Arguments:
        sheet_z (vector, complex): a vector of points discretizing the vortex 
        sheet.

        sheet_dzdt (vector, complex): a vector of velocities at the current
        timestep.

        sheet_dzdt_prev (vector, complex): a vector of velocities from the
        previous timestep. If no such data exists, pass a vector of nans to 
        revert to a 1st order integration scheme.

        dt (scalar): length of timestep.
    
    Returns:
        Vector of new sheet positions (vector, complex)
    '''

    N = np.size(sheet_z)
    for i in prange(N):
        if(not np.isnan(sheet_dzdt_prev[i])):
            sheet_z[i] = sheet_z[i] + dt * (
                1.5 * sheet_dzdt[i] - 0.5 * sheet_dzdt_prev[i]
            )
        else:
            sheet_z[i] = sheet_z[i] + dt * sheet_dzdt[i]

    return(sheet_z)

@njit(parallel=True)
def compute_ds(
    sheet_z
    ):
    N = np.size(sheet_z)
    ds = np.zeros(N)
    # Is this the proper way to handle the periodic BC?
    for i in prange(N-1):
        ds[i] = np.abs(sheet_z[i+1] - sheet_z[i])
    ds[N-1] = 0
    return(ds)