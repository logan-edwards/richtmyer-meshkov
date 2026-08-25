'''
functions.py


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

