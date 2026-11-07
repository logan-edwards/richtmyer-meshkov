'''
ic.py

This file contains the code necessary to generate post-shock initial conditions
for the vortex sheet model, given an initial perturbation geometry, fluid
properties, and shock Mach number.
'''

import numpy as np

def light_to_heavy_smallamp(
    mach_number,
    atwood_number,
    shock_angle,
    gamma,
    wavelength,
    interface
    ):
    '''
    Initial conditions for shock passing light-to-heavy, with small-amplitude
    initial perturbation and non-oblique shock (small incident shock angle)

    This expression is only valid for ideal gases (adiabatic gamma = 1.4)

    NOTE need to check if that's actually the case, i think this might work
    (or at least the circulation expression might work) for the Taylor
    series expansion even for large incident angle (i.e. oblique shock) 
    '''

    N = np.size(interface)
    ic_sheet_strength = np.zeros(N)

    alpha = np.zeros(N) # local angle between the incident shock and interface
    for i in range(N-1):
        x = np.real(interface[i+1] - interface[i])
        y = np.imag(interface[i+1] - interface[i])
        alpha[i] = np.arctan(y/x) - shock_angle
    x = np.real(interface[0] + wavelength - interface[N-1])
    y = np.imag(interface[0] - interface[N-1])
    alpha[N-1] = np.arctan(y/x) - shock_angle

    eta = (1 + atwood_number) / (1 - atwood_number) # density ratio rho-/rho+

    # sheet strength
    # NOTE need to check if this is dimensioned
    ic_sheet_strength = (
        np.sin(alpha)
        * ((2 * gamma**0.5) / (1+gamma))
        * (1 - eta**-0.5)
        * (1 + mach_number**-1 + 2*mach_number**-2)
        * (mach_number - 1)
    )

    return(interface, ic_sheet_strength)

def light_to_heavy_smallamp_oblique(
    mach_number,
    atwood_number,
    shock_angle,
    interface
    ):
    '''
    Initial conditions for shock passing light-to-heavy, with small-amplitude
    initial perturbation and non-oblique shock (small incident shock angle)
    '''

    N = np.size(interface)
    ic_sheet_strength = np.zeros(N)





    return(interface, ic_sheet_strength)

def light_to_heavy_largeamp(
    mach_number,
    atwood_number,
    shock_angle,
    interface
    ):

    N = np.size(interface)
    ic_interface = np.zeros(N, dtype=np.complex128)
    ic_sheet_strength = np.zeros(N)



    return(ic_interface, ic_sheet_strength)

def light_to_heavy_largeamp_oblique(
    mach_number,
    atwood_number,
    shock_angle,
    interface
    ):

    N = np.size(interface)
    ic_interface = np.zeros(N, dtype=np.complex128)
    ic_sheet_strength = np.zeros(N)



    return(ic_interface, ic_sheet_strength)

def phi(xi, zeta):
    return(xi * (1 + (zeta**2)*xi) / ((zeta**2) + xi))