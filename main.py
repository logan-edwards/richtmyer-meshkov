'''
main.py


'''
import functions
import classes

import numpy as np

def run_kinematic_simulation(
    x, 
    y, 
    circulation, 
    final_time,
    dt,
    wavenumber,
    delta,
    enable_animation
    ):
    '''
    run_kinematic_simulation

    Simulates the vortex sheet roll-up from the purely kinematic description,
    i.e. by solving the Birkhoff-Rott equation with constant sheet strength in
    time.

    Args:
        
    Returns:
        Instance of the VortexSheet class
    '''

    vs = classes.VortexSheet(
        x,
        y,
        np.full(np.size(x), np.nan),
        np.full(np.size(x), np.nan),
        circulation
    )

    Nt = int(final_time / dt) + 1
    if(enable_animation == True):
        z_data = np.full((np.size(x), Nt), np.nan+1j*np.nan)
        z_data[:,0] = np.copy(vs.z)

    for i in range(0,Nt):
        velocity_prev = np.copy(vs.dzdt)
        if(enable_animation == True):
            z_data[:,i] = np.copy(vs.z)
        vs.dzdt = functions.compute_sheet_velocity(
            vs.z,
            vs.circulation,
            wavenumber,
            delta
        )
        if(i > 1):
            vs.z = functions.integrate_ab2(
                vs.z,
                vs.dzdt,
                velocity_prev,
                dt
            )
        else:
            vs.z = functions.integrate_euler(
                vs.z,
                vs.dzdt,
                dt
            )
        velocity_prev = np.copy(vs.dzdt)   
        
                 

    if(enable_animation == True):
        functions.animate_sheet(
            z_data,
            np.linspace(0,final_time,Nt),
            'animation.mp4'
        )
    
    return(vs)

def run_dynamic_simulation(
    x,
    y,
    circulation,
    atwood_number,
    final_time,
    dt,
    wavenumber,
    delta,
    enable_animation    
    ):
    '''
    run_dynamic_simulation

    Simulates the vortex sheet roll-up governed by the Birkhoff-Rott equation,
    coupled with the vortex sheet strength evolution.

    Args:
        
    Returns:
        Instance of the VortexSheet class
    '''

    N = np.size(x)

    vs = classes.VortexSheet(
        x,
        y,
        np.full(N, np.nan),
        np.full(N, np.nan),
        circulation
    )
    wavelength = 2*np.pi/wavenumber

    ds = functions.compute_ds(
        vs.z,
        wavelength
    )
    vs.sheet_strength = vs.circulation / ds

    Nt = int(final_time / dt) + 1
    if(enable_animation == True):
        z_data = np.full((N, Nt), np.nan+1j*np.nan)
        z_data[:,0] = np.copy(vs.z)

    for i in range(0,Nt):
        velocity_prev = np.copy(vs.dzdt)
        if(enable_animation == True):
            z_data[:,i] = np.copy(vs.z)
        
        vs.dzdt = functions.compute_sheet_velocity(
            vs.z,
            vs.circulation,
            wavenumber,
            delta
        )

        if(i > 1):
            vs.z = functions.integrate_ab2(
                vs.z,
                vs.dzdt,
                velocity_prev,
                dt
            )
        else:
            vs.z = functions.integrate_euler(
                vs.z,
                vs.dzdt,
                dt
            )
        velocity_prev = np.copy(vs.dzdt)

        # Update the sheet strength, circulaton, etc
        ds = functions.compute_ds(vs.z, wavelength)
        dUds = np.zeros(N, dtype=np.complex128)
        for i in range(1,N-1):
            dUds[i] = (vs.dzdt[i+1] - vs.dzdt[i-1]) / (2*ds[i])
        dUds[0] = (vs.dzdt[1] - vs.dzdt[0]) / ds[0]
        dUds[N-1] = (vs.dzdt[N-1] - vs.dzdt[N-2]) / ds[N-1]

        dgammadt = functions.compute_sheet_strength_derivative(
            0,
            functions.compute_tangent_vector(vs.z, ds, wavelength),
            0,
            0,
            0,
            vs.sheet_strength,
            dUds
        )

        vs.sheet_strength += dt * dgammadt
        vs.circulation = vs.sheet_strength * ds

        # NOTE: It appears that the stretching term is already factored in when
        # circulation is constant, i.e. the stretching term being explicitly
        # computed is actually breaking KCT. Check this condition more closely
        # and perhaps handle the other 3 (fluid accel, flux, ref frame/RT accel)
        # independently before converting to circulation for integration. 
        print(f"Circulation = {np.sum(vs.circulation)}")


    if(enable_animation == True):
        functions.animate_sheet(
            z_data,
            np.linspace(0,final_time,Nt),
            'animation.mp4'
        )
    
    return(vs)    

def main():
    # KRASNY ICS:
    N = 400
    dGamma = np.ones(N)
    dGamma = dGamma * (1/N)
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = i*dGamma[i] + 0.01 * np.sin(2*np.pi*i*dGamma[i])
        y[i] = -0.01 * np.sin(2*np.pi*i*dGamma[i])
    

    run_dynamic_simulation(
        x,
        y,
        dGamma,
        0,
        4,
        0.01,
        2*np.pi,
        0.2,
        True
    )

    return(0)

main()