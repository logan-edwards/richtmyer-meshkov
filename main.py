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
    sheet_strength_init,
    atwood_number,
    rt_accel,
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
    velocity_prev = np.zeros(N, dtype=np.complex128)
    velocity_prev_prev = np.zeros(N, dtype=np.complex128)
    dgammadt_prev = np.zeros(N)

    vs = classes.VortexSheet(
        x,
        y,
        np.full(N, np.nan),
        np.full(N, np.nan),
        sheet_strength_init
    )
    wavelength = 2*np.pi/wavenumber

    Nt = int(final_time / dt) + 1
    if(enable_animation == True):
        z_data = np.full((N, Nt), np.nan+1j*np.nan)
        z_data[:,0] = np.copy(vs.z)

    for i in range(0,Nt):
        ds = functions.compute_ds(
            vs.z,
            wavelength
        )
        dGamma = vs.sheet_strength * ds

        vs.dzdt = functions.compute_sheet_velocity(
            vs.z,
            dGamma,
            wavenumber,
            delta
        )

        vs.tangent_vector = functions.compute_tangent_vector(
            vs.z,
            ds,
            wavelength
        )

        dUds = np.zeros(N, dtype=np.complex128)
        for j in range(1,N-1):
            dUds[j] = (vs.dzdt[j+1] - vs.dzdt[j-1]) / (2*ds[j])
        dUds[0] = (vs.dzdt[1] - vs.dzdt[N-1]) / (2*ds[0])
        dUds[N-1] = (vs.dzdt[0] - vs.dzdt[N-2]) / (2*ds[N-1])

        dUdt = np.zeros(N, dtype=np.complex128)
        for j in range(N):
            if(i>1):
                dUdt[j] = (3*vs.dzdt[j] - 4*velocity_prev[j] + 
                    velocity_prev_prev[j]) / (2*dt)
            elif(i==1):
                dUdt[j] = (vs.dzdt[j] - velocity_prev[j]) / dt
            else:
                dUdt[j] = 0

        if(enable_animation == True):
            z_data[:,i] = np.copy(vs.z)

        if(i >= 1):
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
        if(i >= 1):
            velocity_prev_prev = np.copy(velocity_prev)
        velocity_prev = np.copy(vs.dzdt)

        vs.sheet_strength = functions.update_sheet_strength(
            atwood_number,
            vs.tangent_vector,
            dUdt,
            rt_accel,
            vs.sheet_strength,
            dUds,
            dt,
            ds
        )
        if(i%20==0):
            print(f"\n--- TIMESTEP {i} @ t={i*dt}")
            print(f"Total Circulation = {np.sum(dGamma)}")

    if(enable_animation == True):
        functions.animate_sheet(
            z_data,
            np.linspace(0,final_time,Nt),
            'animation.mp4'
        )
    
    return(vs)

def main():
    # KRASNY ICS:
    N = 1600
    dGamma = np.ones(N)
    dGamma = dGamma * (1/N)
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = i*dGamma[i] + 0.01 * np.sin(2*np.pi*i*dGamma[i])
        y[i] = -0.01 * np.sin(2*np.pi*i*dGamma[i])
    z = x+1j*y
    ds = functions.compute_ds(
        z,
        1
    )

    run_dynamic_simulation(
        x,
        y,
        dGamma/ds,
        0.2,
        0,
        1.5,
        0.0005,
        2*np.pi,
        0.1,
        True
    )

    return(0)

main()