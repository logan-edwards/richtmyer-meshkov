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
    

    run_kinematic_simulation(
        x,
        y,
        dGamma,
        4,
        0.01,
        2*np.pi,
        0.2,
        True
    )

    return(0)

main()