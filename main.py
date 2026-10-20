'''
main.py


'''
import functions
import classes

import numpy as np
import copy

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
    enable_animation,
    save_rate,
    disp,
    disp_rate
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
    velocity_prev1 = np.full(N, np.nan, dtype=np.complex128)
    velocity_prev2 = np.full(N, np.nan, dtype=np.complex128)
    dt_prev1 = 0
    dt_prev2 = 0

    vs = classes.VortexSheet(
        x,
        y,
        np.full(N, np.nan),
        np.full(N, np.nan),
        sheet_strength_init
    )

    wavelength = 2*np.pi/wavenumber

    sim_history = []
    time_history = []

    time = 0
    sim_step = 0
    while(time < final_time):
        # Save for animation if applicable
        if(disp==True):
            if(sim_step%disp_rate==0 and sim_step != 0):
                print(f"-" * 50)
                print(f"\n--- SIM STEP {sim_step} @ t={time}")
                print(f"Total Circulation = {np.sum(dGamma)}")
                print(f"Sheet size = {N} points")
                print(f"Timestep = {dt}")
                print(f"-" * 50)

        sim_history.append(copy.deepcopy(vs)) # NOTE this is fairly expensive and should be changed to only save data when wanted
        time_history.append(time)
        

        vs.z,vs.alpha,velocity_prev1,velocity_prev2 = functions.point_insertion(
            vs.z,
            vs.alpha,
            velocity_prev1,
            velocity_prev2,
            delta / 10
        )
        N = np.size(vs.z)

        # Compute immediately available quantities at time t:
        # ds, d\Gamma, \hat{T}, U, dU/ds
        ds = functions.compute_ds(
            vs.z,
            wavelength
        )
        dGamma = vs.sheet_strength * ds
        vs.tangent_vector = functions.compute_tangent_vector(
            vs.z,
            ds,
            wavelength
        )
        vs.dzdt = functions.compute_sheet_velocity(
            vs.z,
            dGamma,
            wavenumber,
            delta
        )
        dUds = functions.compute_dUds(
            vs.dzdt,
            ds
        )

        # Compute the CFL number and choose dt based on that
        # If CFL is small, increase dt slightly;
        # If CFL is large, cut dt in half;
        # Then: take min value between dt and the time to the end of sim as dt

        CFL = 0.5*dt*np.max(np.abs(atwood_number*vs.sheet_strength)/ds)
        if(CFL < 0.05):
            dt = 1.25 * dt
        elif(CFL > 0.3):
            dt = 0.5 * dt # NOTE this should be refined to repeatedly half dt until CFL is met.
        dt = np.minimum(dt, final_time-time)

        # Compute dU/dt
        # If we are on the 2nd or greater sim step, use BDF2
        # If we are on the 1st sim step, use BDF1
        # If we are on the 0th (initial) step, use analytical expression
        if(sim_step >= 2):
            r = dt_prev1/dt_prev2
            c1 = (1+2*r)/(1+r)
            c2 = (1+r)
            c3 = (r*r)/(1+r)
            dUdt = (c1 * vs.dzdt - c2 * velocity_prev1 + c3 * velocity_prev2)/dt_prev1
        elif(sim_step == 1):
            dUdt = (vs.dzdt - velocity_prev1)/dt_prev1
        else:
            dUdt = np.zeros(N, dtype=np.complex128) # NOTE this is a placeholder for the analytical expression

        # Compute numerical flux dFds
        dFds = functions.compute_dFds(
            vs.sheet_strength,
            ds,
            atwood_number
        )

        # Compute the sheet strength derivative, d\gamma/dt
        dgammadt = (
            -2 * atwood_number * (
            functions.complex_dot(dUdt, vs.tangent_vector)
            - functions.complex_dot(rt_accel, vs.tangent_vector)
            ) 
            - dFds 
            - vs.sheet_strength * functions.complex_dot(dUds, vs.tangent_vector)
        )
        vs.sheet_strength = vs.sheet_strength + dt * dgammadt # NOTE this is a placeholder for potentially higher-order (2nd order) time integration, although the flux is still 1st order(?)
        if(sim_step >= 1):
            vs.z = functions.integrate_ab2_adaptive(
                vs.z,
                vs.dzdt,
                velocity_prev1,
                dt,
                dt_prev1
            )
        else:
            vs.z = functions.integrate_euler(
                vs.z,
                vs.dzdt,
                dt
            )
        velocity_prev2 = velocity_prev1.copy()
        velocity_prev1 = vs.dzdt.copy()
        dt_prev2 = dt_prev1
        dt_prev1 = dt

        sim_step += 1
        time = time + dt

    if(enable_animation == True):
        z_frames = []
        sheet_str_frames = []
        matl_coord_frames = []
        frame_times = []
        for i in range(len(sim_history)):
            if(i%save_rate == 0):
                z_frames.append(sim_history[i].z)
                sheet_str_frames.append(sim_history[i].sheet_strength)
                matl_coord_frames.append(sim_history[i].material_coord)
                strength_plot_frames = [
                    material_coord + 1j * sheet_strength
                    for material_coord, sheet_strength
                    in zip(matl_coord_frames, sheet_str_frames)
                ]
                frame_times.append(time_history[i])
        functions.animate_sheet(
        z_frames,
        frame_times,
        'animation.mp4',
        'equal'
        )
        functions.animate_sheet(
            strength_plot_frames,
            frame_times,
            'sheet_strength.mp4',
            'auto'
        )

    return(sim_history,time_history)


def main():
    # KRASNY ICS:
    N = 1000
    dGamma = np.ones(N)
    dGamma = dGamma * (1/N)
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = i*dGamma[i]
        y[i] = 0
        dGamma[i] = dGamma[i] * 2*np.sin(2*np.pi*x[i])
    z = x+1j*y
    ds = functions.compute_ds(
        z,
        1
    )

    sim_hist, dt_hist = run_dynamic_simulation(
        x,
        y,
        dGamma/ds,
        0.6,
        0j,        # NOTE check sign convention on this
        1.5,
        0.0001,
        2*np.pi,
        0.1,
        True,
        20,
        True,
        20
    )

    return(0)

main()