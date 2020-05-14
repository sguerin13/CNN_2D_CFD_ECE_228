import numpy as np
import sympy as sp
import pylbm
import sys
import obs_gen
import datetime
import os

"""
Von Karman vortex street simulated by Navier-Stokes solver D2Q9

- Modifications:
    - Random Circle Radius and Location in domain
    - Random Fluid Velocity
        - This creates a random reynolds number
    - Simulation runs until it settles


- time step = 1/60th of a second - so grid is 60x120
- wait betwen 2 and 10 seconds to start grabbing frames and grab 10 seconds of clips at a 30 Hz sampling rate

"""

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '*' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


X, Y, LA = sp.symbols('X, Y, LA')
rho, qx, qy = sp.symbols('rho, qx, qy')

def bc_rect(f, m, x, y, rhoo, uo):
    m[rho] = 0.
    m[qx] = rhoo*uo
    m[qy] = 0.

def vorticity(sol):
    qx_n = sol.m[qx]
    qy_n = sol.m[qy]
    vort = np.abs(qx_n[1:-1, 2:] - qx_n[1:-1, :-2]
                  - qy_n[2:, 1:-1] + qy_n[:-2, 1:-1])
    return vort.T

    
date = datetime.date.today()
day = date.day
month = date.month
year = date.year

# parameters
xmin, xmax, ymin, ymax = 0., 2., 0., 1.
radius = np.random.rand(1)*.4 +.05
radius = radius[0]

dx = 1./60

batch_counter = 0

# obs = obs_gen.get_random_triangle()

numpy_save = True
while batch_counter < 333:


    uo = .05

    radius = np.random.rand(1)*.4 +.05
    radius = radius[0]

    Re_Tar = 4000*np.random.rand(1)[0]

    la = 1. # velocity of the scheme
    rhoo = 1. # fluid density

    mu = rhoo*uo*radius*2/Re_Tar # bulk visocity
    # print(mu)
    zeta = 10*mu # shear viscosity
    dummy = 3.0/(la*rhoo*dx)
    s1 = 1.0/(0.5+zeta*dummy)
    s2 = 1.0/(0.5+mu*dummy)
    s  = [0.,0.,0.,s1,s1,s1,s1,s2,s2]
    dummy = 1./(LA**2*rhoo)
    qx2 = dummy*qx**2
    qy2 = dummy*qy**2
    q2  = qx2+qy2
    qxy = dummy*qx*qy



    first = True
    print("Batch #", batch_counter)
    # obs = obs_gen.get_random_triangle()
    x_loc = np.random.rand(1)*1.25 + radius+.05
    y_loc = 0.5*(np.random.rand(1))+radius+.05
    x_loc = x_loc[0]
    y_loc = y_loc[0]
    obs = pylbm.Circle([x_loc, y_loc], radius, label=2)

    dico = {
        
        'box': {
            'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': [0, 1, 0, 0]
        },
        'elements': [obs],
        'space_step': dx,
        'scheme_velocity': LA,
        'schemes': [
            {
                'velocities': list(range(9)),
                'polynomials': [
                    1,
                    LA*X, LA*Y,
                    3*(X**2+Y**2)-4,
                    0.5*(9*(X**2+Y**2)**2-21*(X**2+Y**2)+8),
                    3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y,
                    X**2-Y**2, X*Y
                ],
                'relaxation_parameters': s,
                'equilibrium': [
                    rho,
                    qx, qy,
                    -2*rho + 3*q2,
                    rho - 3*q2,
                    -qx/LA, -qy/LA,
                    qx2 - qy2, qxy
                ],
                'conserved_moments': [rho, qx, qy],
            },
        ],
        'init': {rho: rhoo,
                 qx: uo,
                 qy: 0.
        },
        'parameters': {LA: la},
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiBounceBack}, 'value': (bc_rect, (rhoo, uo))},
            1: {'method': {0: pylbm.bc.NeumannX}},
            2: {'method': {0: pylbm.bc.BouzidiBounceBack}},
        },
        'generator': "cython",
    }
    
    x_range = int((xmax-xmin)/dx)
    y_range = int((ymax-ymin)/dx)
    
    x_search = np.linspace(xmin,xmax, x_range)
    y_search = np.linspace(ymin,ymax, y_range)
    
    presence, sdf = obs_gen.get_distance(x_search,y_search, obs)
    
    sol = pylbm.Simulation(dico)
    
    Re = rhoo*uo*2*radius/mu
    print("Reynolds number {0:10.3e}".format(Re))
    print("time step %f ",sol.dt)
    x, y = sol.domain.x, sol.domain.y
    
    
    if numpy_save:

        # start time
        start_time = np.random.randint(2,10)
        end_time = start_time + 10
        Tf = float(end_time)
        im = 0
        l = Tf / sol.dt / 30
        printProgress(im, l, prefix='Progress:', suffix='Complete', barLength=50)
        path =  os.getcwd() + '\\..\\data\\Circle_Trans_30fps\\'
        filename = 'Circle_Transient_' + str(month) + '_' + str(day) + '_' + str(year) + '_' + str(batch_counter)+ '_' + str(int(Re))
        filename = path + filename
        
        stable = False
        prev_vel = np.zeros((x_range,y_range))
        
        while sol.t < start_time:
            # wait until we start logging data
            for k in range(2):
                sol.one_time_step()
            im += 1
            # printProgress(im, l, prefix='Progress:', suffix='Complete', barLength=50)
            #print(sol.t)
            
            vel = sol.m[qx] + sol.m[qy]*1j
            d_vel = prev_vel - vel
            sp = np.absolute(d_vel)
            max_sp = np.max(sp)

        # start recording
        while sol.t < Tf:
            for k in range(2):
                sol.one_time_step()
            im += 1
            #printProgress(im, l, prefix='Progress:', suffix='Complete', barLength=50)
            
            vel = sol.m[qx] + sol.m[qy]*1j
            d_vel = prev_vel - vel
            sp = np.absolute(d_vel)
            max_sp = np.max(sp)
            # check to see if we are getting NaN's
            # if np.isnan(max_sp):
            #     print('solver failed')
            #     break
            # print("Obs num", obs_num, "Max speed =", max_sp)
            stable =  max_sp < 1e-5
            prev_vel = vel.copy()
            
            if first and stable:
                sdf_map = sdf[:,:,np.newaxis]
                mask    = presence[:,:,np.newaxis]
                vx_map = sol.m[qx][:,:,np.newaxis]
                vy_map = sol.m[qy][:,:,np.newaxis]
                rho_map = sol.m[rho][:,:,np.newaxis]
                first = False
                break

            elif first:
                sdf_map = sdf[:,:,np.newaxis]
                mask    = presence[:,:,np.newaxis]
                vx_map = sol.m[qx][:,:,np.newaxis]
                vy_map = sol.m[qy][:,:,np.newaxis]
                rho_map = sol.m[rho][:,:,np.newaxis]
                first = False



            elif stable:
                #sdf_map = np.dstack((sdf_map, sdf))
                vx_map = np.dstack((vx_map, sol.m[qx]))
                vy_map = np.dstack((vy_map, sol.m[qy]))
                rho_map = np.dstack((rho_map, sol.m[rho]))
                break

            else:
                vx_map = np.dstack((vx_map, sol.m[qx]))
                vy_map = np.dstack((vy_map, sol.m[qy]))
                rho_map = np.dstack((rho_map, sol.m[rho]))

      
        if stable:
            if np.isnan(max_sp):
                print('solver failed')
            else:
                print("Stability reached at t =", sol.t)
                np.savez(filename,sdf = sdf_map,mask = mask, vx=vx_map, vy=vy_map, rho=rho_map,Re = Re,Vo = uo,rhoo = rhoo)
                batch_counter +=1
                print('\n batch no: %i',batch_counter)
        else:
            if np.isnan(max_sp):
                print('solver failed')
            else:
                print("Stability never reached")
                np.savez(filename,sdf = sdf_map,mask = mask, vx=vx_map, vy=vy_map, rho=rho_map,Re = Re,Vo = uo,rhoo = rhoo)
                batch_counter+=1
                print('\n batch no: %i',batch_counter)

        
                
        
    else:
        viewer = pylbm.viewer.matplotlib_viewer
        fig = viewer.Fig()
        ax = fig[0]
        #ax.ellipse([.3/dx, 0.5*(ymin+ymax)/dx+2], [radius/dx, radius/dx], 'r')
        image = ax.image(vorticity(sol), cmap='plasma', clim=[0, .25])
        epsilon = 1e-3
        prev_x = sol.m[qx]
        prev_y = sol.m[qy]
    

        def update(iframe):
            nrep = 2

            for i in range(nrep):
                sol.one_time_step()
            image.set_data(sol.m[qx])
            #print(sol.m[qx][:10,0])
            if np.isnan(sol.m[qx][0,0]):
                print('solver failed')
                sys.exit()
            ax.title = "Solution t={0:f}, nt = {0:f}".format(sol.t,sol.nt)
    
        # run the simulation
        fig.animate(update, interval=1/32)
        fig.show()
