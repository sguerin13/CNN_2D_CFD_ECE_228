import numpy as np
import torch
def sim_to_x_y_1_lag(path):
    '''
    converts the compressed sim data to input and target data with
    y being 1 time step ahead of the x data

    returns x,y
    '''
    # path = 'Re_25.npz'  # demo file
    data_file = np.load(path)
    #data_file.files 

    # data is shape (x,y,t)
    # vx = x_velo, vy = y_velo, rho = fluid density
    vx = data_file['vx']
    vy = data_file['vy']
    rho = data_file['rho']
    # stack along the first dimension
    sim = np.stack((vx,vy,rho),0)

    # data gets truncated by 1 due to the lags
    x_data = np.zeros((sim.shape[0],sim.shape[1],sim.shape[2],sim.shape[3]-1))
    y_data = np.zeros((sim.shape[0],sim.shape[1],sim.shape[2],sim.shape[3]-1))
    
    # iterate over all samples
    for i in range(sim.shape[-1]-1):
        x_data[:,:,:,i] = sim[:,:,:,i]
        y_data[:,:,:,i] = sim[:,:,:,i+1]

    assert x_data[:,:,:,1].all()==y_data[:,:,:,0].all() # sanity check
    return x_data,y_data


