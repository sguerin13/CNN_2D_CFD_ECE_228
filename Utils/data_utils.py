import numpy as np
import torch
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader


def sim_to_x_y(path):
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

    # swap axes so time is the first axis
    sim.swapaxes(0,3) # swap time and data type
    sim.swapaxes(1,3) # swap x and time
    sim.swapaxes(2,3) # swap x and y

    # data should now be (time,data_type,x,y)

    # data gets truncated by 1 due to the lags
    x_data = np.zeros((sim.shape[0]-1,sim.shape[1],sim.shape[2],sim.shape[3]))
    y_data = np.zeros((sim.shape[0]-1,sim.shape[1],sim.shape[2],sim.shape[3]))
    
    # iterate over all samples
    for i in range(sim.shape[0]-1):
        x_data[i,:,:,:] = sim[i,:,:,:]
        y_data[i,:,:,:] = sim[i+1,:,:,:]

    assert x_data[1,:,:,:].all()==y_data[0,:,:,:].all() # sanity check
    return x_data,y_data


def np_to_torch_dataloader(x_in,target,batch = 1):
    '''
    converts our numpy arrays into a dataloader object by first
    creating pytorch tensors and then creating a dataloader object
    from said tensors

    '''
    x_in_tensor = torch.from_numpy(x_in)
    target_tensor = torch.from_numpy(target)
    train_data = data.TensorDataset(x_in_tensor,target_tensor)
    train_data_loader = dataloader.DataLoader(train_data,batch_size = batch,shuffle = False)
    return train_data_loader
