
'''

Data loading and preparation functionality for prediction of 
open channel fluid flows using Deep Learning.

Functions cover:
    - Data Loading from compressed files
    - Data Parsing into X,Y pairs
    - Normalization
    - DataLoader Creation

'''

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler as STD,MinMaxScaler as MinMax
from pickle import dump,load

'''
DATA LOADING SECTION
- function:
    - load_array
    - load_array_with_sdf
    - load_array_with_sdf_and_params
    - load_array_with_sdf_mask_Re
'''

def load_array(path):
    '''
    helper function that loads the compressed numpy file into an array
    and orders the data into a 4-d array that is
    (time_step,metric(vx,vy,rho),y,x)
    
    putting the data in this form will make it easier for processing
    in pytorch
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
    sim = np.swapaxes(sim,0,3) # swap time and data type
    sim = np.swapaxes(sim,1,3) # swap x and time

    # data should now be (time,data_type,y,x)
    return sim

def load_array_with_sdf(path):
    '''
    helper function that loads the compressed numpy file into an array
    and orders the data into a 4-d array that is
    (time_step,metric(vx,vy,rho),y,x). Also load the 2D Array for the SDF
    putting the data in this form will make it easier for processing
    in pytorch
    '''

    # path = 'Re_25.npz'  # demo file
    data_file = np.load(path)
    #data_file.files 

    # data is shape (x,y,t)
    # vx = x_velo, vy = y_velo, rho = fluid density
    vx = data_file['vx']
    vy = data_file['vy']
    rho = data_file['rho']
    sdf = data_file['sdf']
    
    # stack along the first dimension
    sim = np.stack((vx,vy,rho),0)

    # swap axes so time is the first axis
    sim = np.swapaxes(sim,0,3) # swap time and data type
    sim = np.swapaxes(sim,1,3) # swap x and time

    # data should now be (time,data_type,y,x)
    return sim,sdf


def load_array_with_sdf_and_params(path):
    '''
    - helper function that loads the compressed numpy file into an array
    and orders the data into a 4-d array that is
    (time_step,metric(vx,vy,rho),y,x). 
    
    - Also load the 2D Array for the SDF putting the data in this form 
    will make it easier for processing in pytorch

    - Loads simulation parameters into a vector

    '''
    # path = 'Re_25.npz'  # demo file
    data_file = np.load(path)
    #data_file.files 

    # data is shape (x,y,t)
    # vx = x_velo, vy = y_velo, rho = fluid density
    vx = data_file['vx']
    vy = data_file['vy']
    rho = data_file['rho']
    sdf = data_file['sdf'].T # need to make it the same shape as the grid
    mask = data_file['mask'].T
    Re  = data_file['Re'].flatten()[0]
    v_o = data_file['Vo'].flatten()[0]
    rho_o = data_file['rhoo'].flatten()[0]
    
    # create the parameter vector
    param_vec = np.array([Re,v_o,rho_o])


    # stack along the first dimension
    sim = np.stack((vx,vy,rho),0)

    # swap axes so time is the first axis
    sim = np.swapaxes(sim,0,3) # swap time and data type
    sim = np.swapaxes(sim,1,3) # swap x and time

    # data should now be (time,data_type,y,x)
    return sim,sdf,param_vec


def load_array_with_sdf_mask_Re(path):
    '''
    - helper function that loads the compressed numpy file into an array
    and orders the data into a 4-d array that is
    (time_step,metric(vx,vy,rho),y,x). 
    
    - Also load the 2D Array for the SDF putting the data in this form 
    will make it easier for processing in pytorch

    - Loads simulation parameters into a vector

    '''
    # path = 'Re_25.npz'  # demo file
    data_file = np.load(path)
    #data_file.files 

    # data is shape (x,y,t)
    # vx = x_velo, vy = y_velo, rho = fluid density
    vx = data_file['vx']
    vy = data_file['vy']
    rho = data_file['rho']
    sdf = data_file['sdf'].T # need to make it the same shape as the grid
    presence_map = data_file['mask'].T
    mask = ~presence_map # invert the presence map to turn it into a true mask
    Re  = data_file['Re'].flatten()[0]


    # stack along the first dimension
    sim = np.stack((vx,vy,rho),0)

    # swap axes so time is the first axis
    sim = np.swapaxes(sim,0,3) # swap time and data type
    sim = np.swapaxes(sim,1,3) # swap x and time

    # data should now be (time,data_type,y,x)
    return sim,sdf,mask,Re

'''

X,Y PAIR CREATION
- functions:
    - sim_to_x_y
    - sim_to_seqx_y
    - sim_to_seqx_seqy
    - sim_to_auto_enc_data

'''

def sim_to_x_y(sim):

    '''
    converts the uncompress simulation array to input and target data with
    y being 1 time step ahead of the x data
    returns x,y
    '''
    
    # data gets truncated by 1 due to the lags
    x_data = np.zeros((sim.shape[0]-1,sim.shape[1],sim.shape[2],sim.shape[3]))
    y_data = np.zeros((sim.shape[0]-1,sim.shape[1],sim.shape[2],sim.shape[3]))
    
    # iterate over all samples
    for i in range(sim.shape[0]-1):
        x_data[i,:,:,:] = sim[i,:,:,:]
        y_data[i,:,:,:] = sim[i+1,:,:,:]

    assert x_data[1,:,:,:].all()==y_data[0,:,:,:].all() # sanity check
    return x_data,y_data


def param_to_x_y(param):
    
    '''
   shortens the length of the parameter vector to be the same length of the input data

    '''
    #print(param.shape)
    # data gets truncated by 1 due to the lags
    x_data = np.zeros((param.shape[0]-1,param.shape[1]))
    
    # iterate over all samples
    for i in range(param.shape[0]-1):
        x_data[i,:] = param[i,:]
        
    return x_data


def sim_to_seqx_y(sim,length = 5,allow_empty=False):
    '''
    converts the uncompressed simulation array to a sequence of 'length' scenes as the input X data
    and a single output scene y. This will be used for loading data in sequence based models

    'allow_empty' allows for sequence inputs that have a few 'empty' scenes because this sample is
    take from the first steps of a simulation (< 'length' steps in). For example: If we are two
    timesteps into the simulation the x_data would be ([empty,empty,empty,1,2]). The empty scenes
    are filled with zeros
    '''


    if allow_empty == True:

        '''
        ** assuming a simuation of 250 timesteps and length = 5
        X = (empty,empty,empty,empty,1), ........ ,(245,246,247,248,249)
        Y = (2),.....,(250)
        
        Each input of the x data is now of shape (length,3,y,x)
        so in total X becomes a 5D array with (num_examples,length,3,y,x)
        '''

        x_data = np.zeros((sim.shape[0]-1,length,sim.shape[1],sim.shape[2],sim.shape[3]))
        y_data = np.zeros((sim.shape[0]-1,sim.shape[1],sim.shape[2],sim.shape[3]))

        # iterate over all samples
        for i in range(sim.shape[0]-1): # 0 to num_time_steps

            '''
            since we are leaving empty matrices some of the entries in the sequence
            we have to progressively add data points to the sequence until we are pulling 
            from all simulation scenes to fill our sequence. This happens once we reach
            'length' iterations through the dataset filling in the sequences.

            Por ejemplo:

            iter 0:
                (empty,empty,empty,empty,t_0)
            iter 1:
                (empty,empty,empty,t_0,t_1)
            iter 2:
                (empty,empty,t_0,t_1,t_2)
                ...
            iter 4:
                (t_0,t_1,t_2,t_3,t_4)

            '''

            # create our leading entries that have empty arrays for the first (length - 1) entries
            # before moving into the default sequence generation steps
            if i < (length-1):
                
                for j in range(i+1):
                    
                    index = -(i+1)+j
                    x_data[i,index,:,:,:] = sim[j,:,:,:]

                y_data[i,:,:,:] = sim[i+1,:,:,:]
            
                # sanity check
                assert x_data[i,-1,:,:,:].all() == sim[i,:,:,:].all() # make sure that the last element is equal to sim at t_i
                assert x_data[i,-(i+1),:,:].all() == sim[0,:,:,:].all() # make sure the first non-zero is equal to sim at t_0

            else:
                
                # we have to shift our indexing through the sim data back by 'length - 1' since we have already filled up the output array with 
                # 'length' number of entries


                start_ind = i - (length - 1)
                end_ind = i + 1 # + length - length
                max_ind = sim.shape[0] - 2
                
                x_data[i,:,:,:,:] = sim[start_ind:end_ind,:,:,:] # sequence of 'length'
                y_data[i,:,:,:] = sim[end_ind,:,:,:]

                if i == (max_ind):
                    assert end_ind == sim.shape[0]-1
 


    else: # default, there will be no empty matrices
        '''
        ** assuming a simuation of 250 timesteps and length = 5
        X = (1,2,3,4,5), ........ ,(245,246,247,248,249)
        Y = (6),.....,(250)
        
        Each input of the x data is now of shape (length,3,y,x)
        so in total X becomes a 5D array with (num_examples,length,3,y,x)
        '''

        x_data = np.zeros((sim.shape[0]-length,length,sim.shape[1],sim.shape[2],sim.shape[3]))
        y_data = np.zeros((sim.shape[0]-length,sim.shape[1],sim.shape[2],sim.shape[3]))

        # iterate over all samples
        for i in range(sim.shape[0]-length):
            '''
            sanity check

            assuming length = 5 and the scene has 250 time steps
            
            @ i = 0:
                - start_ind = 0,end_in = 5 -> we pulls array indexes 0 - 4 for x, index 5 for y
            @ i = 244
                - start index = 244, end index = 249, we get array indexes of 244-248, index 249 for y

            '''
            start_ind = i
            end_ind = i+length
            max_ind = sim.shape[0] -length - 1 # max array index covered in the range (250 - 5 - 1) = 244


            x_data[i,:,:,:,:] = sim[start_ind:end_ind,:,:,:] # sequence of 'length'
            y_data[i,:,:,:] = sim[end_ind,:,:,:]

            # sanity checkes
            if i == 0:
                assert start_ind == 0
                assert end_ind == length
            if i == (max_ind):
                assert start_ind == max_ind
                assert end_ind == sim.shape[0]-1

    return x_data,y_data


def sim_to_seqx_seqy(sim,x_len=5,y_len=2,allow_empty = False):

    '''
    converts the uncompressed simulation array to a sequence of 'length' scenes as the input X data
    and a sequence of output scenes y. This will be used for loading data in sequence based models

    'allow_empty' allows for sequence inputs that have a few 'empty' scenes because this sample is
    take from the first steps of a simulation (< 'length' steps in). For example: If we are two
    timesteps into the simulation the x_data would be ([empty,empty,empty,1,2]). The empty scenes
    are filled with zeros
    '''


    if allow_empty == True:

        '''
        ** assuming a simuation of 250 timesteps and length = 5
        X = (empty,empty,empty,empty,1), ........ ,(244,245,246,247,248)
        Y = (2,3),.....,(249,250)
        
        Each input of the x data is now of shape (length,3,y,x)
        so in total X becomes a 5D array with (num_examples,length,3,y,x)
        '''

        # y_length limits the overall array length
        x_data = np.zeros((sim.shape[0]-y_len,x_len,sim.shape[1],sim.shape[2],sim.shape[3]))
        y_data = np.zeros((sim.shape[0]-y_len,y_len,sim.shape[1],sim.shape[2],sim.shape[3]))

        # iterate over all samples
        for i in range(sim.shape[0]-y_len): # 0 to num_time_steps

            '''
            since we are leaving empty matrices some of the entries in the sequence
            we have to progressively add data points to the sequence until we are pulling 
            from all simulation scenes to fill our sequence. This happens once we reach
            'length' iterations through the dataset filling in the sequences.

            Por ejemplo:

            iter 0:
                (empty,empty,empty,empty,t_0)
            iter 1:
                (empty,empty,empty,t_0,t_1)
            iter 2:
                (empty,empty,t_0,t_1,t_2)
                ...
            iter 4:
                (t_0,t_1,t_2,t_3,t_4)

            '''

            # create our leading entries that have empty arrays for the first (length - 1) entries
            # before moving into the default sequence generation steps
            if i < (x_len-1):
                
                for j in range(i+1):
                    
                    index = -(i+1)+j
                    x_data[i,index,:,:,:] = sim[j,:,:,:]

                y_data[i,:,:,:,:] = sim[i+1:(i+1+y_len),:,:,:]
            
                # sanity check
                assert x_data[i,-1,:,:,:].all() == sim[i,:,:,:].all() # make sure that the last element is equal to sim at t_i
                assert x_data[i,-(i+1),:,:].all() == sim[0,:,:,:].all() # make sure the first non-zero is equal to sim at t_0

            else:
                
                # we have to shift our indexing through the sim data back by 'length - 1' since we have already filled up the output array with 
                # 'length' number of entries


                start_ind = i - (x_len - 1)
                x_end_ind = i + 1 # + x_len - x_len
                y_end_ind = x_end_ind + y_len
                max_ind = sim.shape[0] - y_len - 1
                
                x_data[i,:,:,:,:] = sim[start_ind:x_end_ind,:,:,:] # sequence of 'length'
                y_data[i,:,:,:,:] = sim[x_end_ind:y_end_ind,:,:,:]

                if i == (max_ind):
                    assert x_end_ind == sim.shape[0]-y_len
 

    else: # default, there will be no empty matrices
        '''
        ** assuming a simuation of 250 timesteps and length = 5
        X = (1,2,3,4,5), ........ ,(245,246,247,248,249)
        Y = (6),.....,(250)
        
        Each input of the x data is now of shape (length,3,y,x)
        so in total X becomes a 5D array with (num_examples,length,3,y,x)
        '''

        # in this case we lose information on both ends
        x_data = np.zeros((sim.shape[0]-x_len-(y_len-1),x_len,sim.shape[1],sim.shape[2],sim.shape[3]))
        y_data = np.zeros((sim.shape[0]-x_len-(y_len-1),y_len,sim.shape[1],sim.shape[2],sim.shape[3]))

        # iterate over all samples
        for i in range(sim.shape[0]-x_len-(y_len-1)):
            '''
            sanity check

            assuming length = 5 and the scene has 250 time steps
            
            @ i = 0:
                - start_ind = 0,end_in = 5 -> we pulls array indexes 0 - 4 for x, index 5 for y
            @ i = 244
                - start index = 244, end index = 249, we get array indexes of 244-248, index 249 for y

            '''


            start_ind = i
            x_end_ind = i+x_len
            y_end_ind = x_end_ind + y_len
            max_ind = sim.shape[0]-x_len-y_len # max array index covered in the range
            
            x_data[i,:,:,:,:] = sim[start_ind:x_end_ind,:,:,:] # sequence of 'length'
            y_data[i,:,:,:,:] = sim[x_end_ind:y_end_ind,:,:,:]

            # sanity checkes
            if i == 0:
                assert start_ind == 0
                assert x_end_ind == x_len
            if i == (max_ind):
                assert start_ind == max_ind
                assert y_end_ind == sim.shape[0]

    return x_data,y_data

def sim_to_auto_enc_data(sim):
    '''
    converts the simulation data to time-stacked dataset for training autoencoders
    where x = y = sim
    '''
    x_data = sim
    y_data = sim # target is the input
    return x_data,y_data
    

'''

DATA NORMALIZATION SECTION
- functions:
    - normalize_sim_data
    - normalize_data -- for non-simulation data
    - partially_fit_sim_scaler
    - save_std_scaler
    - load_std_scaler
    - normalize_w_scaler

'''

def normalize_sim_data(sim,std = True):
    '''
    - This function parses through the simulation data and creates separate scalars
    for the velocity and the density data

    - jointly normalize vx, vy so that they are on the same normalization scale
    - normalize rho on it's own scale
    
    - Create a temporary variable the groups v_x,v_y into a single row and learn
    the scaling factor, it is then applied to the the array, same process to rho

    - Returns the scaler objects so they can be saved if desired
    '''
    # get array shapes for layer
    velo_shape = sim[:,:2,:,:].shape
    rho_shape  = sim[:,-1,:,:].shape

    temp_velos = sim[:,:2,:,:].copy() # copy 
    temp_velos = temp_velos.reshape((-1,1)) # flatten
    temp_rho   = sim[:,-1,:,:].copy() 
    temp_rho   = temp_rho.reshape((-1,1))
    
    if std==True:
        # declare objects
        velo_scaler = STD()
        rho_scaler  = STD()
    else:
        velo_scaler = MinMax()
        rho_scaler  = MinMax()

    # fit to the data
    scaled_velo = velo_scaler.fit_transform(temp_velos) 
    scaled_rho  = rho_scaler.fit_transform(temp_rho)

    # move back into (t,n,y,x) shapes
    scaled_velo = scaled_velo.reshape(velo_shape)
    scaled_rho =  scaled_rho.reshape(rho_shape)

    # expand the dimension of rho to make it a 4D array
    scaled_rho = np.expand_dims(scaled_rho,axis=1)

    # stack along the second dimension
    sim_scaled = np.concatenate((scaled_velo,scaled_rho),axis=1)

    #check to make sure the shape is correct
    assert sim_scaled.shape == sim.shape

    # return scalar objects in case you want to save them for test time
    return sim_scaled,velo_scaler,rho_scaler


def normalize_data(inputs,SDF = False):
    '''
    
    - function for normalizing data that doesn't pertain to direct simulation outputs

    - special functionality for the SDF since it is for a 2D array

    '''

    if SDF == True:
        # get array shapes for layer
        SDF_shape = inputs.shape

        SDF_copy   = inputs.copy() # copy 
        SDF_values = SDF_copy.reshape((-1,1)) # flatten

        # declare objects
        SDF_scaler = STD()

        # fit to the data
        scaled_SDF = SDF_scaler.fit_transform(SDF_values) 

        # move back into (t,n,y,x) shapes
        scaled_SDF = scaled_SDF.reshape(SDF_shape)

        return scaled_SDF,SDF_scaler
    else:
        # we are operating on the parameters
        param_copy = inputs.copy()
        param_scaler = STD()
        scaled_params = param_scaler.fit_transform(param_copy)

        return scaled_params,param_scaler


def partially_fit_sim_scaler(velo_scaler,rho_scaler,data):

    '''
    used to build up scalars over multiple scenes of data
    '''
    # get array shapes for layer
    velo_shape = data[:,:2,:,:].shape
    rho_shape  = data[:,-1,:,:].shape

    temp_velos = data[:,:2,:,:].copy() # copy 
    temp_velos = temp_velos.reshape((-1,1)) # flatten
    temp_rho   = data[:,-1,:,:].copy() 
    temp_rho   = temp_rho.reshape((-1,1))

    # fit to the data
    velo_scaler.partial_fit(temp_velos) 
    rho_scaler.partial_fit(temp_rho)

    del temp_rho
    del temp_velos
    del velo_shape
    del rho_shape

    return velo_scaler,rho_scaler

def partially_fit_data_scaler(scaler,data,SDF=False):

    if SDF == True:
        # get array shapes for layer
        SDF_shape = data.shape

        SDF_copy   = data.copy() # copy 
        SDF_copy = SDF_copy.reshape((-1,1)) # flatten

        # fit to the data
        scaler.partial_fit(SDF_copy) 
        
        del SDF_copy
        del SDF_shape

        return scaler
    
    else: # simulation parameters
        
        # params are a single row
        param_copy = data.copy()
        param_copy = param_copy.reshape((1,-1))
        scaler.partial_fit(param_copy)

        del param_copy

        return scaler


def save_std_scaler(scaler,file_name):

    dump(scaler,open(file_name,'wb'))


def load_std_scaler(file_name):

    scaler = load(open(file_name,'rb'))

    return scaler

def normalize_sim_w_scaler(sim,velo_scaler,rho_scaler):
    
    '''
    normalize the simulation data w/ a pre-trained scaler

    '''

        # get array shapes for layer
    velo_shape = sim[:,:2,:,:].shape
    rho_shape  = sim[:,-1,:,:].shape

    temp_velos = sim[:,:2,:,:].copy() # copy 
    temp_velos = temp_velos.reshape((-1,1)) # flatten
    temp_rho   = sim[:,-1,:,:].copy() 
    temp_rho   = temp_rho.reshape((-1,1))


    # fit to the data
    scaled_velo = velo_scaler.fit_transform(temp_velos) 
    scaled_rho  = rho_scaler.fit_transform(temp_rho)

    # move back into (t,n,y,x) shapes
    scaled_velo = scaled_velo.reshape(velo_shape)
    scaled_rho =  scaled_rho.reshape(rho_shape)

    # expand the dimension of rho to make it a 4D array
    scaled_rho = np.expand_dims(scaled_rho,axis=1)

    # stack along the second dimension
    sim_scaled = np.concatenate((scaled_velo,scaled_rho),axis=1)

    #check to make sure the shape is correct
    assert sim_scaled.shape == sim.shape

    # return scalar objects in case you want to save them for test time
    return sim_scaled


def normalize_data_w_scaler(data,scaler,SDF=False):

    '''
    - function to normalize the data with a loaded scaler object from sklearn
    - returns array of the same shape
        
    '''
    if SDF == True:


        data_shape = data.shape
        temp_data = data.copy()
        temp_data = temp_data.reshape((-1,1)) # flatten

        scaled_data = scaler.transform(temp_data) 

        # move back into (t,n,y,x) shapes
        scaled_data = scaled_data.reshape(data_shape)

    else:
        # parameter vector (1 x num_params)

        temp_data = data.copy()
        temp_data = temp_data.reshape(1,-1) # need to flatten into a row_vector
        scaled_data = scaler.transform(temp_data)

    return scaled_data

'''

TORCH DATA LOADING SECTION
- functions:
    - np_to_torch_dataloader

'''

def np_to_torch_dataloader(x_in,mask,target,batch = 1,Re = None):
    '''
    converts our numpy arrays into a dataloader object by first
    creating pytorch tensors and then creating a dataloader object
    from said tensors
    
    - assumes SDF is embedded in the x_in data
    - provides functionality to include parameter data
    
    '''
    if Re is not None:
        x_in_tensor = torch.from_numpy(x_in)
        Re_tensor = torch.from_numpy(Re)
        mask_tensor = torch.from_numpy(mask)
        target_tensor = torch.from_numpy(target)
        tensor_data = torch.utils.data.TensorDataset(x_in_tensor,Re_tensor,mask_tensor,target_tensor)
        data_loader = torch.utils.data.DataLoader(tensor_data,batch_size = batch,shuffle = True,drop_last=True)
        return data_loader
    
    else:
        
        x_in_tensor = torch.from_numpy(x_in)
        mask_tensor = torch.from_numpy(mask)
        target_tensor = torch.from_numpy(target)
        tensor_data = torch.utils.data.TensorDataset(x_in_tensor,mask_tensor,target_tensor)
        data_loader = torch.utils.data.DataLoader(tensor_data,batch_size = batch,shuffle = True,drop_last=True)
        return data_loader