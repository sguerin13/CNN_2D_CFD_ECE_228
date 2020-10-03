import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Utils import data_utils
import matplotlib.pyplot as plt
import time
import pickle

class Conv_Deconv_Net(nn.Module):
    def __init__(self, width, kernel):
        ''' Here we define our layers. All layers are convolutional, including
            those used for size reduction. Not all layers are in all models'''
        super(Conv_Deconv_Net,self).__init__()
        
        self.input = nn.Conv2d(1,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.cc2 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        #First size reduction
        self.cc3 = nn.Conv2d(width,width, kernel_size = 4, stride = 4 ,padding=0)
        self.cc4 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.cc5 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        #Second size reduction
        self.cc6 = nn.Conv2d(width,width, kernel_size = 4, stride = 4 ,padding=0)
        
        self.ic4 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        
        #First size expansion
        self.dc1 = nn.ConvTranspose2d(width,width, kernel_size = 4, stride = 4)
        self.dc2 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.dc3 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        #Second size_expansion
        self.dc4 = nn.ConvTranspose2d(width,width, kernel_size = 4, stride = 4)
        self.dc5 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        
        self.output = nn.Conv2d(width,2, kernel_size = kernel, stride = 1 ,padding=1)
        
        
        
        
        self.str1 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str2 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str3 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str4 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str5 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str6 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str7 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str8 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str9 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str10 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        self.str11 = nn.Conv2d(width,width, kernel_size = kernel, stride = 1 ,padding=1)
        
        
    def forward_straight(self, x):
        '''A 13-layer flat-width (no x/y dimension reduction) model.'''
        
        f = F.relu(self.input(x))
        
        f = F.relu(self.str1(f))
        f = F.relu(self.str2(f))
        f = F.relu(self.str3(f))
        f = F.relu(self.str4(f))
        f = F.relu(self.str5(f))
        f = F.relu(self.str6(f))
        f = F.relu(self.str7(f))
        f = F.relu(self.str8(f))
        f = F.relu(self.str9(f))
        f = F.relu(self.str10(f))
        f = F.relu(self.str11(f))
        
        o = self.output(f)
        return o
    
    def forward_tiny(self, x):
        ''' A 6 layer model with a modest (4x) size reduction '''
        f = F.relu(self.input(x))
        
        #First size reduction:
        f = F.relu(self.cc3(f))
        f = F.relu(self.cc4(f))
        f = F.relu(self.dc3(f))
        #First size expansion:
        f = F.relu(self.dc4(f))
        o = self.output(f)
        return o
        
        
    
    def forward_short(self, x):
        ''' A 9 layer model with two (4x) size reductions '''
        f = F.relu(self.input(x))
        
        #First size reduction:
        f = F.relu(self.cc3(f))
        f = F.relu(self.cc4(f))
        #Second size reduction:
        f = F.relu(self.cc6(f))
        
        f = F.relu(self.ic4(f))
        
        #First size expansion:
        f = F.relu(self.dc1(f))
        f = F.relu(self.dc3(f))
        #First size expansion:
        f = F.relu(self.dc4(f))
        o = self.output(f)
        return o

    def forward_long(self, x):
        ''' A 13 layer model with two (4x) size reductions '''
        f = F.relu(self.input(x))
        
        f = F.relu(self.cc2(f))
        #First size reduction:
        f = F.relu(self.cc3(f))
        f = F.relu(self.cc4(f))
        f = F.relu(self.cc5(f))
        #Second size reduction:
        f = F.relu(self.cc6(f))
        f = F.relu(self.ic4(f))
        
        #First size expansion:
        f = F.relu(self.dc1(f))
        f = F.relu(self.dc2(f))
        f = F.relu(self.dc3(f))
        #First size expansion:
        f = F.relu(self.dc4(f))
        f = F.relu(self.dc5(f))
        
        o = self.output(f)
        return o

def disp_plots(true, out, title = ""):
    ''' Function to plot and save the true and generated velocity fields'''
    true_np = true.cpu().detach().numpy()
    out_np = out.cpu().detach().numpy()
    
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(title)
    axs[0,0].imshow(true_np[0,0,:,:].T)
    axs[0,0].set_title("X velocities ground truth")
    
    axs[0,1].imshow(out_np[0,0,:,:].T)
    axs[0,1].set_title("X velocities model output")
    
    axs[1,0].imshow(true_np[0,1,:,:].T)
    axs[1,0].set_title("Y velocities ground truth")
    
    axs[1,1].imshow(out_np[0,1,:,:].T)
    axs[1,1].set_title("Y velocities model output")
    
    filename = title.replace(' ', '_').lower() + '.png'
    fig.savefig(filename)
      
# Parameters for validation run -- will be held constant for all runs
validation_params = {'num_epochs' : 400,
                     'num_training': 100,
                     'num_validation': 100}

# default model. Instead of truly broad search, a baseline will be assumed and
# deviations tested
default_model = {'lr' : 1e-4,
                 'model_forward': 'forward_long',
                 'width': 64,
                 'kernel': 3,
                 'weight_decay' : 0}

# Defining the models and giving them names
num_models = 11
models = []
for i in range(num_models):
    models.append(default_model.copy()) # number of models

models[1]['lr'] = 1e-5
models[2]['lr'] = 1e-3
models[3]['width'] = 16
models[4]['width'] = 32
models[5]['width'] = 128
models[6]['model_forward'] = 'forward_short'
models[7]['model_forward'] = 'forward_straight'
models[8]['model_forward'] = 'forward_tiny'
models[9]['weight_decay'] = 1e-4
models[10]['weight_decay'] = 1e-5
    
model_names = ['Baseline',
                'Low Learning Rate (1e-5)',
                'High Learning Rate (1e-3)',
                '32 Filters',
                '128 Filters',
                'Short model',
                'Straight model',
                'Tiny Model',
                '16 filters',
                'High weight decay (1e-4)',
                'Low weight decay (1e-5)']


# Loading the training and validation data by choosing random scenes from the dataset
data, sdfs = data_utils.load_array_with_sdf("stokes_triangle_1000.npz")
num_scenes = np.shape(data)[0]
scenes = np.random.choice(num_scenes,
                          validation_params['num_training'] + 
                          validation_params['num_validation'] ,
                          replace = False)
training_scenes = scenes[:validation_params['num_training']]
val_scenes = scenes[validation_params['num_training']:]

# Moving to GPU
data_to = torch.from_numpy(data).float().cuda()
sdfs_to = torch.from_numpy(sdfs).float().cuda()


for model_num, model in enumerate(models):
    
    # Load parameters for run and set up optimizer, loss
    Net = Conv_Deconv_Net(model['width'], model['kernel']).cuda()
    forward_func = getattr(Net, model['model_forward'])
    optimizer = torch.optim.Adam(Net.parameters(), lr = model['lr'],
                                 weight_decay = model['weight_decay'])
    
    distance = nn.MSELoss()
    
    #Define number of epochs to iterate and max index of scenes used for training
    num_epochs = validation_params['num_epochs']
    
    train_loss_by_epoch = []
    val_loss_by_epoch = []
    
    start = time.time()
    # Run training
    for epoch in range(num_epochs):
        train_loss_by_scene = 0
        val_loss_by_scene = 0
        
        for scene in training_scenes:
            sdf = sdfs_to[scene:scene+1,:,:,:]
            true = data_to[scene:scene+1,:,:,:]
            # forward
            output = forward_func(sdf)
            loss = distance(output, true)
            train_loss_by_scene += loss/validation_params['num_training']
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for val_scene in val_scenes:
                sdf = sdfs_to[val_scene:val_scene+1,:,:,:]
                true = data_to[val_scene:val_scene+1,:,:,:]
                output = forward_func(sdf)
                loss = distance(output, true)
                val_loss_by_scene += loss/validation_params['num_validation']
            
        train_loss_by_epoch.append(train_loss_by_scene.item())
        val_loss_by_epoch.append(val_loss_by_scene.item())
        
        # log
        template = 'Model # [{}/{}], epoch [{}/{}],'\
                   ' train loss:{:.2e}, val loss: {:.2e}'
        print(template.format(model_num+1,
                                num_models,
                                epoch+1,
                                num_epochs,
                                train_loss_by_scene,
                                val_loss_by_scene))
        
    elapsed = time.time() - start    
    
    #Plot loss over time
    plt.figure()
    plt.semilogy(train_loss_by_epoch)
    plt.semilogy(val_loss_by_epoch)
    title = "Loss by epoch: " + model_names[model_num]
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training Loss", "Validation loss"])
    name_compact = model_names[model_num].replace(' ', '_').lower()
    filename = 'loss_' + name_compact + '.png'
    plt.savefig(filename)
    
    
    #Show test image example
    scene = training_scenes[0]
    sdf = sdfs_to[scene:scene+1,:,:,:]
    true = data_to[scene:scene+1,:,:,:]
    output = forward_func(sdf)
    disp_plots(true, output, "Training example " + model_names[model_num])
    
    #Show validation image example
    scene = val_scenes[0]
    sdf = sdfs_to[scene:scene+1,:,:,:]
    true = data_to[scene:scene+1,:,:,:]
    output = forward_func(sdf)
    disp_plots(true, output, "Validation example " + model_names[model_num])
    
    
    record = {}
    record['name'] = model_names[model_num]
    record['train_loss'] = train_loss_by_epoch
    record['val_loss'] = val_loss_by_epoch
    record['elapsed'] = elapsed
    
    pickle.dump(record, open( name_compact + ".p", "wb" ) )
    
    del Net
    


