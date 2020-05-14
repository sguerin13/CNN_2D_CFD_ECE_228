import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Utils import data_utils
import matplotlib.pyplot as plt

class Conv_Deconv_Net(nn.Module):
    def __init__(self):
        super(Conv_Deconv_Net,self).__init__()
        
        self.cc1 = nn.Conv2d(1,64, kernel_size = 3, stride = 1 ,padding=1)
        self.cc2 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        #First size reduction
        self.cc3 = nn.Conv2d(64,64, kernel_size = 4, stride = 4 ,padding=0)
        self.cc4 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        self.cc5 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        #Second size reduction
        self.cc6 = nn.Conv2d(64,64, kernel_size = 4, stride = 4 ,padding=0)
        
        self.ic4 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        
        #First size expansion
        self.dc1 = nn.ConvTranspose2d(64,64, kernel_size = 4, stride = 4)
        self.dc2 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        self.dc3 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        #Second size_expansion
        self.dc4 = nn.ConvTranspose2d(64,64, kernel_size = 4, stride = 4)
        self.dc5 = nn.Conv2d(64,64, kernel_size = 3, stride = 1 ,padding=1)
        self.dc6 = nn.Conv2d(64,2, kernel_size = 3, stride = 1 ,padding=1)
        

    def forward2(self, x):
        f = F.relu(self.cc1(x))
        f = F.relu(self.cc2(f))
        f = F.relu(self.cc3(f))
        f = F.relu(self.cc4(f))
        f = F.relu(self.cc5(f))
        f = F.relu(self.cc6(f))
        
        f = F.relu(self.ic4(f))
        
        f = F.relu(self.dc1(f))
        f = F.relu(self.dc2(f))
        f = F.relu(self.dc3(f))
        f = F.relu(self.dc4(f))
        f = F.relu(self.dc5(f))
        o = self.dc6(f)
        return o

def disp_plots(true, out, title = ""):
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
      
  
Net = Conv_Deconv_Net().cuda()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(Net.parameters())

#Define number of epochs to iterate and max index of scenes used for training
num_epochs = 100
scenes = 2
data, sdfs = data_utils.load_array_with_sdf("stokes_triangle_1000.npz")
data_to = torch.from_numpy(data).float().cuda()
sdfs_to = torch.from_numpy(sdfs).float().cuda()

loss_by_epoch = []
# Run training
for epoch in range(num_epochs):
    loss_by_scene = 0
    for scene in range(scenes):
        sdf = sdfs_to[scene:scene+1,:,:,:]
        true = data_to[scene:scene+1,:,:,:]
        # forward
        output = Net.forward2(sdf)
        loss = distance(output, true)
        loss_by_scene += loss/scenes
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_by_epoch.append(loss_by_scene)
    # log
    print('epoch [{}/{}], loss:{:e}'.format(epoch+1, num_epochs, loss_by_scene))


#Plot loss over time
plt.figure()
plt.semilogy(loss_by_epoch)
plt.title("Loss by epoch")

#Show test image example
disp_plots(true, output, "Test example")

#Show validation image example
scene = scenes+1
sdf = sdfs_to[scene:scene+1,:,:,:]
true = data_to[scene:scene+1,:,:,:]
output = Net.forward2(sdf)
disp_plots(true, output, "Validation example")


