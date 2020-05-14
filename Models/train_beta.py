import torch
import numpy as np
from sklearn.preprocessing import StandardScaler as STD
import Utils.data_utils as du
# from EncoderDecoderSDF import Encoder,Decoder,EncoderDecoder
from SimpleEncDecRevFilters import Encoder,Decoder,EncoderDecoder
import matplotlib.pyplot as plt
import Utils.data_prep as dprep
import os
from model_trainer import Trainer

torch.cuda.empty_cache()
# load scalers
direct = os.getcwd()
path = direct + '/../../Data/data/Circle_Trans_30fps/'
# dprep.build_scalers(path)

velo_scaler = du.load_std_scaler(direct + './Scalers/velo_scaler_circles_5-10-20.pkl')
rho_scaler  = du.load_std_scaler('./Scalers/rho_scaler_circles_5-10-20.pkl')
sdf_scaler  = du.load_std_scaler('./Scalers/sdf_scaler_circles_5-10-20.pkl')
Re_scaler  = du.load_std_scaler('./Scalers/Re_scaler_circles_5-10-20.pkl')

# scaled data returns (x,Re_#,mask,y)
scaled_data = dprep.build_scaled_dataset(path=path,velo_scaler=velo_scaler,rho_scaler=rho_scaler,
                                    sdf_scaler=sdf_scaler,Re_scaler=Re_scaler,
                                    mode = 'x to y',num_scenes=5)

x = scaled_data[0]
Re = scaled_data[1]
mask = scaled_data[2]
y = scaled_data[3]

data_loader = du.np_to_torch_dataloader(x,mask,y,batch = 50,Re = Re)
# model = EncoderDecoder().double()
# model.cuda()
# file_name = 'ConvEncDec_BN_X_Y_5-12_Overfit_Attempt.pth'
# trainer = Trainer(model,file_name)
# trainer.train_net(data_loader,lr = 5e-4,epochs = 500,save_model=True)

# plt.figure(figsize = (20,10))
# plt.plot(trainer.loss_vec)
# plt.title('training loss')
# plt.xlabel('steps')
# plt.ylabel('loss')
# plt.yscale('log')
# plt.show()

# ### test output ### 
# import matplotlib.pyplot as plt
# x_demo = torch.from_numpy(scaled_data[0][0])
# x_demo = x_demo.unsqueeze(0)
# param_demo = torch.from_numpy(scaled_data[1][0])
# param_demo = param_demo.view((1,3))
# trainer.model.cpu()
# trainer.model.eval()
# y = trainer.model.forward(x_demo,param_demo)

# y_numpy = y.detach().numpy()

# i = 0

# plt.figure(figsize=(10,10))
# plt.suptitle('Density',y=.62)
# plt.subplot(1,3,1)
# plt.imshow(scaled_data[2][0,i,:,:])
# plt.title('Target')
# plt.subplot(1,3,2)
# plt.imshow(y_numpy[0,i,:,:],vmin=np.min(scaled_data[2][0,i,:,:]),vmax=np.max(scaled_data[2][0,i,:,:]))
# plt.title('Prediction')
# plt.subplot(1,3,3)
# plt.imshow(scaled_data[2][0,i,:,:] - y_numpy[0,i,:,:])
# plt.title('Difference')
# plt.show()
