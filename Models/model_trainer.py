'''
Class to handle all of the training tasks

Decouples the Training from the nn.Module definition


'''
import torch
import torch.nn as nn
import numpy as np

class Trainer(nn.Module):
    def __init__(self,model,file_name):
        super(Trainer,self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_name = file_name
    def train_net(self,dataset,lr = 5e-4,epochs = 500,save_model=False):

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr = lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.2,patience =50)
        self.loss_vec = []
        self.best_loss = 100
        for eps in range(epochs):
            
            for idx,(data) in enumerate(dataset): # data_loader inputs are (x,param_vec,targets)
                
                x = data[0].to(self.device)
                Re = data[1].to(self.device)
                mask = data[2].to(self.device)
                y = data[3].to(self.device)
                # forward pass
                output = self.model.forward(x,Re)
                loss = criterion(output,t)
                self.loss_vec.extend([loss.cpu().detach().numpy()])
                if loss < self.best_loss:
                    self.best_model_state = self.state_dict()
                    self.best_loss = loss
                
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # scheduler.step(loss)
            print('epoch %i / %i, loss: %f' % (eps+1,epochs,loss.data))
            print('Best Loss: ' % self.best_loss)
    
        if save_model == True:
            file_str = './saved_models/' + self.file_name
            torch.save(self.best_model_state,file_str)
