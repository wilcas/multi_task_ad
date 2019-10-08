import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.utils.data
import os
import numpy as np

class MDADData(torch.utils.data.Dataset):
    def __init__(self,x,y):
        super(MDADData, self).__init__()
        self.x = x
        self.y = y


    def __len__(self):
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                "Data and response have dimension mismatch: X:{}, y:{}".format(
                self.x.shape,self.y.shape))
        else:
            return self.x.shape[0]


    def __getitem__(self,index):
        sample = {'features': self.x[index,:], 'traits': self.y[index,:]}
        return sample


class MLP(nn.module):
    def __init__(self, input_size=500):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,input_size)
        self.fc2 = nn.Linear(input_size,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,1)


    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class MDAD(nn.Module):
    def __init__(self, input_size=500, num_traits= 6):
        super(MDAD, self).__init__()
        net_list = [nn.Linear(input_size,input_size)]
        net_list.append(nn.Linear(input_size, 100))
        self.base_net = nn.ModuleList(net_list)
        self.task_lists = nn.ModuleList([nn.ModuleList([nn.Linear(100,100), nn.Linear(100,1)]) for i in range(num_traits)])


    def forward(self,x):
        for layer in self.base_net:
            x = layer(x)
            x = F.relu(x)
        result = []
        for task in self.task_lists:
            cur_x = x
            for layer in task:
                cur_x = F.relu(layer(cur_x))
            result.append(cur_x)
        return result

def train_MDAD(features, traits, verbose=False, plot_loss=False, save_loss=False, use_validation=False, save_model=False):
    n = features.shape[0]
    num_traits = traits.shape[1]
    input_size = features.shape[1]
    loss_list = [nn.MSELoss() for i in range(num_traits)]

    if save_model and os.path.isfile(save_model):
        model = MDAD(input_size, num_traits).double()
        model.load_state_dict(torch.load(save_model))
        model.eval()

        output = model(features)

        cur_losses = [loss_list[i](output[i].flatten(),traits[:,i]) for i in range(len(loss_list))]
        loss_tot = torch.stack(cur_losses).sum()
        return model, loss_tot
    else:
        model = MDAD(input_size,num_traits).double()
        if torch.cuda.is_available():
            model = model.cuda()
            features = features.cuda()
            traits = traits.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        data = MDADData(features, traits)
        if use_validation:
            train_sz = int(n * 0.8)
            val_sz = n - train_sz
            data_train, data_val = torch.utils.data.random_split(data,(train_sz,val_sz))
            trainloader = torch.utils.data.DataLoader(data_train,batch_size=10,shuffle=True)
        else:
            trainloader = torch.utils.data.DataLoader(data,batch_size=10,shuffle=True)
        i = 0
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        losses = []
        while(i < 200):
            plt.clf()
            for data in trainloader:
                optimizer.zero_grad()
                output = model(data['features'])
                cur_losses = [loss_list[i](output[i][~torch.isnan(data['traits'][:,i])].flatten(),data['traits'][:,i][~torch.isnan(data['traits'][:,i])]) for i in range(len(loss_list))]
                cur_losses = torch.stack(cur_losses)
                cur_loss = cur_losses.sum()
                cur_loss.backward()
                optimizer.step()
            if plot_loss or save_loss:
                if use_validation:
                    output_train = model(data_train[:]['features'])
                    output_val = model(data_val[:]['features'])
                    cur_losses_train = [loss_list[i](output_train[i][~torch.isnan(data_train[:]['traits'][:,i])].flatten(),data_train[:]['traits'][:,i][~torch.isnan(data_train[:]['traits'][:,i])]) for i in range(len(loss_list))]
                    cur_losses_val = [loss_list[i](output_val[i][~torch.isnan(data_val[:]['traits'][:,i])].flatten(),data_val[:]['traits'][:,i][~torch.isnan(data_val[:]['traits'][:,i])]) for i in range(len(loss_list))]
                    cur_losses_train = torch.stack(cur_losses_train)
                    cur_losses_val = torch.stack(cur_losses_val)
                    loss_train =  cur_losses_train.sum()
                    loss_val = cur_losses_val.sum()
                    losses.append((loss_train.detach(), loss_val.detach()))
                    # if plot_loss:
                    #     writer.add_scalar(train_tag,loss_train.item(),global_step=i)
                    #     writer.add_scalar(val_tag,loss_val.item(),global_step=i)
                else:
                    output = model(data['features'])
                    cur_losses_train = [loss_list[i](output_train[i].flatten(),data_train[:]['traits'][:,i]) for i in range(len(loss_list))]
                    cur_losses_train = torch.stack(cur_losses_train)
                    loss_train =  cur_losses_train[~torch.isnan(cur_losses_train)].sum()
                    losses.append(loss_train)
                    # if plot_loss:
                    #     writer.add_scalar(train_tag,loss_train.item(), global_step=i)
            else:
                cur_losses = [loss_list[i](output[i],data['traits'][i]) for i in range(len(loss_list))]
                cur_losses = torch.stack(cur_losses)
                cur_losses = cur_losses[~torch.isnan(cur_losses)].sum()
            scheduler.step(loss_train)

            if verbose:
                if use_validation:
                    print("Loss at epoch {}: train: {}, validation: {}".format(i,loss_train, loss_val))
                else:
                    print("Loss at epoch {}: {}".format(i,loss_tot))
            # if i > 50:
            #     avg_dec = sum([losses[i-j-1][0]-losses[i-j-2][0] for j in range(50)]) / 50
            #     if plot_loss:
            #         writer.add_scalar('avg_dec',avg_dec,global_step=i)
            #     if avg_dec <= 0 and losses[i-1][1] >= losses[i-2][1]:
            #         improved = False
            i += 1
        # if plot_loss:
        #     writer.close()
        if save_loss:
            if use_validation:
                losses_train = [value for (value,_) in losses]
                losses_val = [value for (_,value) in losses]
                plt.plot(np.arange(i),np.array(list(losses_train)), label = "Training Loss")
                plt.plot(np.arange(i),np.array(list(losses_val)), label = "Validation Loss")
                plt.legend()
            else:
                plt.plot(np.arange(i),np.array(losses))
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
            plt.savefig(save_loss)
        if save_model:
            torch.save(model.state_dict(), save_model)
    output = model(features)

    cur_losses = [loss_list[i](output[i].flatten(),traits[:,i]) for i in range(len(loss_list))]
    loss_tot = torch.stack(cur_losses).sum()
    return model.cpu(), loss_tot


def integrated_gradients(model,input):
    pass