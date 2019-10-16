import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import os
import numpy as np
import joblib

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

class SingleClassData(torch.utils.data.Dataset):
    def __init__(self,x,y,cur_trait):
        super(SingleClassData, self).__init__()
        self.x = x
        self.y = y
        self.cur_trait = cur_trait


    def __len__(self):
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                "Data and response have dimension mismatch: X:{}, y:{}".format(
                self.x.shape,self.y.shape))
        else:
            return self.x.shape[0]


    def __getitem__(self,index):
        sample = {'features': self.x[index,:], 'traits': self.y[index, self.cur_trait]}
        return sample


class MLP(nn.Module):
    def __init__(self, input_size=500,hidden_nodes=100,hidden_layers=3):
        super(MLP,self).__init__()
        self.input_layer = nn.Linear(input_size,hidden_nodes)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_nodes,hidden_nodes) for _ in range(hidden_layers)])
        self.output_layer = nn.Linear(hidden_nodes,1)


    def forward(self,x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x


class NestedLinear(nn.Module):
    def __init__(self, input_size=500, hidden_nodes=100,hidden_layers=3):
        super(NestedLinear,self).__init__()
        self.input_layer = nn.Linear(input_size,hidden_nodes)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_nodes,hidden_nodes) for _ in range(hidden_layers)])
        self.output_layer = nn.Linear(hidden_nodes,1)


    def forward(self,x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class MDAD(nn.Module):
    def __init__(self, input_size=500, num_traits= 6, hidden_nodes=100, hidden_layers=3):
        super(MDAD, self).__init__()
        self.input_layer = nn.Linear(input_size,hidden_nodes)
        self.base_net = nn.ModuleList([nn.Linear(hidden_nodes,hidden_nodes) for _ in range(hidden_layers)])
        self.task_lists = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_nodes,hidden_nodes) for _ in range(hidden_layers)]) for _ in range(num_traits)])
        self.outputs = nn.ModuleList([nn.Linear(hidden_nodes,1) for _ in range(num_traits)])

    def forward(self,x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.base_net:
            x = layer(x)
            x = F.relu(x)
        result = []
        for task in self.task_lists:
            cur_x = x
            for layer in task:
                cur_x = F.relu(layer(cur_x))
            result.append(cur_x)
        outputs = []
        for i,out_layer in enumerate(self.outputs):
            tmp_x = out_layer(result[i])
            outputs.append(tmp_x)

        return outputs


def train_MDAD(features, traits, params,verbose=False, save_loss=False):
    num_traits = traits.shape[1]
    input_size = features.shape[1]
    params.update({'input_size': input_size})
    loss_list = [nn.MSELoss() for i in range(num_traits)]
    model = MDAD(**params, num_traits=num_traits).double()
    if torch.cuda.is_available():
        model = model.cuda()
        features = features.cuda()
        traits = traits.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    data = MDADData(features, traits)
    data_train = data
    trainloader = torch.utils.data.DataLoader(data,batch_size=10,shuffle=True)
    i = 0
    losses = []
    while(i < 200):
        for data in trainloader:
            optimizer.zero_grad()
            output = model(data['features'])
            cur_losses = [loss_list[i](output[i][~torch.isnan(data['traits'][:,i])].flatten(),data['traits'][:,i][~torch.isnan(data['traits'][:,i])]) for i in range(len(loss_list))]
            cur_losses = torch.stack(cur_losses)
            cur_loss = cur_losses.sum()
            cur_loss.backward()
            optimizer.step()
        if save_loss or verbose:
            output = model(data_train[:]['features'])
            cur_losses_train = [loss_list[i](output[i][~torch.isnan(data_train[:]['traits'][:,i])].flatten(),data_train[:]['traits'][:,i][~torch.isnan(data_train[:]['traits'][:,i])]) for i in range(len(loss_list))]
            cur_losses_train = torch.stack(cur_losses_train)
            loss_train =  cur_losses_train.sum()
            losses.append(loss_train)

        if verbose:
            print("Loss at epoch {}: {}".format(i,loss_train))
        i += 1
    if save_loss:
        plt.plot(np.arange(i),np.array(losses))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(save_loss)
        return model.cpu(), losses
    return model.cpu()


def run_single_training(trainloader, model, epochs=200):
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    i = 0
    while(i < epochs):
        for data in trainloader:
            optimizer.zero_grad()
            output = model(data['features'])
            cur_loss = loss(output, data['traits'])
            cur_loss.backward()
            optimizer.step()
        i+= 1


def train_single_models(features, traits, params, num_cores = 6):
    n = features.shape[0]
    num_traits = traits.shape[1]
    input_size = features.shape[1]
    params.update({"input_size": input_size})
    linear_models = [NestedLinear(**params).double() for _ in range(num_traits)]
    mlp_models = [MLP(**params).double() for _ in range(num_traits)]
    if torch.cuda.is_available():
        linear_models = [model.cuda() for model in linear_models]
        mlp_models = [model.cuda() for model in mlp_models]
        features = features.cuda()
        traits = traits.cuda()
    data_sets = [SingleClassData(features[~torch.isnan(traits[:,i]),:], traits[~torch.isnan(traits[:,i]),:], i) for i in range(num_traits)]
    trainloaders = [torch.utils.data.DataLoader(data,batch_size=10,shuffle=True) for data in data_sets]
    inputs = [elem for elem in zip(trainloaders,linear_models)] +\
        [elem for elem in zip(trainloaders,mlp_models)]
    with joblib.parallel_backend('loky', n_jobs=num_cores):
        joblib.Parallel()(
            joblib.delayed(run_single_training)(trainloader,model)
            for trainloader,model in inputs)
    linear_models = [linear.cpu() for linear in linear_models]
    mlp_models = [mlp.cpu() for mlp in mlp_models]
    return linear_models, mlp_models