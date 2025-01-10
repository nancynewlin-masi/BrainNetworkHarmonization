import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
import torch.nn.functional
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, layers=[512,512,512]):
        super(MLP, self).__init__()
        self.mlp_list = nn.ModuleList()

        self.mlp_list.append(nn.Linear(in_dim,layers[0]))
        self.mlp_list.append(torch.nn.ReLU())
        #self.mlp_list.append(torch.nn.Tanh())
        for layer_idx in range(len(layers)-1):
            self.mlp_list.append(nn.Linear(layers[layer_idx],layers[layer_idx+1]))
            #self.mlp_list.append(torch.nn.ReLU())
            self.mlp_list.append(torch.nn.ReLU())
        self.mlp_list.append(nn.Linear(layers[-1],out_dim))
        self.mlp = torch.nn.Sequential(*self.mlp_list)

    def forward(self,x):
        return self.mlp(x)

class Conditional_VAE(nn.Module):
    def __init__(self,in_dim, c_dim, z_dim, num_measures):
        super(Conditional_VAE, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=2*z_dim         #128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=z_dim+c_dim, out_features=in_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )

        self.prediction_layer_age = nn.Linear(25,1)
        self.prediction_layer_age_hidden_2 = nn.Linear(50,25)

        self.prediction_layer_age_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_sex_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_sex_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_sex = nn.Linear(25,2)
        self.prediction_layer_cms = nn.Linear(100, num_measures)
        self.hidden_layer_cms = nn.Linear(c_dim + z_dim, 100)
        self.prediction_layer_dx = nn.Linear(25,1)
        self.prediction_layer_dx_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_dx_hidden_2 = nn.Linear(50,25)


        self.softmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()

    def to_z_params(self,x):
        #z = self.enc(x)
        z = self.encoder_hidden_layer(x)
        z = torch.relu(z)
        z = self.encoder_output_layer(z)
        z = torch.relu(z)
        ##print("z shape",z.shape)
        #z_mu = torch.tanh(z[...,:self.z_dim])
        z_mu = z[...,:self.z_dim]
        ##print("zmu shape",z_mu.shape)
        z_log_sigma_sq = z[...,self.z_dim:]
        ##print("z log sigma shape",z_log_sigma_sq.shape)
        return z_mu, z_log_sigma_sq

    def to_z(self, z_mu, z_log_sigma_sq):
        std = torch.exp(0.5 * z_log_sigma_sq)
        eps = torch.randn_like(std)
        z = eps * std + z_mu
        #z = z_mu
        return z


    def forward_train(self,x,c=None,cat_order=0):
        z_mu, z_log_sigma_sq = self.to_z_params(x)
        z = self.to_z(z_mu, z_log_sigma_sq)
       
        z = torch.relu(z)
       

        prediction_age = self.prediction_layer_age(self.prediction_layer_age_hidden_2(self.prediction_layer_age_hidden_1(z)))


        prediction_sex = self.prediction_layer_sex((self.prediction_layer_sex_hidden_2((self.prediction_layer_sex_hidden_1(z)))))
        prediction_dx = self.prediction_layer_dx(F.relu(self.prediction_layer_dx_hidden_2(F.relu(self.prediction_layer_dx_hidden_1(z)))))
        c = c.to(torch.float32)
        predictions = self.prediction_layer_cms(self.hidden_layer_cms(torch.cat((z,c),axis=1)))
        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out)
        return out, z, z_mu, z_log_sigma_sq, predictions, prediction_age, prediction_sex, prediction_dx


