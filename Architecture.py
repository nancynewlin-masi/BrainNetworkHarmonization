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
        # MLP for predicting age from the latent space (one scalar)
        self.prediction_layer_age_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_age_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_age = nn.Linear(25,1)
        
        # MLP for predicting sex from the latent space (two classes, two scalars)
        self.prediction_layer_sex_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_sex_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_sex = nn.Linear(25,2)
        
        # MLP for predicing diagnosis (1,2,3) from latent space (one scalar) NOTE: Maybe change this to a classification problem
        self.prediction_layer_dx_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_dx_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_dx = nn.Linear(25,1)
        
        # MLP for predicting network properties (cnms) from latent space + site encoding (z plus c)
        self.hidden_layer_cms = nn.Linear(c_dim + z_dim, 100)
        self.prediction_layer_cms = nn.Linear(100, num_measures)

        # Extra functions if needed
        self.softmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()

    def to_z_params(self,x):

        z = self.encoder_hidden_layer(x)
        z = torch.relu(z)
        z = self.encoder_output_layer(z)
        z = torch.relu(z)
        ##print("z shape",z.shape)
        #z_mu = torch.tanh(z[...,:self.z_dim]) 
        
        # Get mean (according to VAE literature)
        z_mu = z[...,:self.z_dim]
        # get log sigma squared (according to VAE literature)
        z_log_sigma_sq = z[...,self.z_dim:]

        return z_mu, z_log_sigma_sq

    def to_z(self, z_mu, z_log_sigma_sq):
        std = torch.exp(0.5 * z_log_sigma_sq)
        eps = torch.randn_like(std)
        z = eps * std + z_mu
        #z = z_mu
        return z


    def forward_train(self,x,c=None,cat_order=0):
        z_mu, z_log_sigma_sq = self.to_z_params(x) # get mean and log sigma squared 
        z = self.to_z(z_mu, z_log_sigma_sq) 
       
        z = torch.relu(z) # nonlinear activation
       
        # predict age
        prediction_age = self.prediction_layer_age(self.prediction_layer_age_hidden_2(self.prediction_layer_age_hidden_1(z)))
        
        # predict sex
        prediction_sex = self.prediction_layer_sex((self.prediction_layer_sex_hidden_2((self.prediction_layer_sex_hidden_1(z)))))
        
        # predict diagnosis
        prediction_dx = self.prediction_layer_dx(F.relu(self.prediction_layer_dx_hidden_2(F.relu(self.prediction_layer_dx_hidden_1(z)))))
        
        # get site one-hot encoding
        c = c.to(torch.float32)
        
        # get network measures in site c context/domain
        predictions = self.prediction_layer_cms(self.hidden_layer_cms(torch.cat((z,c),axis=1)))
        
        # reconstruct the connectome from z + c in site c context/domain
        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out) # NOTE: If reconstuction looks bad, try adding non-linear activations to mirror the encoder
        
        return out, z, z_mu, z_log_sigma_sq, predictions, prediction_age, prediction_sex, prediction_dx

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
        # MLP for predicting age from the latent space (one scalar)
        self.prediction_layer_age_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_age_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_age = nn.Linear(25,1)
        
        # MLP for predicting sex from the latent space (two classes, two scalars)
        self.prediction_layer_sex_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_sex_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_sex = nn.Linear(25,2)
        
        # MLP for predicing diagnosis (1,2,3) from latent space (one scalar) NOTE: Maybe change this to a classification problem
        self.prediction_layer_dx_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_dx_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_dx = nn.Linear(25,1)
        
        # MLP for predicting network properties (cnms) from latent space + site encoding (z plus c)
        self.hidden_layer_cms = nn.Linear(c_dim + z_dim, 100)
        self.prediction_layer_cms = nn.Linear(100, num_measures)

        # Extra functions if needed
        self.softmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()

    def to_z_params(self,x):

        z = self.encoder_hidden_layer(x)
        z = torch.relu(z)
        z = self.encoder_output_layer(z)
        z = torch.relu(z)
        ##print("z shape",z.shape)
        #z_mu = torch.tanh(z[...,:self.z_dim]) 
        
        # Get mean (according to VAE literature)
        z_mu = z[...,:self.z_dim]
        # get log sigma squared (according to VAE literature)
        z_log_sigma_sq = z[...,self.z_dim:]

        return z_mu, z_log_sigma_sq

    def to_z(self, z_mu, z_log_sigma_sq):
        std = torch.exp(0.5 * z_log_sigma_sq)
        eps = torch.randn_like(std)
        z = eps * std + z_mu
        #z = z_mu
        return z


    def forward_train(self,x,c=None,cat_order=0):
        z_mu, z_log_sigma_sq = self.to_z_params(x) # get mean and log sigma squared 
        z = self.to_z(z_mu, z_log_sigma_sq) 
       
        z = torch.relu(z) # nonlinear activation
       
        # predict age
        prediction_age = self.prediction_layer_age(self.prediction_layer_age_hidden_2(self.prediction_layer_age_hidden_1(z)))
        
        # predict sex
        prediction_sex = self.prediction_layer_sex((self.prediction_layer_sex_hidden_2((self.prediction_layer_sex_hidden_1(z)))))
        
        # predict diagnosis
        prediction_dx = self.prediction_layer_dx(F.relu(self.prediction_layer_dx_hidden_2(F.relu(self.prediction_layer_dx_hidden_1(z)))))
        
        # get site one-hot encoding
        c = c.to(torch.float32)
        
        # get network measures in site c context/domain
        predictions = self.prediction_layer_cms(self.hidden_layer_cms(torch.cat((z,c),axis=1)))
        
        # reconstruct the connectome from z + c in site c context/domain
        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out) # NOTE: If reconstuction looks bad, try adding non-linear activations to mirror the encoder
        
        return out, z, z_mu, z_log_sigma_sq, predictions, prediction_age, prediction_sex, prediction_dx

