import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
from torch.utils.data import Dataset, DataLoader, Subset,  WeightedRandomSampler
from Dataloader_train import TrainingDataset
from Dataloader_test import TestDataset
import sys

from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from Architecture import AE, Conditional_VAE
import losses
from sklearn.feature_selection import mutual_info_regression
from torch.utils.tensorboard import SummaryWriter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import resample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

CONNECTOME_SIZE = 121*121
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Pass in an ID to make specific tensorboard folder
experimentnote = sys.argv[1]

# Set up Tensorboard log
tensorboard_dir = f'Tensorboard_{experimentnote}'
writer = SummaryWriter(tensorboard_dir)

# Hyperparameters
beta         =     1
age_coeff    =     100
cnm_coeff    =     10  # cnm
recon_coeff  =     10  # recon
sex_coeff    =     10  # sex loss
kl_coeff     =     100 # klloss
dx_coeff     =     100
batch_size = 100       
epochs =    2001
learning_rate = 1e-4



normalize_cnms          =np.array([1,1,1/2,1,1/np.sqrt(10000000),1/1000000,1/200, 1/200,1, 1, 1/250, 1/np.sqrt(CONNECTOME_SIZE)])
normalize_cnms          =torch.from_numpy(normalize_cnms).to(DEVICE)
unnormlizefactor        = np.array([1,1,2,1,np.sqrt(10000000),1000000,200, 200,1, 1, 250, np.sqrt(CONNECTOME_SIZE)])
makepositive            =np.array([0,0,1,0,0,0,0,0,0,0,0,0])
makepositive            = torch.from_numpy(makepositive).to(DEVICE)


# Set up dataset (to be subsampled into training and validation) and folds
transform            = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset        = TrainingDataset()
test_dataset         = TestDataset()
#validation_loader    = DataLoader(test_dataset, batch_size=len(test_dataset))

def random_split(N, train_size=0.8):
    # Create an array of integers from 0 to N
    arr = np.arange(N)
    random_seed = 42
    np.random.seed(random_seed)

    # Shuffle the array
    np.random.shuffle(arr)

    # Calculate the number of samples for the training set
    n_train = int(N * train_size)

    # Split the array into training and validation sets
    train_set = arr[:n_train]
    validation_set = arr[n_train:]

    return train_set, validation_set


measure_names = ["betweenness_centrality","modularity","assortativity","participation","clustering","nodal_strength","local_efficiency","global_efficiency", "density","rich_club","path_length","edge_count"]

experiment=""
# Bootstrap iterations
for iteration in np.arange(1,2):
    train_indices, val_indices = random_split(len(train_dataset))
    train_dataset_subsample   = Subset(train_dataset, train_indices)
    val_dataset_subsample = Subset(train_dataset, val_indices)

    print('Bootstrap iter: {}'.format(iteration))
    groups = np.zeros(len(train_dataset_subsample))
    for i in np.arange(0,len(train_dataset_subsample)):
        groups[i] = train_dataset_subsample[i][5]

    # Calculate class weights
    class_counts = torch.bincount(torch.tensor(groups,dtype=torch.int64))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[groups]
    
    # Create a sampler
    weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    # Create DataLoader
    train_loader            = DataLoader(train_dataset_subsample, batch_size=batch_size, sampler=weighted_sampler)
    validation_loader       = DataLoader(val_dataset_subsample, batch_size=batch_size)
    # create a model from Arch (conditional variational autoencoder)
    # load it to the specified device, either gpu or cpu
    model = Conditional_VAE(in_dim=CONNECTOME_SIZE,c_dim=100, z_dim=100, num_measures=12).to(DEVICE)
    
    # if you are setting a path to a pretrained model/want to continue training on a model, import it here!
    #PATH='/home-local/Aim2/WEIGHTS/Model_EPOCH500_NEWARCH.pt'
    #model.load_state_dict(torch.load(PATH))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define losses
    loss_fn         = losses.BetaVAE_Loss(beta)
    criterion_CE    = nn.BCEWithLogitsLoss()
    dx_criterion = nn.MSELoss()

    for epoch in np.arange(1,epochs):
        print("EPOCH: ",epoch)
        # Set all recording losses to zero
        loss = 0
        training_loss_total = 0
        training_loss_reconstruction = 0
        training_loss_kldivergence = 0
        training_loss_prediction_site1 = 0
        training_loss_prediction_site2 = 0
        training_loss_age = 0
        training_loss_sex = 0
        training_loss_dx = 0
        countofsite1 = 0
        countofsite2 = 0
        training_loss_prediction = 0
        site_counts = 0

        for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, batch_sexlabels, batch_groups,batch_dxlabels,_ in train_loader:
            # Send features, prediction labels, and site labels to device and set dimensions
            batch_features          = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE) # flatten 84x84 connectome
            batch_predictionlabels  = batch_predictionlabels.view(-1,12).to(DEVICE) # 12 graph measures
            batch_sitelabels        = batch_sitelabels.view(-1,100).to(DEVICE) # 1 site label per scan
            makepositive            = torch.zeros_like(batch_predictionlabels)
            makepositive[:,2]       = 1
            batch_predictionlabels  = normalize_cnms.view(-1,12) * (batch_predictionlabels + makepositive) # make all values normalized between 0 and 1 (for model learning)
            batch_predictionlabels  = batch_predictionlabels[:,measures_on==1].to(DEVICE)
            batch_agelabels         = batch_agelabels.view(-1,1).to(DEVICE)
            batch_sexlabels         = batch_sexlabels.view(-1,2).to(DEVICE)
            batch_dxlabels          = batch_dxlabels.view(-1,1).to(DEVICE)

            # Standard optimizer command
            optimizer.zero_grad()

            # Put data through forward pass
            x_hat, z, z_mu, z_log_sigma_sq, model_predictions, model_predictions_age, model_predictions_sex, model_predictions_dx = model.forward_train(batch_features,batch_sitelabels)

            # Reconstruction and KL divergence loss terms
            recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)
            age_error = torch.mean(torch.square((batch_agelabels-model_predictions_age)/batch_agelabels))
            sex_error = criterion_CE(model_predictions_sex, batch_sexlabels)
            dx_error = dx_criterion(model_predictions_dx, batch_dxlabels) #batch_dxlabels.type(torch.LongTensor).to(DEVICE))

            # calculate sex classification accuracy
            sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
            _,sex_predictions = torch.max(sex_predictions, 1)
            _,sex_labels = torch.max(batch_sexlabels, 1)
            sex_accuracy = (sex_predictions == sex_labels).float().mean()
            training_loss_sexacc = sex_accuracy

            # compute training loss for sex classification
            train_sex = torch.sum(sex_labels) / len(sex_labels)

            # Weight KL loss with beta, set parameter at top
            kl_loss = beta * kl_loss
            model_prediction_error = (model_predictions - batch_predictionlabels)/ batch_predictionlabels

            # MSE prediction loss
            cnm_prediction_error = torch.sum(torch.mean(torch.square(model_prediction_error),0))

            # Total Loss
            loss = (kl_coeff * kl_loss) + (recon_coeff * recon_loss) + (cnm_coeff * cnm_prediction_error) + (age_coeff * age_error) + (sex_coeff * sex_error) + (dx_coeff * dx_error)

            # Compute loss gradients and take a step with the Adam optimizer
            loss.backward()
            optimizer.step()

            # Add the mini-batch training loss to epoch loss
            training_loss_total             += loss.item()
            training_loss_reconstruction    += recon_loss.item()
            training_loss_kldivergence      += kl_loss.item()
            training_loss_prediction        += cnm_prediction_error
            site_counts                     += torch.sum(batch_sitelabels)
            training_loss_age               += age_error.item()
            training_loss_sex               += sex_error.item()
            training_loss_dx                += dx_error.item()



        #measure_name=measure_names[measure_id]
        #writer.add_scalars('Cohen\'s D Measure {} Experiment {}'.format(measure_name, experiment),   {'Site 1 {}'.format(iteration): site1predictions_cohensd, 'Site 2 {}'.format(iteration): site2predictions_cohensd, 'Baseline {}'.format(iteration): baseline_cohensd, 'Predicting to Native {}'.format(iteration): predictingnative_cohensd}, epoch)
        # compute the epoch training loss
        training_loss_total     = training_loss_total / len(train_loader)
        training_loss_age       = age_coeff * training_loss_age / len(train_loader)
        training_loss_sex       = sex_coeff * training_loss_sex / len(train_loader)
        training_loss_dx       = dx_coeff * training_loss_dx / len(train_loader)
        training_loss_reconstruction    = recon_coeff * training_loss_reconstruction / len(train_loader)
        training_loss_kldivergence      = kl_coeff * training_loss_kldivergence /  len(train_loader)
        training_loss_prediction = cnm_coeff * torch.sum(training_loss_prediction) / torch.sum(site_counts)  # sum across all sites, normalized by how many samples were in that site


        training_loss_total = training_loss_reconstruction + training_loss_kldivergence + training_loss_prediction + training_loss_age + training_loss_sex + training_loss_dx


        sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
        _,sex_predictions = torch.max(sex_predictions, 1)
        _,sex_labels = torch.max(batch_sexlabels, 1)

        sex_accuracy = (sex_predictions == sex_labels).float().mean()
        validation_loss_sexacc = sex_accuracy


        measure_names = ["betweenness_centrality","modularity","assortativity","participation","clustering","nodal_strength","local_efficiency","global_efficiency", "density","rich_club","path_length","edge_count"]


        writer.add_scalars('Total Loss',   {'Train{}'.format(iteration): training_loss_total}, epoch)
        writer.add_scalars('Reconstruction Loss',   {'Train{}'.format(iteration): training_loss_reconstruction}, epoch)
        writer.add_scalars('KL Loss',   {'Train{}'.format(iteration): training_loss_kldivergence}, epoch)
        writer.add_scalars('Age Loss',   {'Train{}'.format(iteration): training_loss_age}, epoch)
        writer.add_scalars('Sex Loss',   {'Train{}'.format(iteration): training_loss_sex}, epoch)
        writer.add_scalars('Sex Acc',   {'Train{}'.format(iteration): training_loss_sexacc}, epoch)
        writer.add_scalars('Dx Loss',   {'Train{}'.format(iteration): training_loss_dx}, epoch)
        writer.add_scalars('GM Prediction',   {'Train{}'.format(iteration): training_loss_prediction}, epoch)

        # Save the weights
        if epoch % 100 == 0:
            PATH = f'./MODEL_epoch_{epoch}_iteration_{iteration}_{experimentnote}.pt'  # NOTE: @gaurav change this to your path
            torch.save(model.state_dict(), PATH)

        # Evaluate every 5 epochs
        if epoch % 5 == 0:

            validation_loss_prediction_site1 = 0
            validation_loss_prediction_site2 = 0
            validation_loss_kl =0
            validation_loss_age =0
            validation_loss_sex =0
            validation_loss_reconstruction =0
            countofsite1 = 0
            countofsite2 = 0
            val_loss_prediction = 0
            val_loss_age = 0
            val_loss_sex = 0
            val_loss_dx = 0
            val_loss_reconstruction = 0
            val_loss_kldivergence = 0
            val_loss_total = 0



            for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, batch_sexlabels,batch_groups, batch_dxlabels, batch_filepaths in validation_loader:
                batch_features          = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE)
                batch_predictionlabels  = batch_predictionlabels.view(-1,12).to(DEVICE)
                makepositive=torch.zeros_like(batch_predictionlabels)
                makepositive[:,2]=1
                batch_predictionlabels  = normalize_cnms.view(-1,12) * (batch_predictionlabels + makepositive)
                batch_predictionlabels  = batch_predictionlabels[:,measures_on==1].to(DEVICE)
                batch_sitelabels        = batch_sitelabels.view(-1,100).to(DEVICE)
                batch_agelabels         = batch_agelabels.view(-1,1).to(DEVICE)
                batch_dxlabels          = batch_dxlabels.view(-1,1).to(DEVICE)
                batch_sexlabels         = batch_sexlabels.view(-1,2).to(DEVICE)

                x_hat, z, z_mu, z_log_sigma_sq, model_predictions, model_predictions_age, model_predictions_sex, model_predictions_dx = model.forward_train(batch_features,batch_sitelabels)
                recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)
                sex_error = criterion_CE(model_predictions_sex, batch_sexlabels)
                dx_error = dx_criterion(model_predictions_dx, batch_dxlabels.type(torch.LongTensor).to(DEVICE))


                kl_loss = kl_loss * beta
                age_error = torch.mean(torch.square((batch_agelabels-model_predictions_age)/batch_agelabels))
                sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
                _,sex_predictions = torch.max(sex_predictions, 1)
                _,sex_labels = torch.max(batch_sexlabels, 1)
                sex_accuracy = (sex_predictions == sex_labels).float().mean()
                validation_loss_sexacc = sex_accuracy

                model_prediction_error = (model_predictions - batch_predictionlabels)/ batch_predictionlabels

                # MSE prediction loss
                cnm_prediction_error = torch.sum(torch.mean(torch.square(model_prediction_error),0))

                val_site_counts = torch.sum(batch_sitelabels)
                val_loss_prediction  += cnm_prediction_error
                val_loss_age += age_error
                val_loss_sex += sex_error
                val_loss_reconstruction += recon_loss
                val_loss_kldivergence  += kl_loss
                val_loss_dx += dx_error



            val_loss_age       = age_coeff * val_loss_age / len(validation_loader)
            val_loss_sex       = sex_coeff *  val_loss_sex / len(validation_loader)
            val_loss_dx       = dx_coeff *  val_loss_dx / len(validation_loader)
            val_loss_reconstruction    = recon_coeff * val_loss_reconstruction / len(validation_loader)
            val_loss_kldivergence      = kl_coeff * val_loss_kldivergence /  len(validation_loader)
            val_loss_prediction = cnm_coeff * torch.sum(val_loss_prediction) / len(validation_loader)  # sum across all sites, normalized by how many s>
            val_loss_total     = val_loss_age + val_loss_sex + val_loss_reconstruction + val_loss_kldivergence + val_loss_prediction + val_loss_dx

            # Write to tensorboard
            writer.add_scalars('Total Loss',   {'Val{}'.format(iteration): val_loss_total}, epoch)
            writer.add_scalars('Reconstruction Loss',   {'Val{}'.format(iteration): val_loss_reconstruction}, epoch)
            writer.add_scalars('KL Loss',   {'Val{}'.format(iteration): val_loss_kldivergence}, epoch)
            writer.add_scalars('Age Loss',   {'Val{}'.format(iteration): val_loss_age}, epoch)
            writer.add_scalars('Dx Loss',   {'Val{}'.format(iteration): val_loss_dx}, epoch)
            writer.add_scalars('Sex Acc',   {'Val{}'.format(iteration): validation_loss_sexacc}, epoch)
            writer.add_scalars('GM Prediction',   {'Val{}'.format(iteration): val_loss_prediction}, epoch)
            writer.add_scalars('Sex Loss',   {'Val{}'.format(iteration): val_loss_sex}, epoch)


    # Save model weights at the last epoch
    PATH = f'./Model_{iteration}_allsites_end_epoch_{epoch}_{experimentnote}.pt'

    experiment = ''.join(str(x) for x in measures_on)
    torch.save(model.state_dict(), PATH)
