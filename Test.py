import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
#from Dataloader_multimeasure import CustomDataset
from Dataloader_test import TestDataset
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from Architecture import AE, Conditional_VAE
import losses
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import os
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

CONNECTOME_SIZE = 121*121
seed = 42
batch_size = 15000
test_dataset = TestDataset()


test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

experimentname = []
predictionloss_site1 = []
predictionloss_site2 = []
age_loss = []
mutualinformation = []
rfacc = []
delta=1


# Hyperparameters
beta =      1
delta =     100 #cnm
alpha =     100 # recon
kappa =     10000  #sex loss
gamma =     1000  # klloss
batch_size = 1000       # Change when add in more data
epochs =    501
learning_rate = 1e-4



normalize_cnms          =np.array([1,1,1/2,1,1/np.sqrt(10000000),1/1000000,1/200, 1/200,1, 1, 1/250, 1/np.sqrt(CONNECTOME_SIZE)])
normalize_cnms          =torch.from_numpy(normalize_cnms).to(DEVICE)
unnormlizefactor        = np.array([1,1,2,1,np.sqrt(10000000),1000000,200, 200,1, 1, 250, np.sqrt(CONNECTOME_SIZE)])
makepositive            =np.array([0,0,1,0,0,0,0,0,0,0,0,0])
makepositive            = torch.from_numpy(makepositive).to(DEVICE)

test_examples = None
plt.figure(figsize=(20, 4))
index = 0
number = len(test_loader)

site1predictions = np.zeros((len(test_loader),1))
site2predictions = np.zeros((len(test_loader),1))
pred_labels = np.zeros((len(test_loader),1))
site_labels = np.zeros((len(test_loader),1))
predictions = np.zeros((len(test_loader),1))
ageerrors = np.zeros((len(test_loader),1))
filenames = []
seed_value = 42

model = Conditional_VAE(in_dim=CONNECTOME_SIZE,c_dim=100, z_dim=100, num_measures=12).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
PATH=sys.argv[1]
model.load_state_dict(torch.load(PATH))

model.eval()
batchid=0
# Define losses
loss_fn         = losses.BetaVAE_Loss(beta)
criterion_CE    = nn.BCEWithLogitsLoss()
training_loss_total             = 0
training_loss_reconstruction    = 0
training_loss_kldivergence      = 0
training_loss_prediction        = 0
site_counts                     = 0
training_loss_age               = 0
training_loss_sex               = 0
mergeddf = pd.DataFrame(columns=["Path","SiteID","Age","Sex","betweenness_centrality_gt","betweenness_centrality_pred","modularity_gt", "modularity_pred","assortativity_gt","assortativity_pred","participation_gt","participation_pred","clustering_gt","clustering_pred","nodal_strength_gt","nodal_strength_pred","local_efficiency_gt","local_efficiency_pred","global_efficiency_gt", "global_efficiency_pred", "density_gt","density_pred","rich_club_gt","rich_club_pred","path_length_gt","path_length_pred","edge_count_gt","edge_count_pred"])

with torch.no_grad():

        #for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, batch_sexlabels, imgpaths in test_loader:
        for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, batch_sexlabels, batch_groups, batch_dx, imgpaths in test_loader:
            batch_connectomebaseline = batch_features
            # Send features, prediction labels, and site labels to device and set dimensions
            batch_features          = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE) # flatten 84x84 connectome
            batch_predictionlabels  = batch_predictionlabels.view(-1,12).to(DEVICE) # 12 graph measures
            batch_predictionlabels_org = batch_predictionlabels
            batch_sitelabels        = batch_sitelabels.view(-1,100).to(DEVICE) # 1 site label per scan
            makepositive            = torch.zeros_like(batch_predictionlabels)
            makepositive[:,2]       = 1
            batch_predictionlabels  = normalize_cnms.view(-1,12) * (batch_predictionlabels + makepositive)
            batch_dx = batch_dx.view(-1,1).to(DEVICE)


            batch_predictionlabels  = batch_predictionlabels.to(DEVICE)
            batch_agelabels         = batch_agelabels.view(-1,1).to(DEVICE)
            batch_sexlabels         = batch_sexlabels.view(-1,2).to(DEVICE)
            print(batch_sexlabels[0,:], batch_agelabels[0,:], batch_predictionlabels[0,:])
            np.save("dx.npy", batch_dx.detach().cpu())

            np.save("sex.npy", batch_sexlabels.detach().cpu())
            np.save("age.npy", batch_agelabels.detach().cpu())
            np.save("site.npy", batch_sitelabels.detach().cpu())
            # Standard optimizer command
            optimizer.zero_grad()
            batch_sitelabels_same = torch.zeros_like(batch_sitelabels)
            #print("Shape:", batch_sitelabels.shape)
            batch_sitelabels_same[:,0]=1
            # Put data through forward pass
            x_hat, z, z_mu, z_log_sigma_sq, model_predictions, model_predictions_age, model_predictions_sex, model_predictions_dx = model.forward_train(batch_features,batch_sitelabels_same)
            # Perform t-SNE
            np.save("z.npy", z.detach().cpu())
            tsne = TSNE(n_components=3, random_state=42, perplexity=30)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(z.detach().cpu())
            print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

            latent_2d = tsne.fit_transform(pca_result)
            #ax= plt.axes(projection ="3d")
            # Create the plot
            plt.figure(figsize=(8, 6))
            ax= plt.axes(projection ="3d")

            scatter = ax.scatter3D(latent_2d[:, 0], latent_2d[:, 1],latent_2d[:, 2] ,c=batch_dx.detach().cpu(), cmap='rainbow', s=50, alpha=0.5)

            # Add a colorbar
            colorbar = plt.colorbar(scatter)
            colorbar.set_label('Diagnosis')

            # Add labels and title
            plt.title('t-SNE Plot of Latent Space')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            # Show the plot
            #plt.show()
            plt.savefig("TSNE_dx.png")
            plt.figure(figsize=(8, 6))
            ax= plt.axes(projection ="3d")
            scatter = ax.scatter3D(latent_2d[:, 0], latent_2d[:, 1], latent_2d[:, 2],c=batch_sexlabels.detach().cpu()[:,0], cmap='rainbow', s=50, alpha=0.5)

            # Add a colorbar
            colorbar = plt.colorbar(scatter)
            colorbar.set_label('Sex')

            # Add labels and title
            plt.title('t-SNE Plot of Latent Space')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            # Show the plot
            #plt.show()
            plt.savefig("TSNE_sex.png")
            plt.figure(figsize=(8, 6))
            ax= plt.axes(projection ="3d")

            scatter = ax.scatter3D(latent_2d[:, 0], latent_2d[:, 1],latent_2d[:, 2],c=batch_agelabels.detach().cpu(), cmap='rainbow', s=50, alpha=0.5)

            # Add a colorbar
            colorbar = plt.colorbar(scatter)
            colorbar.set_label('Age')

            # Add labels and title
            plt.title('t-SNE Plot of Latent Space')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            # Show the plot
            #plt.show()
            plt.savefig("TSNE_age.png")


            # Reconstruction and KL divergence loss terms
            recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)
            age_error = torch.mean(torch.square((batch_agelabels-model_predictions_age)/batch_agelabels))
            sex_error = criterion_CE(model_predictions_sex, batch_sexlabels)

            sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
            _,sex_predictions = torch.max(sex_predictions, 1)
            _,sex_labels = torch.max(batch_sexlabels, 1)
            sex_accuracy = (sex_predictions == sex_labels).float().mean()
            training_loss_sexacc = sex_accuracy

            train_sex = torch.sum(sex_labels) / len(sex_labels)

            # Weight KL loss with beta, set parameter at top
            kl_loss = beta * kl_loss

            model_prediction_error = (model_predictions - batch_predictionlabels)/ batch_predictionlabels
            #"Model Predictions Shape:", model_predictions.shape)

            # MSE prediction loss
            cnm_prediction_error = torch.sum(torch.mean(torch.square(model_prediction_error),0))

            # Total Loss
            loss = gamma * kl_loss + alpha * recon_loss + delta * cnm_prediction_error + age_error + kappa * sex_error

            # Compute loss gradients and take a step with the Adam optimizer
            #loss.backward()
            #optimizer.step()

            # Add the mini-batch training loss to epoch loss
            training_loss_total             += loss.item()
            training_loss_reconstruction    += recon_loss.item()
            training_loss_kldivergence      += kl_loss.item()
            training_loss_prediction        += cnm_prediction_error
            site_counts                     += torch.sum(batch_sitelabels)
            training_loss_age               += age_error.item()
            training_loss_sex               += sex_error.item()

            makepositive = makepositive.detach().cpu()
            model_predictions = model_predictions.detach().cpu()
            model_predictions_origranges = ((model_predictions * unnormlizefactor) - makepositive).detach().cpu()

            currentdf = pd.DataFrame(columns=["Path","SiteID","Age","Sex","diagnosis", "betweenness_centrality_gt","betweenness_centrality_pred","modularity_gt", "modularity_pred","assortativity_gt","assortativity_pred","participation_gt","participation_pred","clustering_gt","clustering_pred","nodal_strength_gt","nodal_strength_pred","local_efficiency_gt","local_efficiency_pred","global_efficiency_gt", "global_efficiency_pred", "density_gt","density_pred","rich_club_gt","rich_club_pred","path_length_gt","path_length_pred","edge_count_gt","edge_count_pred"])
            currentdf["SiteID"] = torch.argmax(batch_sitelabels, dim=1).detach().cpu()
            currentdf["Sex"] = torch.argmax(batch_sexlabels, dim=1).detach().cpu()
            currentdf["Age"] = batch_agelabels.detach().cpu()
            currentdf["diagnosis"] = batch_dx.detach().cpu()
            currentdf["betweenness_centrality_gt"] = batch_predictionlabels_org[:,0].detach().cpu()
            currentdf["modularity_gt"] = batch_predictionlabels_org[:,1].detach().cpu()
            currentdf["assortativity_gt"] = batch_predictionlabels_org[:,2].detach().cpu()
            currentdf["participation_gt"] = batch_predictionlabels_org[:,3].detach().cpu()
            currentdf["clustering_gt"] = batch_predictionlabels_org[:,4].detach().cpu()
            currentdf["nodal_strength_gt"] = batch_predictionlabels_org[:,5].detach().cpu()
            currentdf["local_efficiency_gt"] = batch_predictionlabels_org[:,6].detach().cpu()
            currentdf["global_efficiency_gt"] = batch_predictionlabels_org[:,7].detach().cpu()
            currentdf["density_gt"] = batch_predictionlabels_org[:,8].detach().cpu()
            currentdf["rich_club_gt"] = batch_predictionlabels_org[:,9].detach().cpu()
            currentdf["path_length_gt"] = batch_predictionlabels_org[:,10].detach().cpu()
            currentdf["edge_count_gt"] = batch_predictionlabels_org[:,11].detach().cpu()


            currentdf["betweenness_centrality_pred"] = model_predictions_origranges[:,0].detach().cpu()
            currentdf["modularity_pred"] = model_predictions_origranges[:,1].detach().cpu()
            currentdf["assortativity_pred"] = model_predictions_origranges[:,2].detach().cpu()
            currentdf["participation_pred"] = model_predictions_origranges[:,3].detach().cpu()
            currentdf["clustering_pred"] = model_predictions_origranges[:,4].detach().cpu()
            currentdf["nodal_strength_pred"] = model_predictions_origranges[:,5].detach().cpu()
            currentdf["local_efficiency_pred"] = model_predictions_origranges[:,6].detach().cpu()
            currentdf["global_efficiency_pred"] = model_predictions_origranges[:,7].detach().cpu()
            currentdf["density_pred"] = model_predictions_origranges[:,8].detach().cpu()
            currentdf["rich_club_pred"] = model_predictions_origranges[:,9].detach().cpu()
            currentdf["path_length_pred"] = model_predictions_origranges[:,10].detach().cpu()
            currentdf["edge_count_pred"] = model_predictions_origranges[:,11].detach().cpu()
            #currentdf["Path"] = imgpaths

            mergeddf = pd.concat([mergeddf, currentdf], ignore_index=True)
            index = index+1

            for sample in np.arange(0,len(batch_sitelabels)):
                    connectome = x_hat[sample].cpu().numpy().reshape(121,121)
                    groupid = batch_groups[sample].item()
                    np.savetxt(f"CONNECTOME_SiteID_{groupid}_Sample_{sample}.csv", connectome, delimiter=",")
                    connectome = batch_connectomebaseline[sample].cpu().numpy().reshape(121,121)
                    groupid = batch_groups[sample].item()
                    np.savetxt(f"CONNECTOME_ORIGINAL_SiteID_{groupid}_Sample_{sample}.csv", connectome, delimiter=",")

            with open(f"filenames.txt", 'a') as f:
                    for s in imgpaths:
                            f.write(str(s) + '\n')


            batchid=batchid+1

mergeddf.to_csv("MERGEDDF.csv")
training_loss_total             = training_loss_total / len(test_loader)
training_loss_reconstruction    = training_loss_reconstruction / len(test_loader)
training_loss_kldivergence      = training_loss_kldivergence / len(test_loader)
training_loss_prediction        = training_loss_prediction / len(test_loader)
training_loss_age               = training_loss_age / len(test_loader)
training_loss_sex               = training_loss_sex / len(test_loader)

print("training_loss_total:",training_loss_total)
print("training_loss_reconstruction:",training_loss_reconstruction)
print("training_loss_kldivergence:",training_loss_kldivergence)
print("training_loss_prediction:",training_loss_prediction)
print("training_loss_age:",training_loss_age)
print("training_loss_sex:",training_loss_sex)
