import numpy as np
import sys

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



features = np.load(sys.argv[1])
y = np.load(sys.argv[2])
#print(y[0,:])
y = np.argmax(y, axis=1)
#print(y[0])
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

latent_2d = tsne.fit_transform(pca_result)
#ax= plt.axes(projection ="3d")
# Create the plot
plt.figure(figsize=(8, 6))
ax= plt.axes(projection ="3d")

scatter = ax.scatter3D(latent_2d[:, 0], latent_2d[:, 1],latent_2d[:, 2] ,c=y, cmap='rainbow', s=50, alpha=0.5)

# Add a colorbar
colorbar = plt.colorbar(scatter)
colorbar.set_label('SiteID')

# Add labels and title
plt.title('t-SNE Plot of Latent Space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Show the plot
#plt.show()
plt.savefig("TSNE_site.png")
