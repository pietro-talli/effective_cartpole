import gym
import numpy as np 
from PIL import Image
from copy import deepcopy
from typing import Optional, Union
import csv
import numpy as np 
from PIL import Image
import torch
import os
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Local libraries
from utilities_dataset import create_dataset, ToTensor, FramesDataset
from Quantizer import Model

# Collect a dataset of tuples (o_t, o_{t+1})
num_samples = 50000 #number of tuples to collect
create_dataset()

batch_size = 128 # batch size to use to train the vq_vae

# Create a dataset
dataset = FramesDataset('dataset/description.csv', 'dataset/images', ToTensor())
dataloader = DataLoader(dataset, batch_size=128,
                        shuffle=True, num_workers=6)



#Parameters of the VQ-VAE

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_training_updates = 100
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 64
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

_, h,w = dataset[0]['curr'].shape


#Training loop of the VQ_VAE

model.train()
train_res_recon_error = []
train_res_perplexity = []

reset = True

for i in range(num_training_updates):
    for i_batch, sample_batched in enumerate(dataloader):
        
        input_tensor = torch.cat((sample_batched['curr'], sample_batched['next']), dim = 1)
        input_tensor = input_tensor.to(device)
        data = 1-input_tensor
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data, reset)
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        reset = False
        
        if (i_batch+1) % 100 == 0:
            #print('%d iterations' % (i+1))
            #print('recon_error: %.5f' % np.mean(train_res_recon_error[-100:]))
            #print('perplexity: %.5f' % np.mean(train_res_perplexity[-100:]))
            #print()
            reset = True #Reset the unused codewords every 100 iteroations
            
torch.save(model.state_dict(), 'models/vq_vae_64.pt')