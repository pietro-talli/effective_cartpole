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
import torch.optim as optim

# Local libraries
from utilities_dataset import create_dataset, ToTensor, FramesDataset
from Quantizer import Model, VectorQuantizerEMA, VectorQuantizer

def obtain_a_new_quantizer(num_codewords):

    # Load the dataset
    batch_size = 128 # batch size to use to train the vq_vae
    dataset = FramesDataset('dataset/description.csv', 'dataset/images', ToTensor())
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=True, num_workers=6)

    #Load the pre-trained vq_vae
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

    model.load_state_dict(torch.load('models/vq_vae_64.pt'))

    # Train different quantizers
    num_codewords = num_codewords # Set a new number of codewords 
    model._vq_vae = VectorQuantizerEMA(num_codewords, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model._vq_vae.parameters(), lr=1e-3, amsgrad=False)

    import torch.nn.functional as F

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    reset = True

    num_training_updates = 100

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
                reset = True #Reset the unused codewords every 100 iterations

    name_of_quantizer = 'models/vq_vae_' + str(num_codewords) + '.pt'
    torch.save(model._vq_vae.state_dict(), name_of_quantizer)
    
#Use the function obtain_a_new_quantizer(num_codewords) to train 
# a new quantizer from an existing Encoder-Decoder pair 
    
Ks = [32,16,8,4,2]
for K in Ks:
    obtain_a_new_quantizer(K)