import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from Quantizer import Model, Model_more_quantizers

#Parameters of the VQ-VAE
batch_size = 128
num_training_updates = 30000

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

#Load pre trained quantizers 
model.load_state_dict(torch.load('models/vq_vae_64.pt'))

model_mq = Model_more_quantizers(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay, [64,32,16,8,4,2]).to(device)

model_mq._encoder.load_state_dict(model._encoder.state_dict())
model_mq._pre_vq_conv.load_state_dict(model._pre_vq_conv.state_dict())
model_mq._decoder.load_state_dict(model._decoder.state_dict())
model_mq._list_of_quantizers[0].load_state_dict(model._vq_vae.state_dict())

model_mq._list_of_quantizers[1].load_state_dict(torch.load('models/vq_vae_32.pt'))
model_mq._list_of_quantizers[2].load_state_dict(torch.load('models/vq_vae_16.pt'))
model_mq._list_of_quantizers[3].load_state_dict(torch.load('models/vq_vae_8.pt'))
model_mq._list_of_quantizers[4].load_state_dict(torch.load('models/vq_vae_4.pt'))
model_mq._list_of_quantizers[5].load_state_dict(torch.load('models/vq_vae_2.pt'))

model_mq._list_of_quantizers[0].to(device)
model_mq._list_of_quantizers[1].to(device)
model_mq._list_of_quantizers[2].to(device)
model_mq._list_of_quantizers[3].to(device)
model_mq._list_of_quantizers[4].to(device)
model_mq._list_of_quantizers[5].to(device)

model_mq._list_of_quantizers[0].eval()
model_mq._list_of_quantizers[1].eval()
model_mq._list_of_quantizers[2].eval()
model_mq._list_of_quantizers[3].eval()
model_mq._list_of_quantizers[4].eval()
model_mq._list_of_quantizers[5].eval()

### Test the environment
env = gym.make('CartPole-v1', render_mode = 'rgb_array')
from utils import get_screen
env.reset()
h,w = get_screen(env).squeeze().numpy().shape
env.close()


# Train a regressor for each quantizer
from utils import PhysicalValueRegressor, train_regressor

# 64 CODEWORDS
def encoding_function(est_state):
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[0](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
    return state_encoded

reg = PhysicalValueRegressor(512, 4)
reg = train_regressor(reg, encoding_function, h, w)
torch.save(reg.state_dict(), 'models/state_estimator_64.pt')

# 32 CODEWORDS
def encoding_function(est_state):
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[1](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
    return state_encoded

reg = PhysicalValueRegressor(512, 4)
reg = train_regressor(reg, encoding_function, h, w)
torch.save(reg.state_dict(), 'models/state_estimator_32.pt')

# 16 CODEWORDS
def encoding_function(est_state):
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[2](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
    return state_encoded

reg = PhysicalValueRegressor(512, 4)
reg = train_regressor(reg, encoding_function, h, w)
torch.save(reg.state_dict(), 'models/state_estimator_16.pt')

#8 CODEWORDS 
def encoding_function(est_state):
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[3](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
    return state_encoded

reg = PhysicalValueRegressor(512, 4)
reg = train_regressor(reg, encoding_function, h, w)
torch.save(reg.state_dict(), 'models/state_estimator_8.pt')

# 4 CODEWORDS
def encoding_function(est_state):
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[4](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
    return state_encoded

reg = PhysicalValueRegressor(512, 4)
reg = train_regressor(reg, encoding_function, h, w)
torch.save(reg.state_dict(), 'models/state_estimator_4.pt')

# 2 CODEWORDS
def encoding_function(est_state):
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[2](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
    return state_encoded

reg = PhysicalValueRegressor(512, 4)
reg = train_regressor(reg, encoding_function, h, w)
torch.save(reg.state_dict(), 'models/state_estimator_2.pt')
