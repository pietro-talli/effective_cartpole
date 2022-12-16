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

# train the control policies

#Initialize the Gym environment
env = gym.make('CartPole-v1', render_mode = 'rgb_array')

# Get the shapes of the state space (observation_space) and action space (action_space)
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

### PARAMETERS
gamma = 0.97   # gamma parameter for the long term reward
replay_memory_capacity = 10000   # Replay memory capacity

target_net_update_steps = 10   # Number of episodes to wait before updating the target network
batch_size = 512 # Number of samples to take from the replay memory for each update
bad_state_penalty = 0   # Penalty to the reward when we are in a bad state (in this case when the pole falls down) 
min_samples_for_training = 1000   # Minimum samples in the replay memory to enable the training

#number ofcodewords
C = 64
num_feat = 8
embedding_dim = 64

from utils import DQN_SMALL

### Initialize the policy network
policy_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)

### Initialize the target network with the same weights of the policy network
target_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

from utils import train_policy_ema

def encoding_function(est_state):
    model_mq._list_of_quantizers[1].eval()
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[0](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,embedding_dim*num_feat))
    return state_encoded

policy_net = train_policy_ema(256, 
                          policy_net, target_net, 16000, target_net_update_steps, min_samples_for_training, 
                          encoding_function)
torch.save(policy_net.state_dict(), 'models/new_policy_net_64.pt')


### Initialize the policy network
policy_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)

### Initialize the target network with the same weights of the policy network
target_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

from utils import train_policy_ema

def encoding_function(est_state):
    model_mq._list_of_quantizers[1].eval()
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[1](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,embedding_dim*num_feat))
    return state_encoded

policy_net = train_policy_ema(256, 
                          policy_net, target_net, 16000, target_net_update_steps, min_samples_for_training, 
                          encoding_function)
torch.save(policy_net.state_dict(), 'models/new_policy_net_32.pt')

### Initialize the policy network
policy_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)

### Initialize the target network with the same weights of the policy network
target_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

def encoding_function(est_state):
    model_mq._list_of_quantizers[2].eval()
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[2](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,embedding_dim*num_feat))
    return state_encoded

policy_net = train_policy_ema(256, 
                          policy_net, target_net, 16000, target_net_update_steps, min_samples_for_training, 
                          encoding_function)
torch.save(policy_net.state_dict(), 'models/new_policy_net_16.pt')

### Initialize the policy network
policy_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)

### Initialize the target network with the same weights of the policy network
target_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

def encoding_function(est_state):
    model_mq._list_of_quantizers[3].eval()
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[3](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,embedding_dim*num_feat))
    return state_encoded

policy_net = train_policy_ema(256, 
                          policy_net, target_net, 16000, target_net_update_steps, min_samples_for_training, 
                          encoding_function)
torch.save(policy_net.state_dict(), 'models/new_policy_net_8.pt')

### Initialize the policy network
policy_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)

### Initialize the target network with the same weights of the policy network
target_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

def encoding_function(est_state):
    model_mq._list_of_quantizers[4].eval()
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[4](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,embedding_dim*num_feat))
    return state_encoded

policy_net = train_policy_ema(256, 
                          policy_net, target_net, 16000, target_net_update_steps, min_samples_for_training, 
                          encoding_function)
torch.save(policy_net.state_dict(), 'models/new_policy_net_4.pt')

### Initialize the policy network
policy_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)

### Initialize the target network with the same weights of the policy network
target_net = DQN_SMALL(embedding_dim*num_feat,action_space_dim)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

def encoding_function(est_state):
    model_mq._list_of_quantizers[5].eval()
    vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
    _, state_encoded, _, encodings = model_mq._list_of_quantizers[5](vq_output_eval, reset = False)
    state_encoded = torch.reshape(state_encoded, (1,embedding_dim*num_feat))
    return state_encoded

policy_net = train_policy_ema(256, 
                          policy_net, target_net, 16000, target_net_update_steps, min_samples_for_training, 
                          encoding_function)
torch.save(policy_net.state_dict(), 'models/new_policy_net_2.pt')