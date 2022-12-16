# Memory for Experience Replay
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = 'cuda'

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None) # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = Transition(*args)  
        self.position = (self.position + 1) % self.capacity # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 

    def __len__(self): 
        return len(self.memory)

def choose_action_softmax(net, state, temperature):
    
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)
    
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)
        
    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = (nn.functional.softmax(net_out / temperature, dim=1)).cpu().numpy()
    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    
    
    
    action = np.random.choice(all_possible_actions, p=softmax_out.squeeze()) # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    
    return action, net_out.cpu().numpy()

    
def choose_action_epsilon_greedy(net, state, epsilon):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8), int(screen_width*0.2):int(screen_width*0.8)]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    
    screen = T.ToPILImage()(screen)
    screen = screen.convert('L')
    #screen = T.Resize(80, interpolation=T.InterpolationMode.BICUBIC)(screen)
    screen = T.ToTensor()(screen)
    return screen

class DQN_SMALL(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        
        self.sample = True
        
        #Dfine the encoder    
        self.linear1 = nn.Linear(in_features=n_inputs, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=n_actions)
            
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)
    
def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, temperature):
        
    #transitions = replay_mem.sample(batch_size)
    transitions = replay_mem.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # Create tensors for each element of the batch
    states      = torch.cat(batch.state)
    states = states.to(device)
    actions     = torch.tensor(batch.action, device = device, dtype=torch.int64)
    rewards     = torch.tensor(batch.reward, dtype=torch.float32)
    rewards = rewards.to(device)
    
    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    non_final_next_states = non_final_next_states.to(device)
    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
        target_net.eval()
        q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = (torch.zeros(batch_size)).to(device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1) # Set the required tensor shape
    
    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()
    
    
def train_policy(batch_size, policy_net, target_net, n_epochs, target_net_update_steps, min_samples_for_training, enc):
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset(seed = 0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    ### Define exploration profile
    initial_value = 3
    num_iterations = n_epochs
    exp_decay = np.exp(-np.log(initial_value) / num_iterations * 4) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
    exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]
    
    ### Initialize the optimizer
    optimizer = torch.optim.SGD(policy_net.parameters(), lr=1e-3) # The optimizer will update ONLY the parameters of the policy network
    gamma = 0.97   # gamma parameter for the long term reward

    ### Initialize the loss function (Huber loss)
    loss_fn = nn.SmoothL1Loss()
    h,w = get_screen(env).squeeze().numpy().shape
    replay_mem = ReplayMemory(10000)
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    bar1 = tqdm(range(len(exploration_profile)), desc='Training')
    
    for episode_num in bar1:

        beta = 0.5
        tau = exploration_profile[episode_num]
        # Reset the environment and get the initial state
        state = env.reset(seed = episode_num)
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False
        prev_screen = get_screen(env)
        curr_screen = get_screen(env)

        input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
        input_tensor = input_tensor.to(device)
        est_state = 1-input_tensor
        state_encoded = enc(est_state).detach()
        # Go on until the pole falls off
        while not done:
            # Choose the action following the policy
            action, q_values = choose_action_epsilon_greedy(policy_net, state_encoded, 0.01)

            # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, _, _ = env.step(action)

            prev_screen = curr_screen
            curr_screen = get_screen(env)

            input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
            input_tensor = input_tensor.to(device)

            est_next_state = 1-input_tensor
            next_state_encoded = enc(est_next_state).detach()
                
            # Reward modification for faster learning
            x, x_dot, theta, theta_dot = env.state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            # Update the final score (+1 for each step)
            score += 1
            
                    # Apply penalty for bad state
            if done: # if the pole has fallen down 
                reward += 0
                next_state = None
            
            # Update the replay memory
            replay_mem.push(state_encoded, action, next_state_encoded, reward)

            # Update the network
            if len(replay_mem) > min_samples_for_training:
                update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, tau)

            # Set the current state for the next iteration
            state_encoded = next_state_encoded

        # Update the target network every target_net_update_steps episodes    
        if episode_num % target_net_update_steps == 0:
            target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

        # Print the final score
        bar1.set_postfix({'EPISODE': episode_num + 1, 'FINAL SCORE': score, 'Temperature': tau})
        
    return policy_net

def train_policy_ema(batch_size, policy_net, target_net, n_epochs, target_net_update_steps, min_samples_for_training, enc):
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset(seed = 0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    ### Define exploration profile
    ### Define exploration profile
    initial_value = 1
    num_iterations = n_epochs
    exploration_profile = [initial_value * np.exp(-i * 0.003) for i in range(num_iterations)]
    
    ### Initialize the optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4) # The optimizer will update ONLY the parameters of the policy network
    gamma = 0.97   # gamma parameter for the long term reward
    beta = 0.5
    ### Initialize the loss function (Huber loss)
    loss_fn = nn.SmoothL1Loss()
    h,w = get_screen(env).squeeze().numpy().shape
    replay_mem = ReplayMemory(10000)
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    bar1 = tqdm(range(len(exploration_profile)), desc='Training')
    
    for episode_num in bar1:

        tau = exploration_profile[episode_num]
        # Reset the environment and get the initial state
        state = env.reset(seed = episode_num)
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False
        prev_screen = get_screen(env)
        curr_screen = get_screen(env)

        input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
        input_tensor = input_tensor.to(device)
        est_state = 1-input_tensor
        state_encoded = enc(est_state).detach()
        # Go on until the pole falls off
        while not done:
            # Choose the action following the policy
            action, q_values = choose_action_epsilon_greedy(policy_net, state_encoded, tau)

            # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, _, _ = env.step(action)

            prev_screen = curr_screen
            curr_screen = get_screen(env)

            input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
            input_tensor = input_tensor.to(device)

            est_next_state = 1-input_tensor
            next_state_encoded = enc(est_next_state).detach()
                
            # Reward modification for faster learning
            x, x_dot, theta, theta_dot = env.state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            # Update the final score (+1 for each step)
            score += 1
            
                    # Apply penalty for bad state
            if done: # if the pole has fallen down 
                reward = 0
                next_state = None
                next_state_encoded = None
            # Update the replay memory
            replay_mem.push(state_encoded, action, next_state_encoded, reward)

            # Update the network
            if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                if score%5 == 0:
                    update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, tau)

            # Set the current state for the next iteration
            state_encoded = next_state_encoded

        # Update the target network every target_net_update_steps episodes    
        if episode_num % target_net_update_steps == 0:
            for name, param in target_net.named_parameters():
                param.data = beta*param.data + (1-beta)*policy_net.state_dict()[name]

        # Print the final score
        bar1.set_postfix({'EPISODE': episode_num + 1, 'FINAL SCORE': score, 'Temperature': tau})
        
    return policy_net


class Memory_rec():
    
    def __init__(self,memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)
    
    def __len__(self): 
        return len(self.memory)
    
    def add_episode(self,epsiode):
        self.memory.append(epsiode)
    
    def get_batch(self,bsize,time_step):
        sampled_epsiodes = random.sample(self.memory,bsize)
        batch = []
        for episode in sampled_epsiodes:
            point = np.random.randint(0,len(episode)-time_step)
            batch.append(episode[point:point+time_step])
        return batch

    

def train_policy_ema_rec(batch_size, policy_net, target_net, n_epochs, target_net_update_steps, min_samples_for_training, enc):
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset(seed = 0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    ### Define exploration profile
    initial_value = 3
    num_iterations = n_epochs
    exp_decay = np.exp(-np.log(initial_value) / num_iterations * 4) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
    exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]
    
    ### Initialize the optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4) # The optimizer will update ONLY the parameters of the policy network
    gamma = 0.97   # gamma parameter for the long term reward
    beta = 0.5
    ### Initialize the loss function (Huber loss)
    loss_fn = nn.SmoothL1Loss()
    h,w = get_screen(env).squeeze().numpy().shape
    replay_mem = Memory_rec(10000)
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    bar1 = tqdm(range(len(exploration_profile)), desc='Training')
    
    epsilon = 0.01
    
    for episode_num in bar1:

        tau = exploration_profile[episode_num]
        # Reset the environment and get the initial state
        state = env.reset(seed = episode_num)
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False
        prev_screen = get_screen(env)
        curr_screen = get_screen(env)

        input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
        input_tensor = input_tensor.to(device)
        est_state = 1-input_tensor
        state_encoded = enc(est_state).detach()
        # Go on until the pole falls off
        
        episode = []
        
        hidden,c = torch.zeros(1,1,512), torch.zeros(1,1,512)
        hidden = hidden.to(device)
        c = c.to(device)
        
        while not done:
            # Choose the action following the policy
            # Evaluate the network output from the current state
            with torch.no_grad():
                policy_net.eval()
                s = torch.tensor(state_encoded, dtype=torch.float32) # Convert the state to tensor
                net_out, (hidden,c) = policy_net(s.view(1,1,512), hidden,c)

            # Get the best action (argmax of the network output)
            best_action = int(net_out.argmax())
            # Get the number of possible actions
            action_space_dim = net_out.shape[-1]

            # Select a non optimal action with probability epsilon, otherwise choose the best action
            if random.random() < epsilon:
                # List of non-optimal actions
                non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
                # Select randomly
                action = random.choice(non_optimal_actions)
            else:
                # Select best action
                action = best_action

            # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, _, _ = env.step(action)

            prev_screen = curr_screen
            curr_screen = get_screen(env)

            input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
            input_tensor = input_tensor.to(device)

            est_next_state = 1-input_tensor
            next_state_encoded = enc(est_next_state).detach()
                
            # Reward modification for faster learning
            x, x_dot, theta, theta_dot = env.state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            # Update the final score (+1 for each step)
            score += 1
            
                    # Apply penalty for bad state
            if done: # if the pole has fallen down 
                reward = 0
                next_state = None
                next_state_encoded = None
            # Update the replay memory
            episode.append((state_encoded, action, reward, next_state_encoded))

            # Update the network
            if len(replay_mem) > 256: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                if score % 5 == 0:
                    update_step_rec(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, tau)

            # Set the current state for the next iteration
            state_encoded = next_state_encoded

        # Update the target network every target_net_update_steps episodes  
        replay_mem.add_episode(episode)
        
        if episode_num % target_net_update_steps == 0:
            target_net.load_state_dict(policy_net.state_dict()) 
            #for name, param in target_net.named_parameters():
            #    param.data = beta*param.data + (1-beta)*policy_net.state_dict()[name]

        # Print the final score
        bar1.set_postfix({'EPISODE': episode_num + 1, 'FINAL SCORE': score, 'Temperature': tau})
        
    return policy_net

    
def update_step_rec(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, temperature):
    policy_net.train()
    BATCH_SIZE = 256
    TIME_STEP = 5
    
    batch = replay_mem.get_batch(bsize=BATCH_SIZE,time_step=TIME_STEP)
            
    current_states = []
    acts = []
    rewards = []
    next_states = []
            
    for b in batch:
        cs,ac,rw,ns = [],[],[],[]
        for element in b:
            cs.append(element[0])
            ac.append(element[1])
            rw.append(element[2])
            ns.append(element[3])
        current_states.append(torch.cat(cs,1))
        acts.append(ac)
        rewards.append(rw)
        next_states.append(torch.cat(ns,1))
            
    
    acts = np.array(acts)
    rewards = np.array(rewards)
            
    torch_current_states = torch.cat(current_states).to(device)
    torch_acts = torch.from_numpy(acts).long().to(device)
    torch_rewards = torch.from_numpy(rewards).float().to(device)
    torch_next_states = torch.cat(next_states).to(device)
    
    h,c = torch.zeros(1,BATCH_SIZE,512).to(device), torch.zeros(1,BATCH_SIZE,512).to(device)
    q_values_next, _ = target_net(torch_next_states, h, c)
    q_values_next = q_values_next.detach()
    next_state_max_q_values = q_values_next.max(dim=1)[0]
    
    expected_state_action_values = torch_rewards[:,TIME_STEP-1] + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    
    q_values, _ = policy_net(torch_current_states, h, c)
    state_action_values = q_values.gather(1, torch_acts[:,TIME_STEP-1].unsqueeze(1))
    
    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()
    
class DQN_SMALL_REC(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()

        self.lstm = nn.LSTM(n_inputs, n_inputs, batch_first = True)
        self.linear1 = nn.Linear(in_features=n_inputs, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=n_actions)
            
    def forward(self, x, h, c):
        
        x, (h_n, c_n) = self.lstm(x, (h, c))
        x = F.relu(x[:,-1,:])
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x) , (h_n, c_n)

from Quantizer import Encoder
    
class Sensor_Q_Net(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        self.encoder = Encoder(2, 128,2,32)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=n_actions)
            
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.linear1(self.flatten(x)))
        x = F.relu(self.linear2(x))
        return self.output(x)  
    
class PhysicalValueRegressor(nn.Module):
    def __init__(self, n_inputs, n_output):
        super().__init__()
        self.linear1 = nn.Linear(in_features=512, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=n_output)
            
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)    

def train_regressor(reg, enc, h, w):
    optimizer = torch.optim.Adam(reg.parameters(), lr=1e-4)
    N_episodes = 1000
    replay_mem = ReplayMemory(10000)
    loss_fn = nn.MSELoss()
    bar1 = tqdm(range(N_episodes), desc='Training')
    reg.to(device)
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset()
    print_loss = 0

    for i in bar1:
        state= env.reset()
        prev_screen = get_screen(env)
        done = False
        losses = []

        while not done:

            true_state, _, done, _, _ = env.step(np.random.randint(2))
            curr_screen = get_screen(env)
            input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
            input_tensor = input_tensor.to(device)
            est_state = 1-input_tensor
            encoded_state = enc(est_state).detach()
            replay_mem.push(encoded_state.cpu(), None, (torch.tensor(true_state)).unsqueeze(0), None)

            prev_screen = curr_screen

            if len(replay_mem) > 1000:
                #transitions = replay_mem.sample(batch_size)
                transitions = replay_mem.sample(512)
                batch = Transition(*zip(*transitions))
                # Create tensors for each element of the batch
                states      = torch.cat(batch.state)
                states = states.to(device)

                true_states = torch.cat(batch.next_state)
                true_states = true_states.to(device)

                loss = loss_fn(reg(states).squeeze(1), true_states)
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
                #nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
                optimizer.step()
                losses.append(loss.item())
                print_loss = np.array(losses).mean()   
        bar1.set_postfix({'EPISODE': i + 1, 'LOSS': print_loss})
    return reg

# procedure for training the sensor (technical level)
from tqdm import tqdm

def train_sensor_policy_PSNR(sensor_policy_net, beta, N_episodes, model_mq,h,w):
    optimizer = torch.optim.Adam(sensor_policy_net.parameters(), lr=1e-4)

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset()
    replay_mem = ReplayMemory(10000)
    loss_fn = nn.SmoothL1Loss()
    bar1 = tqdm(range(N_episodes), desc='Training')
    sensor_policy_net.to(device)
    epsilon_values = np.exp(-np.linspace(0,5,N_episodes))
    average_reward = []
    average_length = []
    mean = 0
    mean_length = 0
    num_feat = 8
    for i in bar1:

        scores = []
        actions = []

        epsilon = epsilon_values[i]
        state= env.reset()
        prev_screen = get_screen(env)
        done = False
        while not done:
            curr_screen = get_screen(env)
            input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
            input_tensor = input_tensor.to(device)
            est_state = 1-input_tensor
            action, q_values = choose_action_epsilon_greedy(sensor_policy_net, est_state, epsilon)
            _, x_recon, _ =  model_mq(est_state, 2**(action+1))
            PSNR = 10*np.log10(1/(F.mse_loss(est_state.cpu(), x_recon.cpu())).detach().numpy())
            reward = PSNR - beta*(action+1)
            replay_mem.push(est_state.cpu(), action, None, reward)
            _, _, done, _, _ = env.step(np.random.randint(2))
            prev_screen = curr_screen
            scores.append(reward)
            actions.append((action+1)*num_feat)

            if len(replay_mem) > 1000:
                update_step_PSNR(sensor_policy_net, replay_mem, optimizer, loss_fn, 128)
                
        mean = np.array(scores).mean()
        mean_length = np.array(actions).mean()
        average_reward.append(mean)
        average_length.append(mean_length)
        bar1.set_postfix({'EPISODE': i + 1, 'MEAN REWARD': mean, 'MEAN LENGTH': mean_length})
    return sensor_policy_net, average_reward, average_length

def update_step_PSNR(policy_net, replay_mem, optimizer, loss_fn, batch_size):
        
    #transitions = replay_mem.sample(batch_size)
    transitions = replay_mem.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # Create tensors for each element of the batch
    states      = torch.cat(batch.state)
    states = states.to(device)
    actions     = torch.tensor(batch.action, device = device, dtype=torch.int64)
    rewards     = torch.tensor(batch.reward, dtype=torch.float32)
    rewards = rewards.to(device)
    
    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))
    # Compute the Huber loss
    loss = loss_fn(state_action_values, rewards.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()
    

def train_sensor_policy_semantically(sensor_policy_net, beta, N_episodes, model_mq, regressors,h,w):
    optimizer = torch.optim.Adam(sensor_policy_net.parameters(), lr=1e-4)

    
    Ks = [5,4,3,2,1,0]
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env.reset()
    replay_mem = ReplayMemory(10000)
    loss_fn = nn.SmoothL1Loss()
    bar1 = tqdm(range(N_episodes), desc='Training')
    sensor_policy_net.to(device)
    epsilon_values = np.exp(-np.linspace(0,5,N_episodes))
    average_reward = []
    average_length = []
    mean = 0
    mean_length = 0
    num_feat = 8
    embedding_dim = 64
    for i in bar1:

        scores = []
        actions = []

        epsilon = epsilon_values[i]
        state= env.reset()
        prev_screen = get_screen(env)
        done = False
        while not done:
            curr_screen = get_screen(env)
            input_tensor = torch.reshape(torch.cat((prev_screen, curr_screen)), (1,2,h,w))
            input_tensor = input_tensor.to(device)
            est_state = 1-input_tensor
            action, q_values = choose_action_epsilon_greedy(sensor_policy_net, est_state, epsilon)
            true_state = env.state
            vq_output_eval = model_mq._pre_vq_conv(model_mq._encoder(est_state))
            _, state_encoded, _, encodings = model_mq._list_of_quantizers[Ks[action]](vq_output_eval, reset = False)
            state_encoded = torch.reshape(state_encoded, (1,1,embedding_dim*num_feat))
            es = regressors[Ks[action]](state_encoded.cpu())
            mse = F.mse_loss(es.squeeze(0), torch.tensor(true_state).unsqueeze(0))
            reward = -mse - beta*(action+1)
            replay_mem.push(est_state.cpu(), action, None, reward.item())
            _, _, done, _, _ = env.step(np.random.randint(2))
            prev_screen = curr_screen
            scores.append(reward.item())
            actions.append((action+1)*num_feat)

            if len(replay_mem) > 1000:
                update_step_PSNR(sensor_policy_net, replay_mem, optimizer, loss_fn, 128)
                mean = np.array(scores).mean()
                mean_length = np.array(actions).mean()

        average_reward.append(mean)
        average_length.append(mean_length)
        bar1.set_postfix({'EPISODE': i + 1, 'MEAN REWARD': mean, 'MEAN LENGTH': mean_length})
    return sensor_policy_net, average_reward, average_length

    
class Sensor_Q_Net_effective(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        self.encoder = Encoder(2, 128,2,32)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=n_actions)
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)  

def choose_action_epsilon_greedy_q(net, state, q, epsilon):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state, q)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()
#def train_sensor_policy_effectively(sensor_policy_net, beta, N_episodes, model_mq, regressors,h,w):