import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import random

class Network(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super(Network, self).__init__()
        self.num_of_inputs = num_of_inputs
        self.num_of_actions = num_of_actions
        self.fc1 = nn.Linear(num_of_inputs, 30)
        self.fc2 = nn.Linear(30, num_of_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        #We do not apply an activation function to the output layer
        h = self.fc2(x)
        return h

#Each field will be a tensor or a tuple of tensors
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def add(self, *args):
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, size):
        return random.sample(self.memory, size)
    
    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 75

class Brain:
    def __init__(self, num_of_inputs, num_of_actions):
        self.policy_net = Network(num_of_inputs, num_of_actions)
        self.target_net = Network(num_of_inputs, num_of_actions)
        self.memory = ExperienceBuffer(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss = []
        
    def select_action(self, state):
        actions = F.softmax(self.policy_net(state)*4, dim=1)
        action = actions.multinomial(1)
        return action
        
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        #Transpose
        batch = Transition(*zip(*batch))
        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        final_state_mask = [x is not None for x in batch.next_state]
        
        non_final_states = torch.cat([x for x in batch.next_state if x is not None])
    
        next_state_values = torch.zeros(BATCH_SIZE)
        #max Q(s,a) for each
        next_state_values[final_state_mask] = self.target_net(non_final_states).detach().max(1)[0]
        
        #(max Q(s,a) * Gamma) + reward
        expected_next_state_values = (next_state_values * GAMMA) + reward
        
        loss = F.smooth_l1_loss(q_values, expected_next_state_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.loss.append(loss)
        if len(self.loss) > 1000:
            del self.loss[0]
        self.optimizer.step()
        
    

