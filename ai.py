import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import random

#Set up the neural network
#Don't need to worry much about this
class Network(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super(Network, self).__init__()
        #30 hidden neurons
        self.fc1 = nn.Linear(num_of_inputs, 30)
        self.fc2 = nn.Linear(30, num_of_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x
        
#Create an object that represents a transition from one state to another
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

#Our experience buffer
class ExperienceBuffer:
    #Initialize it to hold capacity amount of items
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    #This function allows us to add a transition to our memory
    def add(self, *args):
        #Add the transition to memory
        self.memory.append(Transition(*args))
        #If the memory is exceeding capacity
        if len(self.memory) > self.capacity:
            #delete the earliest memory
            del self.memory[0]
    
    #Returns size amount of memories
    def sample(self, size):
        """Returns [size] amount of random transitions"""
        return random.sample(self.memory, size)
    
    #How many memories we have stored
    def __len__(self):
        return len(self.memory)
    
    
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 100

class Brain:
    def __init__(self, num_of_inputs, num_of_actions):
        self.policy_net = Network(num_of_inputs, num_of_actions)
        self.target_net = Network(num_of_inputs, num_of_actions)
        self.memory = ExperienceBuffer(10000)
        #The optimizer is here to help us update our neural network weights
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=0.001)
    
    def select_action(self, state):
        #Select an action using softmax
        #Softmax makes sure the array sums to 1
        #We can use each element as a probability
        actions = F.softmax(self.policy_net(state),dim=1)
        #Using multinomial to select 1 action with the calculated probabilities
        #and return that action
        action = actions.multinomial(1)
        return action
    
    def optimize(self):
        #If we have too little memories, just don't learn
        #Not enough memories to learn anything meaningful
        if len(self.memory) < BATCH_SIZE:
            return
        
        #Get a batch of transitions
        batch = self.memory.sample(BATCH_SIZE)
        
        #Transpose that batch and wrap it in a Transition
        #For example:
        #  If we have 3 transitions in our batch (yes action is a value wrapped in 2 arrays)
        #    Transition(state=[0,1,2,3,4], action=[[1]], next_state=[1,2,3,4,5], reward=[1])
        #    Transition(state=[0,1,2,3,4], action=[[1]], next_state=[1,2,3,4,5], reward=[1])
        #    Transition(state=[0,1,2,3,4], action=[[1]], next_state=[1,2,3,4,5], reward=[1])
        #  We then use *batch on it, and we get
        #  zip(Transition(state=[0,1,2,3,4], action=[[1]], next_state=[1,2,3,4,5], reward=[1]), 
        #      Transition(state=[0,1,2,3,4], action=[[1]], next_state=[1,2,3,4,5], reward=[1]), 
        #      Transition(state=[0,1,2,3,4], action=[[1]], next_state=[1,2,3,4,5], reward=[1]))
        #  What zip does is combines each of the corresponding elements into a tuple
        #  So all the states go into one tuple, the actions into another tuple, the rewards into another, and so on
        #    [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),    states
        #    ([[1]], [[1]], [[1]]),                                   actions
        #    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),     next states
        #    ([1], [1], [1])]                                         rewards
        #  Now we take all of that, and unzip it again, so we get
        #  Transition(state=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        #             action=([[1]], [[1]], [[1]]),
        #             next_state=([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        #             reward=([1], [1], [1])])
        #  Now we have a big transition that holds the all of the states, next_states, rewards, and actions
        #If still confused read this:
        #https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343
        #I suggest reading about the zip function as well, just google "zip python"
        batch = Transition(*zip(*batch))
        
        #Retrieve the states, actions, and reward from the big transition we just made
        #And turn them into tensors (special datatype for neural networks) with torch.cat
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        
        #Pass the states into the policy network to get Q(s,a) for each action
        #Now for each state, we just got 3 q-values
        #We only want the q-value for the action that was actually taken, so we can get that
        #using .gather
        #Suggested reading:
        #https://stackoverflow.com/a/54706716/10850832
        #Includes picture as well
        #Really helpful
        q_values = self.policy_net(states).gather(1, actions)
        
        #Create a list or Trues and Falses
        #Take each next_state in the batch, and if it is not a final state (None is a final state)
        #Then turn it into True, else False
        #[None, [[1,2,3,4,5]], None, [[1,2,3,4,5]] -> [False, True, False, True]
        final_state_mask = [x is not None for x in batch.next_state]
        #Keep only the next_states that are not None
        #torch.cat it to turn it into a tensor
        non_final_states = torch.cat([x for x in batch.next_state if x is not None])
        
        #Intialize a tensor with BATCH_SIZE zeroes
        next_state_values = torch.zeros(BATCH_SIZE)
        #Using the mask we made of Trues and Falses
        #Example: mask = [True, True, False, True, False], next_state_values = [0,0,0,0,0]
        #next_state_values[mask] = [3, 4, 1]
        #next_state_values is now [3, 4, 0, 1, 0]
        #Basically, all the indexes which correspond to False in the mask do not change their value
        #But those who's indexes correspond to True do
        #Okay, with that in mind we pass in the non_final_states into our target network,
        #and then we take the maximum q-value
        #This is max(Q(s,a))
        #.detach() is something that saves memory and improves performance
        # .max(1)[0] does what you think, it takes the maximum q-value for each state
        next_state_values[final_state_mask] = self.target_net(non_final_states).detach().max(1)[0]
        
        #For each value in the next_state_values, we multiply it by the discount factor GAMMA
        #Then we add the reward
        expected_next_state_values = (next_state_values * GAMMA) + reward
        
        #Calculate the loss between the q_values we got
        #And the q_values we expected with the target network
        loss = F.smooth_l1_loss(q_values, expected_next_state_values)
        
        #with the loss, use the optimizer to update the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()