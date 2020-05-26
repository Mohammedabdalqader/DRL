import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[400,300]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(state_size)
        self.bn2 = nn.BatchNorm1d(hidden_units[0])
        self.bn3 = nn.BatchNorm1d(hidden_units[1])
        
        # Hidden Layers 
        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.bn1(states)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[400,300],v_min=-10.0,v_max=10.0,num_atoms=51):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        
        # Hidden Layers
        self.bn = nn.BatchNorm1d(state_size)

        self.fcs1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + action_size , hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], num_atoms)
        
        
        #self.z_atoms = np.linspace(v_min, v_max, num_atoms)
        delta = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x  = self.bn(states)
        x_ = F.relu(self.fcs1(x))
        x  = torch.cat((x_, actions), dim=1)
        x  = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)