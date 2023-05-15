
from torch.nn import ELU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from abc import ABC

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim  ,  frame_stacking = False ):
        super(QNetwork, self).__init__()


        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture


        self.apply(weights_init_)

    def forward(self, state):
        # print(xu.shape)

        x1 = F.elu(self.linear1(state))
        x1 = F.elu(self.linear2(state))
        x1 = F.elu(self.linear3(x1))
        x1 = self.linear4(x1)

        return x1



def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
