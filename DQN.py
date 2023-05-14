
import os

from models import *
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import numpy as np
import math
import random

class DQN(object):
    def __init__(self, env, args):


        self.args = args

        action_space = env.action_space

        self.obs_dim = env.reset()['image'].reshape(-1).shape[0]
        # self.state_size = obs_size

        self.act_dim = action_space.n
        self.gamma = args['gamma']

        self.target_update_interval = args['target_update_interval']
        self.action_space = action_space


        self.device = torch.device("cuda" if args['cuda'] else "cpu")

        # print(self.obs_dim,action_space.n,args['hidden_size'])
        # assert False



        self.q = QNetwork(self.obs_dim, action_space.n, args['hidden_size'] , False).to(device=self.device)
        self.q_target = QNetwork(self.obs_dim, action_space.n, args['hidden_size'] , False).to(self.device)
        self.q_target.train()
        self.q.train()


        self.q_optim = Adam(self.q.parameters(), lr=args['lr'])


        hard_update(self.q_target, self.q)

        self.update_to_q = 0
        self.eps_greedy_parameters = {
            "EPS_START" : args['EPS_start'],
            "EPS_END" : args['EPS_end'] ,
            "EPS_DECAY" : args['EPS_decay']
        }
        self.env_steps = 0




    def select_action(self, state , evaluate):



        with torch.no_grad():
            if evaluate is False:
                self.env_steps += 1
            eps_threshold = self.eps_greedy_parameters['EPS_END'] + (
                            self.eps_greedy_parameters['EPS_START'] - self.eps_greedy_parameters['EPS_END']) * \
                                    math.exp(-1. * self.env_steps / self.eps_greedy_parameters['EPS_DECAY'])

            sample = random.random()
            if sample < eps_threshold and evaluate is False:
                return torch.tensor([[random.randrange(self.act_dim)]],dtype=torch.long).cpu().numpy()[0][0]
            # print(torch.from_numpy(state).float())

            state = torch.from_numpy(state).float().to(device = self.device)

            qf = self.q(state)

            max_ac = qf.max(0)[1]


        return max_ac.detach().cpu().numpy()

    def compute_priorities(self,episode_len , input_vals):
        batch_obs = torch.from_numpy(input_vals['batch_obs']).to(self.device)
        batch_next_obs = torch.from_numpy(input_vals['batch_next_obs']).to(self.device)
        batch_act = torch.from_numpy(input_vals['batch_action']).to(self.device)
        batch_final_flag = torch.from_numpy(input_vals['batch_final_flag']).to(self.device)
        batch_gammas = torch.from_numpy(input_vals['batch_gammas']).to(self.device)
        batch_rewards = torch.from_numpy(input_vals['batch_rewards']).to(self.device)

        self.q.train()

        # print(batch_obs.shape,batch_obs.dtype)

        qf = self.q(batch_obs)

        # print(qf.shape)
        # print(batch_act.view(-1, 1).shape)
        # print(batch_act)

        qf = qf.gather(1, batch_act.view(-1, 1).long())

        # print(qf.shape)

        with torch.no_grad():
            qf_target = self.q_target(batch_next_obs)
            # print(qf_target.shape)

            max_idx = self.q(batch_next_obs).max(1)[1]

            qf_target = qf_target.gather(1, max_idx.view(-1, 1).long())
            next_q_value = batch_rewards + batch_gammas * batch_final_flag * qf_target.view(-1)

        priorities = torch.abs(qf.view(-1) - next_q_value)

        return priorities.detach().cpu().numpy()




    def update_parameters(self, memory, batch_size):

        if self.args['PER'] is True:
            batch_obs, batch_next_obs, batch_act, batch_rewards, batch_gammas, batch_final_flag , tree_idx, is_weight = memory.sample(batch_size)
        else:
            batch_obs, batch_next_obs, batch_act, batch_rewards, batch_gammas, batch_final_flag  = memory.sample(
                batch_size)



        batch_obs = torch.from_numpy(batch_obs).to(self.device)
        batch_next_obs = torch.from_numpy(batch_next_obs).to(self.device)
        batch_act = torch.from_numpy(batch_act).to(self.device)
        batch_final_flag = torch.from_numpy(batch_final_flag).to(self.device)
        batch_gammas = torch.from_numpy(batch_gammas).to(self.device)
        batch_rewards = torch.from_numpy(batch_rewards).to(self.device)


        # print(batch_obs.shape)
        # print(batch_next_obs.shape)
        # print(batch_rewards)
        # print(batch_act)
        # print(batch_final_flag)


        self.q.train()

        # print(batch_obs.shape,batch_obs.dtype)


        qf = self.q(batch_obs)

        # print(qf.shape)
        # print(batch_act.view(-1, 1).shape)
        # print(batch_act)

        qf = qf.gather(1, batch_act.view(-1, 1).long())

        # print(qf.shape)

        with torch.no_grad():
            qf_target = self.q_target(batch_next_obs)
            # print(qf_target.shape)

            max_idx = self.q(batch_next_obs).max(1)[1]



            qf_target = qf_target.gather(1, max_idx.view(-1, 1).long())
            next_q_value = batch_rewards + batch_gammas * batch_final_flag * qf_target.view(-1)

        qf_loss = F.mse_loss(qf.view(-1), next_q_value, reduce=False)
        if self.args['PER'] is True:
            is_weight = torch.from_numpy(is_weight).to(self.device)
            qf_loss = qf_loss * is_weight
            priorities = torch.abs(qf.view(-1) - next_q_value)


        qf_loss = qf_loss.mean()

        self.q_optim.zero_grad()

        qf_loss.backward()

        self.q_optim.step()
        self.update_to_q += 1

        if self.args['PER'] is True:
            memory.update_priorities(tree_idx, priorities.detach().cpu().numpy())

        if self.update_to_q % self.target_update_interval == 0:
            hard_update(self.q_target, self.q)





        return qf_loss.item()