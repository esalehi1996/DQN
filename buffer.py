from copy import deepcopy
import random
import torch
import numpy as np
# from SumTree import SumTree,MinTree
import torch.nn.functional as F


import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.write = 0

    # update to the root node

    def reset(self):
        self.tree = numpy.zeros(2 * self.capacity - 1)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        # print(self.write,p)
        # print(self.tree)
        idx = self.write + self.capacity - 1

        # self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        # print(idx)
        dataIdx = idx - self.capacity + 1

        # return (idx, self.tree[idx], self.data[dataIdx])
        return (idx, self.tree[idx], dataIdx)

class MinTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.ones(2 * capacity - 1) * numpy.inf
        self.n_entries = 0
        self.write = 0

    # update to the root node

    def reset(self):
        self.tree = numpy.ones(2 * self.capacity - 1) * numpy.inf
        self.n_entries = 0
        self.write = 0

    def min(self):
        return self.tree[0]

    def add(self , p):
        idx = self.write + self.capacity - 1

        # self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):

        self.tree[idx] = p
        self._propagate(idx)

    def _propagate(self, idx):
        parent = (idx - 1) // 2

        self.tree[parent] = min(self.tree[2 * parent + 1] , self.tree[2 * parent + 2])

        if parent != 0:
            self._propagate(parent)








class buffer:
    def __init__(self, capacity, obs_dim, act_dim, args):
        self.capacity = capacity
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = args['gamma']
        self.forward_len = args['forward_len']
        self.batch_size = args['batch_size']
        if args['PER']:
            self.PER = True
            self.SumTree = SumTree(capacity)
            self.MinTree = MinTree(capacity)
            self.PER_e = args['PER_e']
            self.PER_a = 0.6
            self.PER_beta = 0.4
            total_updates = args['num_steps'] / args['rl_update_every_n_steps']
            self.PER_beta_increment_per_sampling = (1 - self.PER_beta)/total_updates
        else:
            self.PER = False

        # if args['alg_type'] == 'full_obs':
        self.buffer_obs = np.zeros([self.capacity, self.obs_dim], dtype=np.float32)
        self.buffer_next_obs = np.zeros([self.capacity, self.obs_dim], dtype=np.float32)
        # elif args['alg_type'] == 'frame_stack':
        #     self.buffer_obs = np.zeros([self.capacity, args['frame_stacking_length'] , self.obs_dim], dtype=np.float32)
        #     self.buffer_next_obs = np.zeros([self.capacity, args['frame_stacking_length'] ,self.obs_dim], dtype=np.float32)
        # else:
        #     assert False
        self.buffer_action = np.zeros([self.capacity], dtype=np.int32)

        self.buffer_rewards = np.zeros([self.capacity], dtype=np.float32)
        self.buffer_final_flag = np.ones([self.capacity], dtype=np.int32)
        self.buffer_gammas = np.zeros([self.capacity], dtype=np.float32)
        self.position = 0

        self.full = False

    def reset(self, seed):
        random.seed(seed)

        self.position = 0


        self.full = False
        if self.PER is True:
            self.SumTree.reset()
            self.MinTree.reset()
            self.PER_beta = 0.4

    def sample(self, batch_size):



        if self.PER is False:
            tmp = self.position
            if self.full:
                tmp = self.capacity
            idx = np.random.choice(tmp, batch_size, replace=False)
        else:
            idx = np.zeros(batch_size, dtype=int)
            tree_idx = np.zeros(batch_size, dtype=int)
            priorities = np.zeros(batch_size)

            segment = self.SumTree.total() / batch_size

            self.PER_beta = np.min([1., self.PER_beta + self.PER_beta_increment_per_sampling])

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)

                s = random.uniform(a, b)

                # print(i, a, b , s)
                (id, p, didx) = self.SumTree.get(s)
                priorities[i] = p
                idx[i] = didx
                tree_idx[i] = id

            sampling_probabilities = priorities / self.SumTree.total()
            p_min = self.MinTree.min() / self.SumTree.total()
            is_max = np.power(self.SumTree.n_entries * p_min, -self.PER_beta)
            is_weight = np.power(self.SumTree.n_entries * sampling_probabilities, -self.PER_beta)
            is_weight /= is_max




        batch_obs = self.buffer_obs[idx,:]
        batch_next_obs = self.buffer_next_obs[idx,:]
        batch_action = self.buffer_action[idx]
        batch_rewards = self.buffer_rewards[idx]
        batch_gammas = self.buffer_gammas[idx]
        batch_final_flag = self.buffer_final_flag[idx]


        if self.PER is True:
            return batch_obs , batch_next_obs , batch_action , batch_rewards , batch_gammas , batch_final_flag , tree_idx , is_weight
        else:
            return batch_obs, batch_next_obs, batch_action, batch_rewards, batch_gammas, batch_final_flag

    def push(self, ep_obs, ep_actions, ep_rewards , agent):


        # print(len(ls_obs))

        # ep_obs = ls_obs
        # ep_actions = [i for i in range(10)]
        # ep_rewards = [i + 0.1 for i in range(10)]

        # print(len(ep_obs))
        # print(ep_actions)
        # print(ep_rewards)



        episode_len = len(ep_obs)
        start_index = self.position
        buffer_fill = False



        # for i in range(episode_len):
        #     print(i,sum_rewards(ep_rewards[i:min(i+self.forward_len,episode_len-1)]  , self.gamma) , ep_rewards[i:min(i+self.forward_len,episode_len-1)])


        for i in range(episode_len):
            # print('-------------',i,'---------------')

            self.buffer_obs[self.position,:] = ep_obs[i]
            self.buffer_action[self.position] = ep_actions[i]
            if i == episode_len - 1:
                self.buffer_final_flag[self.position] = 0
            self.buffer_next_obs[self.position,:] = ep_obs[min(i + self.forward_len,episode_len-1)]
            self.buffer_gammas[self.position] = self.gamma ** min(self.forward_len, episode_len - 1 - i)
            if i == episode_len - 1:
                self.buffer_rewards[self.position] = ep_rewards[i]
            else:
                self.buffer_rewards[self.position] = sum_rewards(ep_rewards[i:min(i+self.forward_len,episode_len-1)] , self.gamma)




            if self.full is False and self.position + 1 == self.capacity:
                self.full = True
            if self.position + 1 == self.capacity:
                buffer_fill = True
            # if self.position + 1 == self.capacity:
            #     buffer_fill = True

            self.position = (self.position + 1) % self.capacity

        # print(self.position)
        # print(self.buffer_action[:self.position])
        # print(self.buffer_rewards[:self.position])
        # print(self.buffer_rewards[:self.position])
        # print(self.buffer_final_flag[:self.position])
        # assert False

        # print(start_index , self.position_r2d2 , len(hidden_list))
        if self.PER is True:
            if buffer_fill is False:
                input_vals = {
                    "batch_obs" : self.buffer_obs[start_index:self.position, :],
                    "batch_next_obs" : self.buffer_next_obs[start_index:self.position, :] ,
                    "batch_action" : self.buffer_action[start_index:self.position] ,
                    "batch_rewards": self.buffer_rewards[start_index:self.position],
                    "batch_gammas": self.buffer_gammas[start_index:self.position],
                    "batch_final_flag": self.buffer_final_flag[start_index:self.position],
                }
            else:
                # print('filled')
                input_vals = {
                    "batch_obs": np.concatenate((self.buffer_obs[start_index:, :],self.buffer_obs[:self.position, :])),
                    "batch_next_obs": np.concatenate((self.buffer_next_obs[start_index:, :],self.buffer_next_obs[:self.position, :])),
                    "batch_action": np.concatenate((self.buffer_action[start_index:],self.buffer_action[:self.position])),
                    "batch_rewards": np.concatenate((self.buffer_rewards[start_index:],self.buffer_rewards[:self.position])),
                    "batch_gammas": np.concatenate((self.buffer_gammas[start_index:],self.buffer_gammas[:self.position])),
                    "batch_final_flag": np.concatenate((self.buffer_final_flag[start_index:],self.buffer_final_flag[:self.position]))
                    }
                # print(input_vals['batch_learn_hist'].shape)
                # print(self.buffer_learning_history[start_index:, :, :].shape,self.buffer_learning_history[:self.position_r2d2, :, :].shape )



            # print(input_vals)

            priorities = agent.compute_priorities(episode_len , input_vals)

            # print(priorities)

            priorities = (priorities + self.PER_e) ** self.PER_a

            # print(priorities)

            for p in priorities:
                # print(p)
                self.SumTree.add(p)
                self.MinTree.add(p)


    def update_priorities(self,tree_idx,priorities):
        # print(priorities)
        priorities = (priorities + self.PER_e) ** self.PER_a


        for p,idx in zip(priorities,tree_idx):
            # print(p)
            self.SumTree.update(idx,p)
            self.MinTree.update(idx,p)


    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.position




def sum_rewards(reward_list, gamma):
    ls = [reward_list[i] * gamma ** i for i in range(0,len(reward_list))]
    return sum(ls)
