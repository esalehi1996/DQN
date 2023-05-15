import itertools

import numpy as np
import os
# from DQN import agent
# from buffer import buffer
from torch.utils.tensorboard import SummaryWriter
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from buffer import buffer
from DQN import DQN
from PIL import Image
import matplotlib.pyplot as plt
import pickle
def run_exp(args):
    writer = SummaryWriter(args['logdir'])
    list_of_test_rewards_allseeds = []
    list_of_discount_test_rewards_allseeds = []
    list_of_nonzero_reward_count_allseeds = []
    if args['env_name'][:8] == 'MiniGrid':
        env = gym.make(args['env_name'])
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 400
    args['max_env_steps'] = max_env_steps
    if args['alg_type'][:8] == 'full_obs':
        env = FullyObsWrapper(env)
        obs_size = env.reset()['image'].reshape(-1).shape[0]
        print(env.action_space, obs_size , env.reset()['image'].shape)
    else:
        obs_size = env.reset()['image'].reshape(-1).shape[0] * args['frame_stacking_length']
        print(env.action_space, obs_size , env.reset()['image'].shape)
    memory = buffer(args['replay_size'], obs_size, env.action_space.n, args)
    for seed in range(args['num_seeds']):
        list_of_test_rewards = []
        list_of_discount_test_rewards = []
        list_of_nonzero_reward_count = []
        print('-------------------------------------')
        print('seed number '+str(seed)+' running')
        print('-------------------------------------')
        total_numsteps = 0
        # k_steps = 0
        updates = 1
        agent = DQN(env, obs_size , args)

        memory.reset(seed)

        ls_running_rewards = []
        avg_reward = 0
        avg_episode_steps = 0
        avg_q_loss = 0
        k_episode = 0
        num_nonzero_rewards = 0
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            # print('reset')
            state = env.reset()
            ls_obs = []
            ls_obs_ = []
            ls_actions = []
            ls_rewards = []
            action = 0
            reward = 0
            step = 1

            while not done:
                state = state['image'].reshape(-1)
                ls_obs_.append(state)
                if args['alg_type'] == "frame_stack":
                    if step < args['frame_stacking_length']:
                        empty = np.zeros(state.shape[0]*(args['frame_stacking_length']-step))
                        non_empty = np.array(ls_obs_[:step]).reshape(-1)
                        # print(empty.shape,non_empty.shape)
                        state = np.concatenate((empty,non_empty))
                        # print(state.shape)
                    else:
                        state = np.array(ls_obs_[step-args['frame_stacking_length']:step]).reshape(-1)
                        # print(state.shape)

                ls_obs.append(state)


                action = agent.select_action(state , evaluate = False)
                # action = env.action_space.sample()

                next_state, reward, done, _ = env.step(action)  # Step
                ls_actions.append(action)
                ls_rewards.append(reward)
                if reward != 0:
                    num_nonzero_rewards += 1
                if len(memory) > args['batch_size'] and total_numsteps % args['rl_update_every_n_steps'] == 0:
                    q_loss  = agent.update_parameters(memory, args['batch_size'])
                    updates += 1
                    avg_q_loss += q_loss
                episode_steps += 1
                total_numsteps += 1
                step += 1
                episode_reward = reward + episode_reward

                state = next_state
                if total_numsteps % args['logging_freq'] == args['logging_freq']-1:
                    avg_reward , avg_discount_adj_reward = log_test_and_save(env, agent, writer, args, avg_reward, avg_q_loss, updates, k_episode, i_episode, total_numsteps, avg_episode_steps , seed  , num_nonzero_rewards)
                    list_of_test_rewards.append(avg_reward)
                    list_of_discount_test_rewards.append(avg_discount_adj_reward)
                    list_of_nonzero_reward_count.append(num_nonzero_rewards/args['logging_freq'])

                    num_nonzero_rewards = 0
                    avg_reward = 0
                    avg_episode_steps = 0
                    avg_q_loss = 0
                    updates = 0
                    k_episode = 0
                if episode_steps >= max_env_steps:
                    break


            # print('done')
            memory.push(ls_obs, ls_actions, ls_rewards,agent)  # Append transition to memory
            k_episode += 1
            # print(ls_states,ls_actions,ls_rewards)
            ls_running_rewards.append(episode_reward)
            avg_reward = avg_reward + episode_reward
            avg_episode_steps = episode_steps + avg_episode_steps



            if total_numsteps > args['num_steps']:
                break

        # print('making_videoo')
        # if args['env_name'][:8] == 'MiniGrid':
        #     make_video(env,agent,args,seed , state_size)
        list_of_test_rewards_allseeds.append(list_of_test_rewards)
        list_of_discount_test_rewards_allseeds.append(list_of_discount_test_rewards)
        list_of_nonzero_reward_count_allseeds.append(list_of_nonzero_reward_count)



    env.close()
    writer.close()
    arr_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    arr_d_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    arr_count_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    for i in range(args['num_seeds']):
        arr_r[i,:] = np.array(list_of_test_rewards_allseeds[i])
        arr_d_r[i,:] = np.array(list_of_discount_test_rewards_allseeds[i])
        arr_count_r[i,:] = np.array(list_of_nonzero_reward_count_allseeds[i])

    np.save(args['logdir']+'/'+args['exp_name']+'_arr_r',arr_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_d_r', arr_d_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_freq_nonzero_rewards', arr_count_r)

#
#
# def make_video(env , sac , args , seed , state_size):
#     num_episodes = 10
#     l = 0
#     env.reset()
#     render = env.render()
#     # full_img = full_img.reshape(1, full_img.shape[0], full_img.shape[1], full_img.shape[2])
#     max_size = max(args['max_env_steps']//4 , 200)
#     full_img = np.zeros([num_episodes * max_size +1 , render.shape[0], render.shape[1], render.shape[2]], dtype=np.uint8)
#     # print(full_img.shape, full_img.dtype)
#     #
#     # assert False
#
#     for ep_i in range(num_episodes):
#         # print(ep_i)
#         start = True
#         hidden_p = None
#         action = 0
#         reward = 0
#         state = env.reset()
#         done = False
#         steps = 0
#         while not done:
#             img = env.render()
#             full_img[l,:,:,:] = img
#             if args['env_name'][:8] == 'MiniGrid':
#                 state = state['image']
#                 state = sac.get_encoded_obs(state)
#             else:
#                 state = convert_int_to_onehot(state, state_size)
#             l += 1
#             steps += 1
#             action, hidden_p = sac.select_action(state, action, reward, hidden_p, start, False, evaluate=True)
#             # action = env.action_space.sample()
#             next_state, reward, done, _ = env.step(action)
#             state = next_state
#             if start == True:
#                 start = False
#             if steps >= args['max_env_steps']//4 and steps >= 200:
#                 break
#     # print(l)
#     # print(full_img.shape)
#     # imgs = [Image.fromarray(img) for img in full_img]
#     path = os.path.join(args['logdir'], 'Seed_' + str(seed) + '_video.mp4')
#     # imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=200, loop=0)
#     clip = mpy.ImageSequenceClip(list(full_img[:l+1,:,:,:]), fps=5)
#     clip.write_videofile(path)
#
#
#
#
#
#
#
def log_test_and_save(env, agent, writer, args, avg_reward, avg_q_loss, updates, k_episode, i_episode, total_numsteps, avg_episode_steps , seed  , num_nonzero_rewards):
    # if total_numsteps % int(args['num_steps']/10) == int(args['num_steps']/10)  - 1:
    #     agent.save_model(args['logdir'],seed , total_numsteps)
    # else:
    #     agent.save_model(args['logdir'], seed, -1)
    avg_running_reward = avg_reward / k_episode
    avg_reward = 0.
    avg_discount_adj_reward = 0.
    episodes = 10
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_rewards = []
        ls_obs = []
        done = False
        step = 1
        while not done:
            state = state['image'].reshape(-1)
            ls_obs.append(state)
            if args['alg_type'] == "frame_stack":
                if step < args['frame_stacking_length']:
                    empty = np.zeros(state.shape[0] * (args['frame_stacking_length'] - step))
                    non_empty = np.array(ls_obs[:step]).reshape(-1)
                    # print(empty.shape,non_empty.shape)
                    state = np.concatenate((empty, non_empty))
                    # print(state.shape)
                else:
                    state = np.array(ls_obs[step - args['frame_stacking_length']:step]).reshape(-1)

            step += 1
            action = agent.select_action(state , evaluate = True)
            # action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            episode_reward += reward

            state = next_state
            if step >= args['max_env_steps']:
                break
        avg_reward += episode_reward
        rets = []
        R = 0
        for i, r in enumerate(episode_rewards[::-1]):
            R = r + args['gamma'] * R
            rets.insert(0, R)
        avg_discount_adj_reward += rets[0]
    avg_reward /= episodes
    avg_discount_adj_reward /= episodes


    # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    # if args['model_alg'] == 'AIS':
    if True:
        if updates == 0:
            avgql = 0
        else:
            avgql = avg_q_loss / (updates)
        print(
            "Seed: {}, Episode: {}, Total_num_steps: {},  episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_test_discount_adjusted_reward: {}, avg_q_loss: {}".format(
                seed,i_episode, total_numsteps, avg_episode_steps / k_episode, avg_running_reward, avg_reward,
                avg_discount_adj_reward, avgql))

    writer.add_scalar('Seed'+str(seed)+'Evaluation reward', avg_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Discount adjusted Evaluation reward', avg_discount_adj_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Training reward', avg_running_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Average Steps for each episode', avg_episode_steps, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Loss/Value', avgql, total_numsteps)
    print("----------------------------------------")
    writer.flush()

    return avg_reward , avg_discount_adj_reward

