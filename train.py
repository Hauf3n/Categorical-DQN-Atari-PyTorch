import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from Agents import C51_Agent
from Atari_Wrapper import Atari_Wrapper
from Env_Runner import Env_Runner
from Experience_Replay import Experience_Replay

device = torch.device("cuda:0")
dtype = torch.float

def make_transitions(obs, actions, rewards, dones):
    # observations are in uint8 format
    
    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t+1],
                       int(not dones[t])))
        
    return tuples

def train(hyperparameters):
    
    # ARGS
    env_name = hyperparameters.env
    env_with_lives = hyperparameters.lives
    
    v_min = hyperparameters.v_min
    v_max = hyperparameters.v_max
    num_atoms = hyperparameters.atoms
    
    num_stacked_frames = hyperparameters.stacked_frames
    
    replay_memory_size = hyperparameters.replay_memory_size
    min_replay_size_to_update = hyperparameters.replay_size_to_update
    
    lr = hyperparameters.lr
    gamma = hyperparameters.gamma
    
    minibatch_size = hyperparameters.minibatch_size
    steps_rollout = hyperparameters.steps_rollout
    
    start_eps = hyperparameters.start_eps
    final_eps = hyperparameters.final_eps
    final_eps_frame = hyperparameters.final_eps_frame
    total_steps = hyperparameters.total_steps
    
    target_net_update = hyperparameters.target_net_update
    save_model_steps = hyperparameters.save_model_steps
    report = hyperparameters.report  
    
    # INIT
    delta_z = (v_max - v_min)/ (num_atoms - 1) 
    value_support = torch.tensor([ v_min +  (i * delta_z) for i in range(num_atoms)]).to(device)
    
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=env_with_lives)

    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    eps_interval = start_eps-final_eps

    agent = C51_Agent(in_channels, num_actions, num_atoms, value_support, start_eps).to(device)
    target_agent = C51_Agent(in_channels, num_actions, num_atoms, value_support, start_eps).to(device)
    target_agent.load_state_dict(agent.state_dict())

    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent)
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    num_steps = 0
    num_model_updates = 0

    corrected_index = torch.tensor([num_atoms*i for i in range(minibatch_size)]).to(device)
    corrected_index = corrected_index.repeat_interleave(num_atoms).reshape(minibatch_size, num_atoms)

    # TRAIN
    start_time = time.time()
    while num_steps < total_steps:
        
        # set agent exploration | cap exploration after x timesteps to final epsilon
        new_epsilon = np.maximum(final_eps, start_eps - ( eps_interval * num_steps/final_eps_frame))
        agent.set_epsilon(new_epsilon)
        
        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)
        
        # add
        num_steps += steps_rollout
        
        # check if update
        if num_steps < min_replay_size_to_update:
            continue
        
        # update
        for update in range(4):
            optimizer.zero_grad()
            
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(device).to(dtype)) / 255 
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(device)
            
            # uint8 to float32 and normalize to 0-1
            next_obs = (torch.stack([i[3] for i in minibatch]).to(device).to(dtype)) / 255
            
            dones = torch.tensor([i[4] for i in minibatch]).to(device)
            
            #  *** C51 ***
            
            # get atom probabilities for obs and next obs
            
            obs_Ps = agent(obs)[range(minibatch_size), actions] # get atoms from used action
            next_obs_Ps = target_agent(next_obs).detach() # will be used as label later
            
            # get a* from target network
            best_a = target_agent.greedy(next_obs_Ps)
            # get next_obs atoms from a*
            next_obs_Ps = next_obs_Ps[range(minibatch_size), best_a]
            
            # ^T_z 
            tmp=torch.ones(minibatch_size, num_atoms).to(device).to(dtype) * (gamma * value_support)
            
            rewards = rewards.repeat_interleave(num_atoms).reshape(minibatch_size,num_atoms)
            dones = dones.repeat_interleave(num_atoms).reshape(minibatch_size,num_atoms)
            
            T_z = rewards + tmp * dones
            # clip to value interval
            T_z = torch.clamp(T_z, min=v_min, max=v_max)
            
            # b 
            b = (T_z - torch.tensor(v_min).to(device)) / torch.tensor(delta_z).to(device)
            
            # l , u
            l = torch.floor(b)
            u = torch.ceil(b)
            
            # distribute probability
            m_l = next_obs_Ps * (u-b)
            m_u = next_obs_Ps * (b-l)
            
            # much faster than using a loop
            m = torch.zeros(minibatch_size*num_atoms).to(device).to(dtype)
            l = l + corrected_index
            u = u + corrected_index
            m = m.index_add(0,l.reshape(-1).long(),m_l.reshape(-1))
            m = m.index_add(0,u.reshape(-1).long(),m_u.reshape(-1))
            m = m.reshape(minibatch_size,num_atoms)
            
            # cross entropy loss
            loss =  torch.mean( - torch.sum( m * torch.log(obs_Ps) ,dim=1))
            
            loss.backward()
            optimizer.step()
            
        num_model_updates += 1
         
        # update target network
        if num_model_updates%target_net_update == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # print time
        if num_steps%report < steps_rollout:
            end_time = time.time()
            print(f'*** total steps: {num_steps} | time: {end_time - start_time} ***')
            start_time = time.time()
        
        # save the dqn after some time
        if num_steps%save_model_steps < steps_rollout:
            torch.save(agent,f"{env_name}-{num_steps}.pt")

        env.close()
    

if __name__ == "__main__":
    
    hyperparameters = argparse.ArgumentParser()
    
    # set hyperparameter
    
    hyperparameters.add_argument('-lr', type=float, default=2.5e-4)
    hyperparameters.add_argument('-v_min', type=float, default=-10)
    hyperparameters.add_argument('-v_max', type=float, default=10)
    hyperparameters.add_argument('-atoms', type=int, default=51)
    hyperparameters.add_argument('-env', default='PongNoFrameskip-v4')
    hyperparameters.add_argument('-lives', type=bool, default=False)
    hyperparameters.add_argument('-stacked_frames', type=int, default=4)
    hyperparameters.add_argument('-replay_memory_size', type=int, default=250000)
    hyperparameters.add_argument('-replay_size_to_update', type=int, default=20000)
    hyperparameters.add_argument('-gamma', type=float, default=0.99)
    hyperparameters.add_argument('-minibatch_size', type=int, default=32)
    hyperparameters.add_argument('-steps_rollout', type=int, default=16)
    hyperparameters.add_argument('-start_eps', type=float, default=1)
    hyperparameters.add_argument('-final_eps', type=float, default=0.05)
    hyperparameters.add_argument('-final_eps_frame', type=int, default=1000000)
    hyperparameters.add_argument('-total_steps', type=int, default=25000000)
    hyperparameters.add_argument('-target_net_update', type=int, default=625)
    hyperparameters.add_argument('-save_model_steps', type=int, default=500000)
    hyperparameters.add_argument('-report', type=int, default=50000)
    
    train(hyperparameters.parse_args())