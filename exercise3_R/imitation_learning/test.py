from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, hl, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    state = rgb2gray(state)
    state = torch.tensor(state).unsqueeze(0).repeat(hl, 1, 1).unsqueeze(0)
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...
        
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        # if step < 10:
        #     a = [0.0, 1.0, 0.0]
        # else:
        #     a = agent.predict(state)
        #     a = id_to_action(a.argmax(dim=1).squeeze())
        a = agent.predict(state)
        a = a.argmax(dim=2).squeeze()[hl - 1]
        a = id_to_action(a)

        next_state, r, done, info = env.step(a)   
        episode_reward += r

        next_state = torch.tensor(rgb2gray(next_state)).unsqueeze(0).unsqueeze(0)
        state = torch.cat((state[:,1:,:,:], next_state), dim=1)
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      

    hl = 30
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = BCAgent(torch.device('cpu'), lr=0.0001, history_length=hl)
    agent.load("models/agent.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, hl, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
