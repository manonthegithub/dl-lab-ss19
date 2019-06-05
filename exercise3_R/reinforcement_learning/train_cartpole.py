import sys
sys.path.append("../") 

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id, _ = agent.act(state=state, deterministic=deterministic, p=[0.5, 0.5])
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, num_eval_episodes, eval_cycle, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard_t = Evaluation(os.path.join(tensorboard_dir, "train"), 'cart_pole_train',
                             ["episode_reward", "a_0", "a_1"]
                             )
    tensorboard_e = Evaluation(os.path.join(tensorboard_dir, "eval"), 'cart_pole_eval',
                             ["episode_reward", "a_0", "a_1"]
                             )

    # training
    for i in range(num_episodes):
        print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard_t.write_episode_data(i, eval_dict={  "episode_reward" : stats.episode_reward,
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})
        print(stats.episode_reward)

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...
        if i % eval_cycle == 0:

            res = {
                "episode_reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1)
            }

            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False)
                res['episode_reward'] += stats.episode_reward
                res['a_0'] += stats.get_action_usage(0)
                res['a_1'] += stats.get_action_usage(1)

            res['episode_reward'] /= num_eval_episodes
            res['a_0'] /= num_eval_episodes
            res['a_1'] /= num_eval_episodes

            tensorboard_e.write_episode_data(j, eval_dict=res)

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))
   
    tensorboard_t.close_session()
    tensorboard_e.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes
    num_episodes = 150

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    Q = MLP(state_dim, num_actions)
    Q_target = MLP(state_dim, num_actions)
    agent = DQNAgent(Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4)

    train_online(env, agent, num_episodes, num_eval_episodes, eval_cycle)

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
 
