# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from utils import *
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from tensorboard_evaluation import *
from utils import EpisodeStats

def run_episode(env, agent, deterministic, skip_frames=2,  do_training=True, rendering=True, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist)
    actions = {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0
    }
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state=state, deterministic=deterministic, p=[0.3, 0.175, 0.175, 0.25, 0.1])
        action = id_to_action(action_id, 0.7, True)
        actions[action_id] += 1

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1
    print('Actions stats ' + str(actions))

    return stats


def train_online(env, agent, eval_cycle, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    max_ts = 1000
    ts = 100
 
    print("... train agent")
    tensorboard_t = Evaluation(os.path.join(tensorboard_dir, "train"),'train', ["episode_reward", "straight", "left", "right", "accel", "brake"])
    tensorboard_e = Evaluation(os.path.join(tensorboard_dir, "eval"), 'eval',["episode_reward", "straight", "left", "right", "accel", "brake"])

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
       
        stats = run_episode(env, agent, history_length=history_length, max_timesteps=ts, deterministic=False, do_training=True)

        tensorboard_t.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })
        print('Reward ' + str(stats.episode_reward))

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        if i % eval_cycle == 0:
            cum_stats = {
                "episode_reward": 0.0,
                "straight": 0.0,
                "left": 0.0,
                "right": 0.0,
                "accel": 0.0,
                "brake": 0.0
            }
            runs = 5
            for j in range(runs):
                stats = run_episode(env, agent, history_length=history_length, max_timesteps=ts, deterministic=True, do_training=False)
                cum_stats['episode_reward'] += stats.episode_reward
                cum_stats['straight'] += stats.get_action_usage(STRAIGHT)
                cum_stats['left'] += stats.get_action_usage(LEFT)
                cum_stats['right'] += stats.get_action_usage(RIGHT)
                cum_stats['accel'] += stats.get_action_usage(ACCELERATE)
                cum_stats['brake'] += stats.get_action_usage(BRAKE)
            cum_stats['episode_reward'] /= runs
            cum_stats['straight'] /= runs
            cum_stats['left'] /= runs
            cum_stats['right'] /= runs
            cum_stats['accel'] /= runs
            cum_stats['brake'] /=runs
            print(cum_stats)
            tensorboard_e.write_episode_data(i, eval_dict=cum_stats)

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            print(agent.save(os.path.join(model_dir, "dqn_agent.ckpt")))

        if ts < max_ts:
            ts += 3

    tensorboard_t.close_session()
    tensorboard_e.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 1000
    eval_cycle = 20

    env = gym.make('CarRacing-v0').unwrapped

    hl = 9
    num_actions = 5

    model_dir = "./models_carracing"
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    Q = CNN(history_length=hl + 1, n_classes=num_actions)
    Q_target = CNN(history_length=hl + 1, n_classes=num_actions)
    agent = DQNAgent(Q, Q_target, num_actions, gamma=0.6, batch_size=32, epsilon=0.1, tau=0.01, lr=1e-4)
    fn = os.path.join(model_dir, 'dqn_agent.ckpt')
    if os.path.exists(fn):
        agent.load(fn)

    train_online(env, agent, eval_cycle, num_episodes=num_eval_episodes, history_length=hl, model_dir=model_dir)

