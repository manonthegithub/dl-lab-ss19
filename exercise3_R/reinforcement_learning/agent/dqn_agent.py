import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tao: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(
            self.batch_size)

        batch_states = torch.tensor(batch_states).to(device).float()
        batch_next_states = torch.tensor(batch_next_states).to(device).float()
        batch_rewards = torch.tensor(batch_rewards).to(device).float()
        batch_dones = torch.tensor(np.invert(batch_dones).astype(int)).to(device).float()

        self.Q.train()
        self.Q_target.train()
        self.optimizer.zero_grad()

        vs = self.Q_target(batch_next_states)
        if len(vs.shape) == 2:
            a = vs.max(dim=1)[0]
            td_target = batch_rewards + self.gamma * a * batch_dones
            out = self.Q(batch_states)[range(batch_actions.shape[0]), batch_actions]
        else:
            a = vs[:, vs.shape[1] - 1, :].max(dim=1)[0]
            td_target = batch_rewards + self.gamma * a * batch_dones
            out = self.Q(batch_states)[range(batch_actions.shape[0]), batch_states.shape[1] - 1, batch_actions]

        loss = self.loss_function(out, td_target)
        loss.backward()
        self.optimizer.step()
        soft_update(self.Q_target, self.Q, self.tau)

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #              self.Q.update(...)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)

    def act(self, state, deterministic, p):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        self.Q.eval()
        state = torch.tensor(state).to(device).float().unsqueeze(0)
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            vs = self.Q(state)
            if (len(vs.shape)) == 2:
                action_id = vs.argmax(dim=1)[0].item()
            else:
                action_id = vs.argmax(dim=2)[0][vs.shape[1] - 1].item()

        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(range(self.num_actions), p=p)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)
        return file_name

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
