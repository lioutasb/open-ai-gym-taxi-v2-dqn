import gym
import math
import time
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('Taxi-v2').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

LEARNING_RATE = 0.01
NUM_EPISODES = 5000
MEMORY = 50000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_END = 0.01
MAX_EPISODE = 50
TARGET_UPDATE = 10
HIDDEN_DIM = 64
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.n

SOLVE_TAXI_MESSAGE = """Task : \n
1) The cab(YELLOW) should find the shortest path to BLUE(passenger) 
2) Perform a "pickup" action to board the passenger which turns the cab(GREEN)
3) Take the passenger to the PINK(drop location) using the shortest path
4) Perform a "dropoff" action
"""


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQN, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)[:,0,:]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x

policy_net = DQN(N_STATES,N_ACTIONS,HIDDEN_DIM).to(device)
target_net = DQN(N_STATES,N_ACTIONS,HIDDEN_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

memory = ReplayMemory(MEMORY)


def to_variable(x, type=torch.long):
    return torch.autograd.Variable(torch.tensor(x, dtype=type).to(device))

def get_Q(net, state):
    state = to_variable(np.array(state).reshape(-1, 1))
    net.train(mode=False)
    return net(state)


def get_action(state, eps):
    if np.random.rand() < eps:
        return np.random.choice(N_ACTIONS)
    else:
        policy_net.train(mode=False)
        scores = get_Q(policy_net, state)
        print(scores)
        _, argmax = torch.max(scores.data, 1)
        return int(argmax.cpu().numpy())



episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    states = np.vstack([x.state for x in transitions])
    actions = np.array([x.action for x in transitions])
    rewards = np.array([x.reward for x in transitions])
    next_states = np.vstack([x.next_state for x in transitions])
    done = np.array([x.done for x in transitions])

    # Q_predict = get_Q(policy_net, states)
    # Q_target = Q_predict.clone().data.cpu().numpy()
    # Q_target[np.arange(len(Q_target)), actions] = rewards + GAMMA * np.max(get_Q(target_net, next_states).data.cpu().numpy(), axis=1) * ~done
    # Q_target = to_variable(Q_target)

    Q_predict = get_Q(policy_net, states)

    Q_next_state = np.argmax(get_Q(policy_net, next_states).data.cpu().numpy(), axis=1).reshape(-1)
    Q_target = Q_predict.clone().data.cpu().numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + GAMMA * np.choose(Q_next_state, get_Q(target_net, next_states).data.cpu().numpy().T) * ~done
    Q_target = to_variable(Q_target, type=torch.float)


    policy_net.train(mode=True)
    optim.zero_grad()
    loss = loss_fn(Q_predict, Q_target)
    loss.backward()
    optim.step()

def epsilon_annealing(epsiode, max_episode, min_eps):
    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)

def clear_screen(delay=1):
    time.sleep(delay)
    os.system('clear')

def log_progress(env, reward=0, total_reward=0, delay=None, message=None):
    if type(message) is str:
        print(message)
    env.render()
    print('Reward:', reward)
    print('Cumulative reward', total_reward)
    clear_screen(delay)

def init_message(attempt, perf):
    return 'Initial State : {}'.format(perf_message(attempt=attempt, perf=perf))

def perf_message(attempt, perf):
    return '{}\nAttempt: {} | Average reward (until last episode): {:.2f}'.format(
        SOLVE_TAXI_MESSAGE, 
        attempt + 1, 
        perf
    )


perf = 0
score  = 0
for i_episode in range(NUM_EPISODES):
    clear_screen(0)
    state = env.reset()
    log_progress(env, delay=0.5, message=init_message(i_episode, perf))
    total_reward = 0
    eps = epsilon_annealing(i_episode, MAX_EPISODE, EPS_END)
    done = False
    t = 0
    while not done:
        action = get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        log_progress(env, reward=reward, total_reward=total_reward, delay=0.5,message=perf_message(i_episode, perf))

        if done:
            reward = -1

        memory.push(state, action, next_state, reward, done)

        state = next_state

        optimize_model()
        t += 1
        if done:
            episode_durations.append(t)
            plot_durations()

    score += total_reward
    perf = score/(i_episode + 1)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
