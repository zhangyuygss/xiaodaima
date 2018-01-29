import gym
import torch
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
env = gym.make('CartPole-v0').unwrapped
# env.reset()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('transition',
                        ['state', 'action', 'reward', 'next_state'])


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

    def get_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


screen_width = 600
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)


model = DQN()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 128
GAMMA = 0.999

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


step_num = 0
# steps_done = 0

def choose_action(state):
    global step_num
    eps_thresh = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step_num / EPS_DECAY)
    step_num += 1
    if random.random() > eps_thresh:
        action = model.forward(Variable(state, volatile=True)).data.max(1)[1].view(1,1)
    else:
        action = LongTensor([[random.randrange(2)]])
    return action


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = memory.get_batch(BATCH_SIZE)
    batch = Transition(*zip(*batch))
    # Q(state), Q network
    batch_state = Variable(torch.cat(batch.state))
    batch_action = Variable(torch.cat(batch.action))
    Q_sa = model.forward(batch_state).gather(1, batch_action)
    # reward + Q(next_state), target Q network
    batch_reward = Variable(torch.cat(batch.reward))
    batch_next_state = Variable(torch.cat([x for x in batch.next_state if x is not None]),
                                volatile=True)
    next_state_mask = ByteTensor(tuple(map(lambda x: x is not None, batch.next_state)))
    Q_next_sa = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    Q_next_sa[next_state_mask] = model.forward(batch_next_state).max(1)[0]
    Q_next_sa.volatile = False
    target = batch_reward + GAMMA * Q_next_sa
    loss = F.smooth_l1_loss(Q_sa, target)
    # Backward computation
    optimizer.zero_grad()
    loss.backward()
    for para in model.parameters():
        para.grad.data.clamp_(-1,1)
    optimizer.step()


episode_durations = []


def train():
    epsiodes = 40
    for ep in range(epsiodes):
        print('Training epsiode {}'.format(ep))
        env.reset()
        cur_screen = pre_screen = get_screen()
        for t in count():
            state = cur_screen - pre_screen
            action = choose_action(state)
            _, reward, game_over, _ = env.step(action[0,0])
            reward = Tensor([reward])
            pre_screen = cur_screen
            cur_screen = get_screen()
            if not game_over:
                next_state = cur_screen - pre_screen
            else:
                next_state = None
            memory.push(state, action, reward, next_state)

            optimize_model()
            if game_over:
                print('Steps took in this epsiode: {}'.format(t))
                episode_durations.append(t + 1)
                plot_durations()
                break
    print('Training complete')


if __name__=='__main__':
    train()
