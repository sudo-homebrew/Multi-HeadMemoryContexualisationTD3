import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnext50_32x4d
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.manualaction import ManualAction, INPUTTIMEOUT
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING, BATCH_SIZE

from ..common.ounoise import OUNoise
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES

from .off_policy_agent import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# ResNext_HalfFullDecent_repeat_ddpg.py
# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        ResNext_output_size = 1024
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # --- define layers here ---
        self.fa1 = resnext50_32x4d(pretrained=True).to(device)
        self.fa1.fc = nn.Linear(in_features=self.fa1.fc.in_features, out_features=ResNext_output_size)
        self.fa2 = nn.Linear(ResNext_output_size, int(ResNext_output_size / 2 ** 2))
        self.fa3 = nn.Linear(int(ResNext_output_size / 2 ** 2), int(ResNext_output_size / 2 ** 4))
        self.fa4 = nn.Linear(int(ResNext_output_size / 2 ** 4), int(ResNext_output_size / 2 ** 8))
        self.fa5 = nn.Linear(int(ResNext_output_size / 2 ** 8), action_size)
        
        # --- define layers until here ---

#        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- input preprocessing here ---
        if states.dim() == 1:
            reshaped_states = states.view(1, 1, 61, 6).repeat(1, 3, 1, 1)
            x1 = torch.relu(self.fa1(reshaped_states))[0]
        if states.dim() == 2:
            reshaped_states = states.view(BATCH_SIZE, 1, 61, 6).repeat(1, 3, 1, 1)
            # --- define forward pass here ---
            x1 = torch.relu(self.fa1(reshaped_states))
        
        x2 = torch.relu(self.fa2(x1))
        x3 = torch.relu(self.fa3(x2))
        x4 = torch.relu(self.fa4(x3))
        action = torch.tanh(self.fa5(x4))

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)

        # --- define layers here ---
        self.l1 = nn.Linear(state_size, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, actions):
        # --- define forward pass here ---
        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        return x


class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.manual_action = ManualAction()
        
        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0), np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)]
        
    def get_action_manual(self):
        m_action = self.manual_action.get_action()
        return m_action

    def train(self, state, action, reward, state_next, done):
        # optimize critic
        action_next = self.actor_target(state_next)
        Q_next = self.critic_target(state_next, action_next)
        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q = self.critic(state, action)

        loss_critic = self.loss_function(Q, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        pred_a_sample = self.actor(state)
        loss_actor = -1 * (self.critic(state, pred_a_sample)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        return [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]
