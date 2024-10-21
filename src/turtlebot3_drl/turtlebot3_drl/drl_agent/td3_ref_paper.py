import numpy as np
import copy
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..common.manualaction import ManualAction
from ..common.settings import POLICY_NOISE, POLICY_NOISE_CLIP, POLICY_UPDATE_FREQUENCY
from ..common.ounoise import OUNoise

from .off_policy_agent import OffPolicyAgent, Network

from .unet_lite2 import UNet

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# https://github.com/djbyrne/TD3
# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.
# Small obstacles detection paper code


class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        K = 20
        self.fa1 = nn.Linear(state_size - 6, 256)
        self.fa2 = nn.Linear(state_size - 6, 256)
        self.fa3 = nn.Linear(256, K)
        self.fa4 = nn.Linear(256, K)

        self.fa5 = nn.Linear(2 * (K // 3) + 4, K)
        self.fa6 = nn.Linear(K, 100)
        self.fa7 = nn.Linear(100, action_size)

        # Env NN
        self.opfa1 = nn.Linear(4, 256)

        self.maxpool = nn.MaxPool1d(3)
        self.lrelu = torch.nn.LeakyReLU()

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            tmp_env_states = states[:, -6:]
            cat_dim = 1
            env_states = torch.cat([tmp_env_states[:, :2], tmp_env_states[:, 4:]], dim=cat_dim)

            for index, s in enumerate(lidar_states[:]):
                if index == 0:
                    state_x = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    state_y = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                else:
                    x_tmp = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    y_tmp = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    state_x = torch.cat([state_x, x_tmp], dim=0)
                    state_y = torch.cat([state_y, y_tmp], dim=0)

        else:
            lidar_states = states[:360]
            tmp_env_states = states[-6:]
            cat_dim = 0
            env_states = torch.cat([tmp_env_states[:2], tmp_env_states[4:]], dim=cat_dim)

            for index, s in enumerate(lidar_states[:]):
                if index == 0:
                    state_x = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    state_y = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                else:
                    x_tmp = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    y_tmp = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    state_x = torch.cat([state_x, x_tmp], dim=0)
                    state_y = torch.cat([state_y, y_tmp], dim=0)

        x = self.lrelu(self.fa1(state_x)) * torch.sigmoid(self.opfa1(env_states))
        y = self.lrelu(self.fa2(state_y)) * torch.sigmoid(self.opfa1(env_states))
        x = self.lrelu(self.fa3(x))
        y = self.lrelu(self.fa4(y))

        xy = torch.cat([self.maxpool(x.unsqueeze(0))[0,:], self.maxpool(y.unsqueeze(0))[0,:], env_states], cat_dim)

        xy = self.lrelu(self.fa5(xy))
        xy = self.lrelu(self.fa6(xy))
        action = torch.tanh(self.fa7(xy))

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x, x], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action


class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        K = 20
        self.fa1 = nn.Linear(state_size - 6, 256)
        self.fa2 = nn.Linear(state_size - 6, 256)
        self.fa3 = nn.Linear(256, K)
        self.fa4 = nn.Linear(256, K)

        # Env NN
        self.opfa1 = nn.Linear(4, 256)

        self.maxpool = nn.MaxPool1d(3)
        self.lrelu = torch.nn.LeakyReLU()
        # Q1
        # --- define layers here ---
        self.q1fa5 = nn.Linear(2 * (K // 3) + 4 + action_size, 100)
        self.q1fa6 = nn.Linear(100, 100)
        self.q1fa7 = nn.Linear(100, 1)


        # Q2
        # --- define layers here ---
        self.q2fa5 = nn.Linear(2 *( K // 3) + 4, 100)
        self.q2fa6 = nn.Linear(100, 100)
        self.q2fa7 = nn.Linear(100, 1)

        self.apply(super().init_weights)

    def forward(self, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            tmp_env_states = states[:, -6:]
            cat_dim = 1
            env_states = torch.cat([tmp_env_states[:, :2], tmp_env_states[:, 4:]], dim=cat_dim)

            for index, s in enumerate(lidar_states[:]):
                if index == 0:
                    state_x = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    state_y = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                else:
                    x_tmp = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    y_tmp = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    state_x = torch.cat([state_x, x_tmp], dim=0)
                    state_y = torch.cat([state_y, y_tmp], dim=0)

        else:
            lidar_states = states[:360]
            tmp_env_states = states[-6:]
            cat_dim = 0
            env_states = torch.cat([tmp_env_states[:2], tmp_env_states[4:]], dim=cat_dim)

            for index, s in enumerate(lidar_states[:]):
                if index == 0:
                    state_x = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    state_y = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                else:
                    x_tmp = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    y_tmp = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    state_x = torch.cat([state_x, x_tmp], dim=0)
                    state_y = torch.cat([state_y, y_tmp], dim=0)

        x = self.lrelu(self.fa1(state_x)) * torch.sigmoid(self.opfa1(env_states))
        y = self.lrelu(self.fa2(state_y)) * torch.sigmoid(self.opfa1(env_states))
        x = self.lrelu(self.fa3(x))
        y = self.lrelu(self.fa4(y))

        xy = torch.cat([self.maxpool(x.unsqueeze(0))[0,:], self.maxpool(y.unsqueeze(0))[0,:], env_states, actions], cat_dim)

        # Q1
        q1 = self.lrelu(self.q1fa5(xy))
        q1 = self.lrelu(self.q1fa6(q1))
        q1 = torch.tanh(self.q1fa7(q1))


        # Q2
        q2 = self.lrelu(self.q1fa5(xy))
        q2 = self.lrelu(self.q1fa6(q2))
        q2 = torch.tanh(self.q1fa7(q2))

        return q1, q2

    def Q1_forward(self, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            tmp_env_states = states[:, -6:]
            cat_dim = 1
            env_states = torch.cat([tmp_env_states[:, :2], tmp_env_states[:, 4:]], dim=cat_dim)

            for index, s in enumerate(lidar_states[:]):
                if index == 0:
                    state_x = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    state_y = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                else:
                    x_tmp = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    y_tmp = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states[0])) * index)) * s).unsqueeze(0)
                    state_x = torch.cat([state_x, x_tmp], dim=0)
                    state_y = torch.cat([state_y, y_tmp], dim=0)

        else:
            lidar_states = states[:360]
            tmp_env_states = states[-6:]
            cat_dim = 0
            env_states = torch.cat([tmp_env_states[:2], tmp_env_states[4:]], dim=cat_dim)

            for index, s in enumerate(lidar_states[:]):
                if index == 0:
                    state_x = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    state_y = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                else:
                    x_tmp = (torch.sin(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    y_tmp = (torch.cos(torch.tensor((2 * torch.pi/len(lidar_states)) * index)) * s).unsqueeze(0)
                    state_x = torch.cat([state_x, x_tmp], dim=0)
                    state_y = torch.cat([state_y, y_tmp], dim=0)

        x = self.lrelu(self.fa1(state_x)) * torch.sigmoid(self.opfa1(env_states))
        y = self.lrelu(self.fa2(state_y)) * torch.sigmoid(self.opfa1(env_states))
        x = self.lrelu(self.fa3(x))
        y = self.lrelu(self.fa4(y))

        xy = torch.cat([self.maxpool(x.unsqueeze(0))[0,:], self.maxpool(y.unsqueeze(0))[0,:], env_states, actions], cat_dim)

        q1 = self.lrelu(self.q1fa5(xy))
        q1 = self.lrelu(self.q1fa6(q1))
        q1 = torch.tanh(self.q1fa7(q1))

        return q1


class TD3(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.manual_action = ManualAction()

        # DRL parameters
        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        # TD3 parameters
        self.policy_noise   = POLICY_NOISE
        self.noise_clip     = POLICY_NOISE_CLIP
        self.policy_freq    = POLICY_UPDATE_FREQUENCY

        self.last_actor_loss = 0

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(torch.float32).to(self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(torch.float32).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(0, 1.0), -1.0, 1.0), np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)]
    
    def get_action_manual(self):
        return self.manual_action.get_action()

    def train(self, state, action, reward, state_next, done):
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        action_next = (self.actor_target(state_next) + noise).clamp(-1.0, 1.0)
        Q1_next, Q2_next = self.critic_target(state_next, action_next)
        Q_next = torch.min(Q1_next, Q2_next)

        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q1, Q2 = self.critic(state, action)

        loss_critic = self.loss_function(Q1, Q_target) + self.loss_function(Q2, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        if self.iteration % self.policy_freq == 0:
            # optimize actor
            loss_actor = -1 * self.critic.Q1_forward(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss]