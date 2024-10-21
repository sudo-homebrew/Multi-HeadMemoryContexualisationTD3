import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..common.manualaction import ManualAction
from ..common.settings import POLICY_NOISE, POLICY_NOISE_CLIP, POLICY_UPDATE_FREQUENCY, BATCH_SIZE
from ..common.ounoise import OUNoise

from .off_policy_fbe_agent import OffPolicyAgent, Network

from .fbe_unet_model import UNet

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# https://github.com/djbyrne/TD3
# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.
# TD3 with Learnerble Convolutional neural network


class Actor(Network):
    def __init__(self, name, map_size, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        self.fa1 = nn.Linear(state_size - 4, hidden_size)
        self.fa2 = nn.Linear(hidden_size, int(hidden_size * 2))
        self.fa3 = nn.Linear(int(hidden_size * 2), int(hidden_size * 2))
        self.fa4 = nn.Linear(int(hidden_size * 2), int(hidden_size / 2))

        self.fa5 = nn.Linear(hidden_size + hidden_size // 2 + hidden_size // 2 ** 3, int(hidden_size / 2 ** 1))
        self.fa6 = nn.Linear(int(hidden_size / 2 ** 1), int(hidden_size / 2 ** 2))
        self.fa7 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))
        self.fa8 = nn.Linear(int(hidden_size / 2 ** 3), action_size)

        # --- conv layers for feature extraction of LiDAR---
        self.conv_iter = 3
        self.pooling_kernel_size = 2
        inner_channel_size = 2 ** 6
        fc_size = int(state_size / (self.pooling_kernel_size ** self.conv_iter)) * inner_channel_size

        self.filterT1 = nn.Conv1d(1, inner_channel_size, 4, padding='same', padding_mode='circular')
        self.filterT2 = nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular')
        self.filterT3 = nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular')

        self.conv_batch_norm =nn.BatchNorm1d(inner_channel_size)
        self.maxpool = nn.MaxPool1d(self.pooling_kernel_size)

        self.conv_fc = nn.Linear(fc_size, int(hidden_size / 2 ** 1))

        # Env NN
        self.opfa1 = nn.Linear(4, int(hidden_size / 2 ** 2))
        self.opfa2 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))
        # self.opfa3 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))

        # --- conv for map ---
        self.unet = UNet(3, 1)
        # self.unet = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.mfa1 = nn.Linear(map_size // 3, hidden_size)
        # self.mfa2 = nn.Linear(hidden_size, hidden_size)
        self.mfa3 = nn.Linear(hidden_size, hidden_size // 2)
        # self.mfa4 = nn.Linear(int(hidden_size * 2), int(hidden_size * 2))
        # self.mfa5 = nn.Linear(int(hidden_size * 2), hidden_size // 2)

        self.dropout = nn.Dropout(0.5)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm1_1 = nn.LayerNorm(int(hidden_size / 2) + int(hidden_size / 2 ** 3))
        self.layer_norm2 = nn.LayerNorm(int(hidden_size / 2 ** 1))
        self.layer_norm3 = nn.LayerNorm(int(hidden_size / 2 ** 2))
        self.layer_norm4 = nn.LayerNorm(int(hidden_size / 2 ** 3))

        # Custom Activation fucntion
        self.silu = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, map, states, visualize=False):
        # --- define forward pass here ---
        ### TODO ###
        # Reshape map and use as state #
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -4:]
            map_feature = torch.reshape(map, (BATCH_SIZE, 3, 128, 128))
            cat_dim = 1
        else:
            lidar_states = states[:360]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-4:]
            map_feature = torch.reshape(map, (1, 3, 128, 128))
            cat_dim = 0

        x = self.silu(self.fa1(lidar_states))
        x = self.silu(self.fa2(x))
        x = self.silu(self.fa3(x))
        x = self.layer_norm2(self.fa4(x))
        x = torch.sigmoid(x)

        feature = self.maxpool(self.silu(self.conv_batch_norm(self.filterT1(lidar_features))))
        feature = self.maxpool(self.silu(self.conv_batch_norm(self.filterT2(feature))))
        feature = self.maxpool(self.silu(self.conv_batch_norm(self.filterT3(feature))))
        feature = torch.flatten(feature, start_dim=cat_dim)
        feature = torch.sigmoid(self.conv_fc(feature))

        opx = self.silu(self.opfa1(env_states))
        opx = self.silu(self.opfa2(opx))

        map_feature = self.unet(map_feature)
        if cat_dim == 1:
            map_feature = map_feature.view(BATCH_SIZE, -1)
        else:
            map_feature = torch.flatten(map_feature)
        map_feature = self.silu(self.mfa1(map_feature))
        # map_feature = self.silu(self.mfa2(map_feature))
        map_feature = self.layer_norm2(self.mfa3(map_feature))
        # map_feature = self.layer_norm2(self.mfa4(map_feature))
        map_feature = torch.sigmoid(map_feature)


        x = torch.cat((x, feature, map_feature, opx), cat_dim)

        x = self.layer_norm2(self.fa5(x))
        x = self.silu(x)
        x = self.layer_norm3(self.fa6(x))
        x = self.silu(x)
        x = self.layer_norm4(self.fa7(x))
        x = self.silu(x)

        action = torch.tanh(self.fa8(x))

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x, x], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action


class Critic(Network):
    def __init__(self, name, map_size, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        # Feature Extraction Conv
        self.conv_iter = 3
        self.pooling_kernel_size = 2
        inner_channel_size = 2 ** 6
        fc_size = int(state_size / (self.pooling_kernel_size ** self.conv_iter)) * inner_channel_size

        self.q1filterT1 = nn.Conv1d(1, inner_channel_size, 4, padding='same', padding_mode='circular')
        self.q1filterT2 = nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular')
        self.q1filterT3 = nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular')

        self.q2filterT1 = nn.Conv1d(1, inner_channel_size, 4, padding='same', padding_mode='circular')
        self.q2filterT2 = nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular')
        self.q2filterT3 = nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular')

        self.conv_batch_norm =nn.BatchNorm1d(inner_channel_size)
        self.maxpool = nn.MaxPool1d(self.pooling_kernel_size)

        self.conv_fc1 = nn.Linear(fc_size, int(hidden_size / 6))
        self.conv_fc2 = nn.Linear(fc_size, int(hidden_size / 6))

        # Custom Activation fucntion
        self.silu = nn.SiLU()

        # Q1
        # --- define layers here ---
        self.q1l1 = nn.Linear(state_size - 4, int(hidden_size / 6))

        self.q1l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.q1l3 = nn.Linear(int(hidden_size / 6) * 3 + int(hidden_size / 2), hidden_size)
        self.q1l4 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.q1l5 = nn.Linear(int(hidden_size / 2), int(hidden_size / 2 ** 2))
        self.q1l6 = nn.Linear(int(hidden_size / 2 ** 2), 1)

        self.q1op1 = nn.Linear(4, int(hidden_size / 2 ** 2))
        self.q1op2 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 6))

        # Q2
        # --- define layers here ---
        self.q2l1 = nn.Linear(state_size - 4, int(hidden_size / 6))

        self.q2l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.q2l3 = nn.Linear(int(hidden_size / 6) * 3 + int(hidden_size / 2), hidden_size)
        self.q2l4 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.q2l5 = nn.Linear(int(hidden_size / 2), int(hidden_size / 2 ** 2))
        self.q2l6 = nn.Linear(int(hidden_size / 2 ** 2), 1)

        self.q2op1 = nn.Linear(4, int(hidden_size / 2 ** 2))
        self.q2op2 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 6))

        self.apply(super().init_weights)

    def forward(self, map, states, actions):
        ### TODO ###
        # Reshape map and use as state #
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -4:]
            cat_dim = 1
        else:
            lidar_states = states[:360]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-4:]
            cat_dim = 0

        # Q1
        xs = torch.sigmoid(self.q1l1(lidar_states))

        xf = self.maxpool(self.silu(self.conv_batch_norm(self.q1filterT1(lidar_features))))
        xf = self.maxpool(self.silu(self.conv_batch_norm(self.q1filterT2(xf))))
        xf = self.maxpool(self.silu(self.q1filterT3(xf)))
        xf = torch.flatten(xf, start_dim=cat_dim)
        xf = torch.sigmoid(self.conv_fc1(xf))

        xop = self.silu(self.q1op1(env_states))
        xop = self.silu(self.q1op2(xop))

        xa = self.silu(self.q1l2(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x = self.silu(self.q1l3(x))
        x = self.silu(self.q1l4(x))
        x1 = self.silu(self.q1l5(x))


        # Q2
        xs = torch.sigmoid(self.q2l1(lidar_states))

        xf = self.maxpool(self.silu(self.conv_batch_norm(self.q2filterT1(lidar_features))))
        xf = self.maxpool(self.silu(self.conv_batch_norm(self.q2filterT2(xf))))
        xf = self.maxpool(self.silu(self.q2filterT3(xf)))
        xf = torch.flatten(xf, start_dim=cat_dim)
        xf = torch.sigmoid(self.conv_fc2(xf))

        xop = self.silu(self.q2op1(env_states))
        xop = self.silu(self.q2op2(xop))

        xa = self.silu(self.q2l2(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x = self.silu(self.q2l3(x))
        x = self.silu(self.q2l4(x))
        x2 = self.silu(self.q2l5(x))

        return x1, x2

    def Q1_forward(self, map, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -4:]
            cat_dim = 1
        else:
            lidar_states = states[:360]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-4:]
            cat_dim = 0

        xs = torch.sigmoid(self.q1l1(lidar_states))

        xf = self.maxpool(self.silu(self.conv_batch_norm(self.q1filterT1(lidar_features))))
        xf = self.maxpool(self.silu(self.conv_batch_norm(self.q1filterT2(xf))))
        xf = self.maxpool(self.silu(self.q1filterT3(xf)))
        xf = torch.flatten(xf, start_dim=cat_dim)
        xf = torch.sigmoid(self.conv_fc1(xf))

        xop = self.silu(self.q1op1(env_states))
        xop = self.silu(self.q1op2(xop))

        xa = self.silu(self.q1l2(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x = self.silu(self.q1l3(x))
        x = self.silu(self.q1l4(x))
        x1 = self.silu(self.q1l5(x))

        return x1


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

    def get_action(self, map, state, is_training, step, visualize=False):
        map = torch.from_numpy(np.asarray(map)).to(dtype=torch.float32, device=self.device)
        state = torch.from_numpy(np.asarray(state)).to(dtype=torch.float32, device=self.device)
        action = self.actor(map, state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(dtype=torch.float32, device=self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(0, 1.0), -1.0, 1.0), np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)]

    def get_action_manual(self):
        return self.manual_action.get_action()

    def train(self, map, state, action, reward, state_next, map_next, done):
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        action_next = (self.actor_target(map_next, state_next) + noise).clamp(-1.0, 1.0)
        Q1_next, Q2_next = self.critic_target(map_next, state_next, action_next)
        Q_next = torch.min(Q1_next, Q2_next)

        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q1, Q2 = self.critic(map, state, action)

        loss_critic = self.loss_function(Q1, Q_target) + self.loss_function(Q2, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        if self.iteration % self.policy_freq == 0:
            # optimize actor
            loss_actor = -1 * self.critic.Q1_forward(map, state, self.actor(map, state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss]