import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..common.manualaction import ManualAction
from ..common.settings import POLICY_NOISE, POLICY_NOISE_CLIP, POLICY_UPDATE_FREQUENCY
from ..common.ounoise import OUNoise

from .off_policy_agent import OffPolicyAgent, Network

from .unet_model import UNet

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# https://github.com/djbyrne/TD3
# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.
# TD3 with UNET


class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        self.device = torch.device('mps')
        # --- define layers here ---
        # self.fa1 = nn.Linear(state_size - 6, hidden_size)
        # self.fa2 = nn.Linear(hidden_size, int(hidden_size * 2))
        # self.fa3 = nn.Linear(int(hidden_size * 2), int(hidden_size * 2))
        # self.fa4 = nn.Linear(int(hidden_size * 2), int(hidden_size / 2) + int(hidden_size / 2 ** 3))

        self.fa5 = nn.Linear(hidden_size, int(hidden_size / 2 ** 1))
        self.fa6 = nn.Linear(int(hidden_size / 2 ** 1), int(hidden_size / 2 ** 2))
        self.fa7 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))
        self.fa8 = nn.Linear(int(hidden_size / 2 ** 3), action_size)

        # --- conv layers for feature extraction ---
        self.unet = UNet(1, 1)
        self.conv_fc = nn.Linear(state_size - 6, int(hidden_size / 2 ** 2))

        # Env NN
        self.opfa1 = nn.Linear(6, int(hidden_size / 2 ** 2))
        self.opfa2 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))
        # self.opfa3 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))

        self.dropout = nn.Dropout(0.5)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm1_1 = nn.LayerNorm(int(hidden_size / 2) + int(hidden_size / 2 ** 3))
        self.layer_norm2 = nn.LayerNorm(int(hidden_size / 2 ** 1))
        self.layer_norm3 = nn.LayerNorm(int(hidden_size / 2 ** 2))
        self.layer_norm4 = nn.LayerNorm(int(hidden_size / 2 ** 3))

        # Custom Activation fucntion
        self.silu = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -6:]
            cat_dim = 1
        else:
            lidar_states = states[:360]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-6:]
            cat_dim = 0

        x = self.silu(self.fa1(lidar_states))
        x = self.silu(self.fa2(x))
        x = self.silu(self.fa3(x))
        x = self.layer_norm1_1(self.fa4(x))
        x = torch.sigmoid(x)

        feature = torch.sigmoid(self.conv_fc(self.unet(lidar_features)))
        if cat_dim == 1:
            feature = feature[:, 0, :]
        else:
            feature = feature[0, 0, :]

        opx = self.silu(self.opfa1(env_states))
        opx = self.silu(self.opfa2(opx))

        x = torch.cat((x, feature, opx), cat_dim)

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
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.device = torch.device('mps')

        # Custom Activation fucntion
        self.silu = nn.SiLU()

        # Q1
        # --- define layers here ---
        self.q1l1 = nn.Linear(state_size - 6, int(hidden_size / 6))

        self.q1l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.q1l3 = nn.Linear(int(hidden_size / 6) * 3 + int(hidden_size / 2), hidden_size)
        self.q1l4 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.q1l5 = nn.Linear(int(hidden_size / 2), int(hidden_size / 2 ** 2))
        self.q1l6 = nn.Linear(int(hidden_size / 2 ** 2), 1)

        # Feature Extraction Conv
        self.q1unet = UNet(1, 1)
        self.q1conv_fc = nn.Linear(state_size - 6, int(hidden_size / 6))

        self.q1op1 = nn.Linear(6, int(hidden_size / 2 ** 2))
        self.q1op2 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 6))

        # Q2
        # --- define layers here ---
        self.q2l1 = nn.Linear(state_size - 6, int(hidden_size / 6))

        self.q2l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.q2l3 = nn.Linear(int(hidden_size / 6) * 3 + int(hidden_size / 2), hidden_size)
        self.q2l4 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.q2l5 = nn.Linear(int(hidden_size / 2), int(hidden_size / 2 ** 2))
        self.q2l6 = nn.Linear(int(hidden_size / 2 ** 2), 1)

        # Feature Extraction Conv
        self.q2unet = UNet(1, 1)
        self.q2conv_fc = nn.Linear(state_size - 6, int(hidden_size / 6))

        self.q2op1 = nn.Linear(6, int(hidden_size / 2 ** 2))
        self.q2op2 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 6))

        self.apply(super().init_weights)

    def forward(self, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -6:]
            cat_dim = 1
        else:
            lidar_states = states[:360]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-6:]
            cat_dim = 0

        # Q1
        xs = torch.sigmoid(self.q1l1(lidar_states))
        xf = torch.sigmoid(self.q1conv_fc(self.q1unet(lidar_features)))
        if cat_dim == 1:
            xf = xf[:, 0, :]
        else:
            xf = xf[0, 0, :]
        xop = self.silu(self.q1op1(env_states))
        xop = self.silu(self.q1op2(xop))

        xa = self.silu(self.q1l2(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x = self.silu(self.q1l3(x))
        x = self.silu(self.q1l4(x))
        x1 = self.silu(self.q1l5(x))


        # Q2
        xs = torch.sigmoid(self.q2l1(lidar_states))
        xf = torch.sigmoid(self.q2conv_fc(self.q2unet(lidar_features)))
        if cat_dim == 1:
            xf = xf[:, 0, :]
        else:
            xf = xf[0, 0, :]
        xop = self.silu(self.q2op1(env_states))
        xop = self.silu(self.q2op2(xop))

        xa = self.silu(self.q2l2(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x = self.silu(self.q2l3(x))
        x = self.silu(self.q2l4(x))
        x2 = self.silu(self.q2l5(x))

        return x1, x2

    def Q1_forward(self, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :360]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -6:]
            cat_dim = 1
        else:
            lidar_states = states[:360]
            env_states = states[-6:]
            cat_dim = 0

        xs = torch.sigmoid(self.q1l1(lidar_states))
        xf = torch.sigmoid(self.q1conv_fc(self.q1unet(lidar_features)))
        if cat_dim == 1:
            xf = xf[:, 0, :]
        else:
            xf = xf[0, 0, :]
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