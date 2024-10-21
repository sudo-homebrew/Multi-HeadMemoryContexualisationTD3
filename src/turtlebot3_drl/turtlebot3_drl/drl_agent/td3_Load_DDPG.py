import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..common.manualaction import ManualAction
from ..common.settings import POLICY_NOISE, POLICY_NOISE_CLIP, POLICY_UPDATE_FREQUENCY
from ..common.ounoise import OUNoise

from .off_policy_agent import OffPolicyAgent, Network

from .unet_lite2 import UNet
from .ddpg import DDPG

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# https://github.com/djbyrne/TD3
# Modified by: Seunghyeop
# Description: This code has been modified to train the Turtlebot3 Waffle_pi model.
# TD3 with UNET and FCNN2


class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        self.env_size = 6
        self.state_size = state_size - self.env_size
        # --- define layers here ---
        self.fa1 = nn.Linear(state_size - self.env_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, int(hidden_size * 2))
        self.fa3 = nn.Linear(int(hidden_size * 2), hidden_size // 2)

        # --- conv layers for feature extraction ---
        output_size = 4
        self.unet = UNet(1, output_size)
        self.conv_fc1 = nn.Linear((state_size - self.env_size) * output_size, hidden_size)
        self.conv_fc2 = nn.Linear(hidden_size, hidden_size // 2)

        # Env NN
        self.opfa1 = nn.Linear(6, hidden_size // 2 ** 2)
        self.opfa2 = nn.Linear(hidden_size // 2 ** 2, hidden_size // 2)

        # concated NN
        self.fin1 = nn.Linear(int((hidden_size // 2) * 3), int(hidden_size * 2))
        self.fin2 = nn.Linear(int(hidden_size * 2), int(hidden_size * 2))
        self.fin3 = nn.Linear(int(hidden_size * 2), hidden_size)
        self.fin4 = nn.Linear(hidden_size, int(hidden_size / 2 ** 1))

        # self.fin5 = nn.Linear(int(hidden_size / 2 ** 1), action_size)

        self.fin5 = nn.Linear(int(hidden_size / 2 ** 1), int(hidden_size / 2 ** 2))
        self.fin6 = nn.Linear(int(hidden_size / 2 ** 2), int(hidden_size / 2 ** 3))
        self.fin7 = nn.Linear(int(hidden_size / 2 ** 3), action_size)

        self.dropout = nn.Dropout(0.5)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2 ** 1)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2 ** 2)
        self.layer_norm4 = nn.LayerNorm(hidden_size // 2 ** 3)
        self.layer_norm5 = nn.LayerNorm(int(hidden_size * 2))
        self.layer_norm6 = nn.LayerNorm(2)

        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2 ** 1)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2 ** 1)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size // 2 ** 1)
        self.batch_norm5 = nn.BatchNorm1d(int(hidden_size * 2))

        # Custom Activation fucntion
        self.silu = nn.SiLU()
        self.custom = Custom()

        self.apply(super().init_weights)


        file_name = '/Users/sunghjopnam/turtlebot3_drlnav/src/turtlebot3_drl/model/examples_waffle_pi/ddpg_0_stage_10/actor_stage10_episode15400.pt'
        tmp_model = DDPG('mps', 1)
        tmp_model.networks[0].load_state_dict(torch.load(file_name, 'mps'))
        with torch.no_grad():
                    self.fin2.weight.copy_(tmp_model.networks[0].fa3.weight)
                    self.fin3.weight.copy_(tmp_model.networks[0].fa4.weight)
                    self.fin4.weight.copy_(tmp_model.networks[0].fa5.weight)
                    self.fin5.weight.copy_(tmp_model.networks[0].fa6.weight)
                    self.fin6.weight.copy_(tmp_model.networks[0].fa7.weight)
                    self.fin7.weight.copy_(tmp_model.networks[0].fa8.weight)

                    self.fin2.bias.copy_(tmp_model.networks[0].fa3.bias)
                    self.fin3.bias.copy_(tmp_model.networks[0].fa4.bias)
                    self.fin4.bias.copy_(tmp_model.networks[0].fa5.bias)
                    self.fin5.bias.copy_(tmp_model.networks[0].fa6.bias)
                    self.fin6.bias.copy_(tmp_model.networks[0].fa7.bias)
                    self.fin7.bias.copy_(tmp_model.networks[0].fa8.bias)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        if len(states.size()) == 2:
            lidar_states = states[:, :self.state_size]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -self.env_size:]
            cat_dim = 1
        else:
            lidar_states = states[:self.state_size]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-self.env_size:]
            cat_dim = 0

        x1 = self.custom(self.fa1(lidar_states))
        x2 = self.custom(self.fa2(x1))
        x3 = self.silu(self.layer_norm2(self.fa3(x2)))

        feature1 = self.silu(self.conv_fc1(torch.flatten(self.unet(lidar_features), start_dim=cat_dim)))
        feature2 = self.silu(self.layer_norm2(self.conv_fc2(feature1)))

        opx1 = self.custom(self.opfa1(env_states * 2))
        opx2 = self.custom(self.opfa2(opx1))

        x5 = torch.cat((x3, feature2, opx2), cat_dim)

        if cat_dim == 1:
            x6 = self.silu(self.batch_norm5(self.fin1(x5)))
            x7 = self.silu(self.batch_norm5(self.fin2(x6)))
            x8 = self.custom(self.batch_norm1(self.fin3(x7)))
            x9 = self.custom(self.batch_norm2(self.fin4(x8)))
            # action = torch.tanh(self.fin5(x))
            x10 = self.custom(self.layer_norm3(self.fin5(x9)))
            x11 = self.custom(self.layer_norm4(self.fin6(x10)))
        else:
            x6 = self.silu(self.fin1(x5))
            x7 = self.silu(self.fin2(x6))
            x8 = self.custom(self.fin3(x7))
            x9 = self.custom(self.fin4(x8))
            # action = torch.tanh(self.fin5(x))
            x10 = self.custom(self.fin5(x9))
            x11 = self.custom(self.fin6(x10))

        x12 = self.fin7(x11)
        action = torch.tanh(x12)
        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, x12, [x1, x2, x3, feature1, feature2,
                                                        opx1, opx2, x6, x7, x8, x9, x10, x11],
                                                        [self.fa1.bias, self.fa2.bias, self.fa3.bias, self.conv_fc1.bias, self.conv_fc2.bias,
                                                        self.opfa1.bias, self.opfa2.bias, self.fin1.bias, self.fin2.bias, self.fin3.bias, self.fin4.bias,
                                                        self.fin5.bias, self.fin6.bias])
        # -- define layers to visualize until here ---
        return action


class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.env_size = 6
        self.state_size = state_size - self.env_size
        # --- define Q1 layers here ---
        self.q1fa = nn.Linear(state_size - self.env_size, hidden_size // 2 ** 3)

        # --- Q1 conv layers for feature extraction ---
        output_size = 3
        self.q1unet = UNet(1, output_size)
        self.q1conv_fc = nn.Linear((state_size - self.env_size) * output_size, hidden_size // 2 ** 3)

        # Q1 Env NN
        self.q1op = nn.Linear(self.env_size, hidden_size // 2 ** 3)

        # Q1 Action NN
        self.q1a = nn.Linear(action_size, hidden_size // 2 ** 3)

        # Q1 concated NN
        self.q1fin = nn.Linear(hidden_size // 2, 1)


        # --- define Q2 layers here ---
        self.q2fa1 = nn.Linear(state_size - self.env_size, hidden_size // 2 ** 3)

        # --- define Q2 layers here ---
        self.q2fa = nn.Linear(state_size - self.env_size, hidden_size // 2 ** 3)

        # --- Q2 conv layers for feature extraction ---
        output_size = 3
        self.q2unet = UNet(1, output_size)
        self.q2conv_fc = nn.Linear((state_size - self.env_size) * output_size, hidden_size // 2 ** 3)

        # Q2 Env NN
        self.q2op = nn.Linear(self.env_size, hidden_size // 2 ** 3)

        # Q2 Action NN
        self.q2a = nn.Linear(action_size, hidden_size // 2 ** 3)

        # Q2 concated NN
        self.q2fin = nn.Linear(hidden_size // 2, 1)



        # Drop out and normalisations
        self.dropout = nn.Dropout(0.5)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(int(hidden_size / 2 ** 1))

        # Custom Activation fucntion
        self.silu = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :self.state_size]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -self.env_size:]
            cat_dim = 1
        else:
            lidar_states = states[:self.state_size]
            lidar_features = torch.unsqueeze(lidar_states, dim=0).unsqueeze(dim=0)
            env_states = states[-self.env_size:]
            cat_dim = 0

        # Q1
        xs = self.silu(self.q1fa(lidar_states))
        xf = self.silu(self.q1conv_fc(torch.flatten(self.q1unet(lidar_features), start_dim=cat_dim)))
        xop = self.silu(self.q1op(env_states))
        xa = self.silu(self.q1a(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x1 = self.silu(self.q1fin(x))


        # Q2
        xs = self.silu(self.q2fa(lidar_states))
        xf = self.silu(self.q2conv_fc(torch.flatten(self.q1unet(lidar_features), start_dim=cat_dim)))
        xop = self.silu(self.q2op(env_states))
        xa = self.silu(self.q2a(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x2 = self.silu(self.q1fin(x))

        return x1, x2

    def Q1_forward(self, states, actions):
        if len(states.size()) == 2:
            lidar_states = states[:, :self.state_size]
            lidar_features = lidar_states.unsqueeze(dim=1)
            env_states = states[:, -self.env_size:]
            cat_dim = 1
        else:
            lidar_states = states[:self.state_size]
            env_states = states[-self.env_size:]
            cat_dim = 0

        xs = self.silu(self.q1fa(lidar_states))
        xf = self.silu(self.q1conv_fc(torch.flatten(self.q1unet(lidar_features), start_dim=cat_dim)))
        xop = self.silu(self.q1op(env_states))
        xa = self.silu(self.q1a(actions))
        x = torch.cat((xs, xf, xop, xa), cat_dim)
        x1 = self.silu(self.q1fin(x))

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
        # self.actor_lr_scheduler = self.create_lr_scheduler(self.actor_optimizer)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)
        # self.critic_lr_scheduler = self.create_lr_scheduler(self.critic_optimizer)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(dtype=torch.float32, device=self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(dtype=torch.float32, device=self.device)
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
        # self.critic_lr_scheduler.step()

        if self.iteration % self.policy_freq == 0:
            # optimize actor
            loss_actor = -1 * self.critic.Q1_forward(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()
            # self.actor_lr_scheduler.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss]

class CustomActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = (input * torch.tanh(input) + input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = (torch.tanh(input) + input * (1 - torch.tanh(input) ** 2) + 1)

        return grad_output * grad_input

class Custom(nn.Module):
    def forward(self, input):
        return CustomActivation.apply(input)