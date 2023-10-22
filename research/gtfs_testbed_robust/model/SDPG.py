import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler

import os

'''
SAMPLE-BASED DISTRIBUTIONAL POLICY GRADIENT  
https://arxiv.org/pdf/2001.02652.pdf
'''


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1, seed=1):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

    def forward(self, s):
        x = self.elu(self.linear1(s))
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        x = self.elu(self.linear4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, n_stops=22, action_dim=1, seed=1, noise_dim=1, atom_num=51):
        super(Critic, self).__init__()
        hidden1 = 400

        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim + 1 + noise_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Linear(hidden1, 1)
        self.fc_noise = nn.Linear(1,noise_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.n_stops = n_stops

        self.noise_dim = noise_dim
        self.quant_num = atom_num

        # produce Q based on noise and use fixed quantile to match the distribution
        self.tau = torch.Tensor((2 * np.arange(atom_num) + 1) / (2.0 * atom_num)).view(1, -1)
        print('tau size',self.tau.size())

    # for ablation study
    def forward(self, xs):
        m = xs[0].size(0)
        noise = torch.normal(0,1,[m,self.quant_num, 1])
        # noise embedding
        noise_emb = self.fc_noise(noise)
        x, a = xs
        # Ego evaluation
        for n in range(self.quant_num):
            ego = torch.cat([x, a, noise_emb[:,n,:].view(-1,self.noise_dim)], 1)
            out1 = self.fc1(ego)
            out1 = self.relu(out1)
            out1 = self.fc2(out1)
            out1 = self.relu(out1)
            if n==0:
                Q = self.fc3(out1)
            else:
                Q = torch.cat([Q,self.fc3(out1)],1)
        # Q:[batch_size, n_atoms]
        return Q


class Agent():
    def __init__(self, state_dim, name, seed, n_stops=22, buslist=None):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0

        self.critic = Critic(state_dim, n_stops=n_stops, action_dim=1, seed=seed)
        self.critic_target = Critic(state_dim, n_stops=n_stops, action_dim=1, seed=seed)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim, seed=seed)
        self.actor_target = Actor(self.state_dim, seed=seed)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        a = self.actor(state).squeeze(0).detach().numpy()

        return a

    def learn(self, memories, batch=16, bus_id=None):

        n_samples = 0
        batch_s, batch_a, batch_r, batch_ns, batch_d = [], [], [], [], []
        batch = min(len(memories), batch)
        memory = random.sample(memories, batch)

        for s, fp, a, r, ns, nfp, d in memory:
            batch_s.append(s)
            batch_d.append(d)
            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_d = torch.tensor(batch_d, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_s_ = torch.tensor(batch_ns, dtype=torch.float)

        # update critic
        #Q [batch_size, n_atoms]
        q = self.critic([b_s, b_a]).view(batch,-1)
        q_sorted, indices = torch.sort(q)
        #q_target [batch_size, n_atoms]
        q_target = b_r + self.gamma * self.critic_target([b_s_, self.actor_target(b_s_).detach()]).detach().view(batch,-1) * (1. - b_d)
        q_target_sorted, indices = torch.sort(q_target)
        # (m , 1, n_atoms)
        q_target_sorted = q_target_sorted.unsqueeze(1).detach()
        q_sorted = q_sorted.unsqueeze(2)
        u = q_target_sorted - q_sorted # (m, n_atoms, n_atoms)
        weight = torch.abs(self.critic.tau.unsqueeze(0) - u.le(0.).float())  # (m, n_atoms, n_atoms)
        def huber(x, k=1.0):
            return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
        loss = huber(u)
        # (m, n_atoms, n_atoms)
        qloss = torch.mean(weight * loss, dim=1).mean(dim=1)

        self.critic_optim.zero_grad()
        qloss.mean().backward()
        self.critic_optim.step()

        # update actor
        # self.actor.zero_grad()
        policy_loss = self.critic([b_s, self.actor(b_s)])

        # take gradient step
        self.actor_optim.zero_grad()
        policy_loss = -policy_loss.mean(dim=1)
        policy_loss.mean().backward()
        self.actor_optim.step()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)


        return policy_loss.data.numpy(), qloss.data.numpy()

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))
        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

        # print('Save: ' + abspath + "/save/" + str(self.name) +'_'+str(model))

    def load(self, model):
        try:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
            state_dict = torch.load(path)
            self.critic.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)

            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
            state_dict = torch.load(path)
            self.critic.load_state_dict(state_dict)


