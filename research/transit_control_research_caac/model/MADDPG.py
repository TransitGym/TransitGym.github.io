import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler
import os


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1,seed=1):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def forward(self, s):
        x = self.elu(self.linear1(s))
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        x =  self.elu(self.linear4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, n_stops=22, action_dim=1, seed=1,n_buses=1):
        super(Critic, self).__init__()
        hidden1 = 400

        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim + 1  , hidden1)
        self.fc12 = nn.Linear(n_buses-1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Linear(hidden1, 1)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.n_stops = n_stops


    def forward(self, xs):
        x,  wa,a  = xs
        h1 = torch.cat([x,  a ], 1)
        out1 = self.fc1(h1) + self.fc12(wa)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.relu(out1)
        Q = self.fc3(out1)

        return Q


class Agent():
    def __init__(self, state_dim, name, seed, n_stops=22, buslist=None):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0
        self.buslist = buslist

        self.critic = Critic(state_dim, n_stops=n_stops, action_dim=1, seed=seed,n_buses=len(buslist))
        self.critic_target = Critic(state_dim, n_stops=n_stops, action_dim=1, seed=seed,n_buses=len(buslist))
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

    def learn(self, memories, batch=16 ,bus_id=None):

        batch_s, batch_a, batch_r, batch_ns,batch_d = [], [], [], [], []
        batch = min(len(memories), batch)
        memory = random.sample(memories, batch)

        whole_a = []
        whole_na = []

        b = 0
        self.bus_hash = {}
        for k,v in self.buslist.items():
            if k!=bus_id:
                self.bus_hash[k]=b
                b+=1

        for s, fp, a, r, ns, nfp  in memory:
            batch_s.append(s)
            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)
            wa = [0. for _ in range(len(self.buslist)-1)]
            for i_ in range(len(fp)):
                fp_=fp[i_]
                b = fp_[-1]
                gap = fp_[-2]

                if i_>0 and gap==0:
                    wa[self.bus_hash[b]]=fp_[self.state_dim ]
            whole_a.append(wa)

            wa = [0. for _ in range(len(self.buslist)-1 )]
            for i_ in range(len(nfp)):
                fp_=nfp[i_]
                b = fp_[-1]
                gap = fp_[-2]
                if i_ > 0 and gap == 0:
                    wa[self.bus_hash[b]]=fp_[self.state_dim ]
            whole_na.append(wa)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_s_ = torch.tensor(batch_ns, dtype=torch.float)

        wb_a  = torch.tensor(whole_a , dtype=torch.float).view(-1, len(self.buslist)-1)
        wb_na = torch.tensor(whole_na, dtype=torch.float).view(-1, len(self.buslist)-1)

        # update critic
        Q = self.critic([b_s, wb_a, b_a])
        q_target = b_r + self.gamma * self.critic_target([b_s_, wb_na, self.actor_target(b_s_).detach() ]).detach()
        loss_fn = nn.MSELoss()
        qloss = loss_fn(Q, q_target)
        self.critic_optim.zero_grad()
        qloss.mean().backward()
        self.critic_optim.step()
        # update actor
        policy_loss = self.critic([b_s, wb_a, self.actor(b_s) ])

        # take gradient step
        self.actor_optim.zero_grad()
        policy_loss = -policy_loss
        policy_loss.mean().backward()
        self.actor_optim.step()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)
        print('maddpg train...')

        return policy_loss.data.numpy(), qloss.data.numpy()

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))
        path = abspath + "/save/" + str(self.name) + '_' + str(model) +str(self.seed)+ "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) +str(self.seed)+ "_critic.pth"
        torch.save(self.critic.state_dict(), path)

    def load(self, model):
        try:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "/save/" + str(self.name) + '_' + str(model) +str(self.seed)+ "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)


