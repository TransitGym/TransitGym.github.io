import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler

import os

'''
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
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
    def __init__(self, state_dim, n_agents=1, action_dim=1, seed=1):
        super(Critic, self).__init__()
        self.embed_dim = 400
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.fc1 = nn.Linear(int(state_dim/n_agents) + 1, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc3 = nn.Linear(self.embed_dim, 1)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def forward(self,xs):
        x, a = xs
        # Ego evaluation
        ego = torch.cat([x, a], 1)
        out1 = self.fc1(ego)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.relu(out1)
        Q = self.fc3(out1)
        return Q

    def mixer(self,  states, actions,mask):
        # states: full state
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)

        agent_qs = self.forward([states.view(-1,int(self.state_dim/self.n_agents)),actions.view(-1,1)])*mask.view(-1,1)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = self.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot



class Agent():
    def __init__(self, state_dim, name, seed, n_stops=22, buslist=None):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.agent_num = len( buslist)
        self.learn_step_counter = 0
        self.buslist = buslist
        b = 0
        self.bus_hash = {}
        for k, v in self.buslist.items():
            self.bus_hash[k] = b
            b += 1

        input_dim = state_dim * self.agent_num  #  state*agent_num
        self.critic = Critic(input_dim,  action_dim=1, seed=seed,n_agents=self.agent_num)
        self.critic_target = Critic(input_dim, action_dim=1, seed=seed,n_agents=self.agent_num)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim, seed=seed)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)


    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        a = self.actor(state).squeeze(0).detach().numpy()
        return a

    def learn(self, memories, batch=16, bus_id=None):

        batch_s, batch_a, batch_r, batch_ns, batch_obs,batch_nobs, batch_as,batch_nas = [], [], [], [], [],[], [],[]
        batch = min(len(memories), batch)
        memory = random.sample(memories, batch)
        batch_qs_mask = []
        batch_nqs_mask = []
        batch_d = []
        for s, fp, a, r, ns, nfp  in memory:
            batch_s.append(s)
            batch_a.append(a)
            batch_ns.append(ns)
            wobs = [0. for _ in range(self.agent_num*self.state_dim)]
            wa = [0. for _ in range(self.agent_num)]
            qs_mask = [0. for _ in range(self.agent_num)]
            k=0
            for i_ in range(len(fp)):
                fp_ = fp[i_]
                b = fp_[-1]
                gap = fp_[-2]
                if gap == 0:
                    wa[k] = fp_[self.state_dim]
                    wobs[k * self.state_dim:(k + 1) * self.state_dim] = fp_[:self.state_dim]
                    qs_mask[k] = 1
                    k+=1
                    # wa[self.bus_hash[b]] = fp_[self.state_dim]
                    # wobs[self.bus_hash[b]*self.state_dim:(self.bus_hash[b]+1)*self.state_dim] = fp_[:self.state_dim]
                    # qs_mask[self.bus_hash[b]] = 1
            batch_obs.append(wobs)
            batch_as.append(wa)
            batch_qs_mask.append(qs_mask)

            nqs_mask = [0. for _ in range(self.agent_num)]
            wobs_ = [0. for _ in range(self.agent_num * self.state_dim)]
            wa_ = [0. for _ in range(self.agent_num)]
            k = 0
            for i_ in range(len(nfp)):
                fp_ = nfp[i_]
                b = fp_[-1]
                gap = fp_[-2]
                if gap == 0:
                    wa_[k] = fp_[self.state_dim]
                    wobs_[k * self.state_dim:(k + 1) * self.state_dim] = fp_[:self.state_dim]
                    nqs_mask[k] = 1
                    k += 1
                    # wa_[self.bus_hash[b]] = fp_[self.state_dim]
                    # wobs_[self.bus_hash[b]*self.state_dim:(self.bus_hash[b]+1)*self.state_dim] = fp_[:self.state_dim]
                    # nqs_mask[self.bus_hash[b]] = 1
            batch_nobs.append(wobs_)
            batch_nas.append(wa_)
            batch_nqs_mask.append(nqs_mask)
            batch_r.append(r)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        batch_nobs = torch.tensor(batch_nobs, dtype=torch.float)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_as = torch.tensor(batch_as, dtype=torch.float)
        batch_nas = torch.tensor(batch_nas, dtype=torch.float)
        batch_nqs_mask = torch.tensor(batch_nqs_mask, dtype=torch.float)
        batch_qs_mask = torch.tensor(batch_qs_mask, dtype=torch.float)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_s_ = torch.tensor(batch_ns, dtype=torch.float)


        # update critic
        q = self.critic.mixer( batch_obs,batch_as,batch_qs_mask).view(-1,1)
        q_next = self.critic_target.mixer( batch_nobs,batch_nas,batch_nqs_mask).detach().view(-1,1)
        q_target = b_r + self.gamma *q_next
        loss_fn = nn.MSELoss()
        qloss = loss_fn(q, q_target)
        self.critic_optim.zero_grad()
        qloss.mean().backward()
        self.critic_optim.step()

        # update actor
        policy_loss = -self.critic([b_s,self.actor(b_s)]).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()



        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.critic_target, self.critic, tau=0.02)


        return policy_loss.data.numpy(), qloss.data.numpy()

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))
        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)


    def load(self, model):
        try:
            abspath = os.path.abspath(os.path.dirname(__file__))
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            print('Load: ' + path)
            self.actor.load_state_dict(state_dict)
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
            state_dict = torch.load(path)
            # print('Load: ' + path)
            # self.critic.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            print('Load: ' + path)
            self.actor.load_state_dict(state_dict)

            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
            # state_dict = torch.load(path)
            # self.critic.load_state_dict(state_dict)


