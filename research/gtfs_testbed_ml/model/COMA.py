import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler
import pandas as pd
import os
'''
Counterfactual Multi-Agent Policy Gradients
https://ojs.aaai.org/index.php/AAAI/article/view/11794
'''

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1, seed=1):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(input_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, output_size)
        self.mean_linear.weight.data.uniform_(-1e-3, 1e-3)
        self.mean_linear.bias.data.uniform_(-1e-3, 1e-3)

        self.log_std_linear = nn.Linear(hidden_size, output_size)
        self.log_std_linear.weight.data.uniform_(-1e-3, 1e-3)
        self.log_std_linear.bias.data.uniform_(-1e-3, 1e-3)

        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

    def forward(self, s):
        x1 =  self.elu(self.linear1(s))
        x1 =  self.elu(self.linear2(x1))
        x2 = self.elu(self.linear3(s))
        x2 = self.elu(self.linear4(x2))
        mean = self.elu(self.mean_linear(x1))
        log_std = (self.log_std_linear(x2))
        log_std = torch.clamp(log_std, -20., 2.)

        return mean, log_std


    def evaluate(self, s, epsilon=1e-6):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = Normal(0, 1.)
        z = normal.sample()
        action = (mean + std * z )

        log_prob = Normal(mean, std).log_prob(mean + std * z ) #- torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def act(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = Normal(0, 1.)
        z = normal.sample()
        action = (mean+std*z)
        return action[0]

class Critic(nn.Module):
    def __init__(self, state_dim,  action_dim=1, seed=1):
        super(Critic, self).__init__()
        hidden1 = 400

        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim , hidden1)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Linear(hidden1, 1)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()


    def forward(self, s):
        out1 = self.fc1(s)
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
        self.agent_num = len( buslist)
        self.learn_step_counter = 0
        self.buslist = buslist
        b = 0
        self.bus_hash = {}
        for k, v in self.buslist.items():
            self.bus_hash[k] = b
            b += 1

        input_dim =   state_dim * self.agent_num + self.agent_num # id + state*agent_num + action
        self.critic = Critic(input_dim,  action_dim=1, seed=seed)
        self.critic_target = Critic(input_dim, action_dim=1, seed=seed)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.005)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim, seed=seed)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.00006)


    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        a = self.actor.act(state).squeeze(0).detach().numpy()
        log_std_df = pd.DataFrame()
        log_std_df['a'] = list(a)
        try:
            if self.log==0:
                log_std_df.to_csv('record.csv', mode='a', header=False)
        except:
            self.log=0
            log_std_df.to_csv('record.csv')
        return a

    def build_input_critic(self, agent_id, observations, actions):
        batch_size = len(observations)

        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)

        observations = torch.tensor(observations, dtype=torch.float).view(batch_size, self.state_dim * self.agent_num)
        actions = torch.tensor(actions, dtype=torch.float).view(batch_size,self.agent_num)
        input_critic = torch.cat([observations, actions], dim=-1)
        # input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic

    def learn(self, memories, batch=16, bus_id=None):

        n_samples = 0
        batch_s, batch_a, batch_r, batch_ns, batch_obs,batch_nobs, batch_as,batch_nas = [], [], [], [], [],[], [],[]
        batch = min(len(memories), batch)
        memory = random.sample(memories, batch)

        for s, fp, a, r, ns, nfp in memory:
            batch_s.append(s)
            batch_a.append(a)

            batch_ns.append(ns)

            wobs = [0. for _ in range(self.agent_num*self.state_dim)]
            wa = [0. for _ in range(self.agent_num)]
            k = 0
            for i_ in range(len(fp)):
                fp_ = fp[i_]
                b = fp_[-1]
                gap = fp_[-2]
                if gap == 0:
                    wa[k] = fp_[self.state_dim]
                    wobs[k* self.state_dim:(k+ 1) * self.state_dim] = fp_[ :self.state_dim]
                    k+=1
                    # wa[self.bus_hash[b]] = fp_[self.state_dim]
                    # wobs[self.bus_hash[b]*self.state_dim:(self.bus_hash[b]+1)*self.state_dim] = fp_[:self.state_dim]

            batch_obs.append(wobs)
            batch_as.append(wa)


            wobs_ = [0. for _ in range(self.agent_num * self.state_dim)]
            wa_ = [0. for _ in range(self.agent_num)]
            k=0
            for i_ in range(len(nfp)):
                fp_ = nfp[i_]
                b = fp_[-1]
                gap = fp_[-2]
                if gap == 0:
                    wa_[k] = fp_[self.state_dim]
                    wobs_[k * self.state_dim:(k+ 1) * self.state_dim] = fp_[:self.state_dim]
                    k+=1
                    # wa_[self.bus_hash[b]] = fp_[self.state_dim]
                    # wobs_[self.bus_hash[b]*self.state_dim:(self.bus_hash[b]+1)*self.state_dim] = fp_[:self.state_dim]

            batch_nobs.append(wobs_)
            batch_nas.append(wa_)
            batch_r.append(r)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        batch_input_critic = self.build_input_critic(self.bus_hash[bus_id], batch_obs, batch_as)
        batch_input_critic_ = self.build_input_critic(self.bus_hash[bus_id], batch_nobs, batch_nas)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_s_ = torch.tensor(batch_ns, dtype=torch.float)

        # update critic
        q = self.critic(batch_input_critic)
        q_next = self.critic_target(batch_input_critic_).detach()
        q_target = b_r + self.gamma *q_next
        loss_fn = nn.MSELoss()
        qloss = loss_fn(q, q_target)
        self.critic_optim.zero_grad()
        qloss.mean().backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optim.step()

        # update actor
        baseline = []
        sample=10
        for i in range(batch_input_critic.size(0)):
            prob = np.zeros(sample)
            q = np.zeros(sample)
            for j in range(sample):
                input_critic_b = batch_input_critic[i]
                action, log_prob, z, mean, log_std = self.actor.evaluate(b_s[i])
                input_critic_b[self.state_dim * self.agent_num + self.bus_hash[bus_id]] = action.detach()
                prob[j]=torch.exp(Normal(mean, torch.exp(log_std)).log_prob(action )).detach().item()
                q[j]=self.critic_target(input_critic_b).detach().item()
            prob = prob/np.sum(prob)
            baseline.append(np.sum(prob*q))

        baseline = torch.tensor(baseline, dtype=torch.float).view(-1, 1)
        advantage = self.critic(batch_input_critic).detach() - baseline
        new_a, action_log_probs, epsilon, mean, log_std = self.actor.evaluate(b_s)

        # log_std_df = pd.DataFrame()
        # log_std_df['log_std']= list(log_std.detach().numpy())
        # log_std_df['log_m'] = list(mean.detach().numpy())
        # log_std_df['a'] = list(new_a.detach().numpy())
        # try:
        #     if self.log==0:
        #         log_std_df.to_csv('record.csv', mode='a', header=False)
        # except:
        #     self.log=0
        #     log_std_df.to_csv('record.csv')

        policy_loss = -(Normal(mean, log_std.exp()).log_prob(b_a)*advantage).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_optim.step()
        # for name, param in self.actor.named_parameters():
        #     if param.grad!=None:
        #         print(name )
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


