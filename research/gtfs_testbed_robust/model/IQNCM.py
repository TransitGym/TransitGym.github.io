import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler
from model.confidence import *
import os
from model.MetaConfidence import MC
import copy
import scipy.sparse as sp
import math


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1, seed=1):
        super(Actor, self).__init__()
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, output_size)
        self.w1 = torch.nn.Parameter(data=torch.rand([input_size, hidden_size]), requires_grad=True)
        self.b1 = torch.nn.Parameter(data=torch.rand( hidden_size ), requires_grad=True)
        self.w2 = torch.nn.Parameter(data=torch.rand([hidden_size, hidden_size]), requires_grad=True)
        self.b2 = torch.nn.Parameter(data=torch.rand(hidden_size), requires_grad=True)
        self.w3 = torch.nn.Parameter(data=torch.rand([hidden_size, hidden_size]), requires_grad=True)
        self.b3 = torch.nn.Parameter(data=torch.rand(hidden_size), requires_grad=True)
        self.w4 = torch.nn.Parameter(data=torch.rand([hidden_size, output_size]), requires_grad=True)
        self.b4 = torch.nn.Parameter(data=torch.rand(output_size), requires_grad=True)
        self.init()
    def init(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        nn.init.normal_(self.w1.data, 0.0, stdv)
        nn.init.uniform_(self.b1.data, -stdv, stdv)
        stdv = 1. / math.sqrt(self.w2.size(1))
        nn.init.normal_(self.w2.data, 0.0, stdv)
        nn.init.uniform_(self.b2.data, -stdv, stdv)

        stdv = 1. / math.sqrt(self.w3.size(1))
        nn.init.normal_(self.w3.data, 0.0, stdv)
        nn.init.uniform_(self.b3.data, -stdv, stdv)

        stdv = 1. / math.sqrt(self.w4.size(1))
        nn.init.normal_(self.w4.data, 0.0, stdv)
        nn.init.uniform_(self.b4.data, -stdv, stdv)

        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

    def forward(self, s):
        x = self.elu(s@self.w1+self.b1)
        x = self.elu(x @ self.w2 + self.b2)
        x = self.elu(x @ self.w3 + self.b3)
        x = self.elu(x @ self.w4 + self.b4)
        # x = self.elu(self.linear1(s))
        # x = self.elu(self.linear2(x))
        # x = self.elu(self.linear3(x))
        # x = self.elu(self.linear4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, n_stops=22, action_dim=1, seed=1,tau_embed_dim=32, quant_num=16,embed_num=8):
        super(Critic, self).__init__()
        hidden1 = 400

        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim + 1, hidden1)
        self.fc2 = nn.Linear(hidden1, tau_embed_dim)
        self.fc3 = nn.Linear(tau_embed_dim, tau_embed_dim)

        self.fc_tau_embedding = nn.Linear(embed_num, tau_embed_dim)
        self.fc_quantile = nn.Linear(tau_embed_dim, 1)
        # self.hyper_tau_w1 = nn.Sequential(nn.Linear(state_dim + 1, hidden1),
        #                                   nn.ReLU(),
        #                                   nn.Linear(hidden1, tau_embed_dim * embed_num))
        # self.hyper_tau_b1 = nn.Linear(state_dim + 1, tau_embed_dim)
        #
        # self.hyper_tau_w2 = nn.Sequential(nn.Linear(state_dim + 1, tau_embed_dim))
        # self.hyper_tau_b2 = nn.Linear(state_dim + 1, 1)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.n_stops = n_stops
        self.tau_embed_dim = tau_embed_dim
        self.quant_num = quant_num
        self.embed_num = embed_num

    def forward(self, xs, taus=None):
        m = xs[0].size(0)
        if taus == None:
            presum_tau = torch.rand(1, self.quant_num) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
            taus = torch.cumsum(presum_tau, dim=1)
            taus = taus.repeat([m, 1])

        batch_size = taus.shape[0]
        N = taus.shape[1]
        # embed quantile
        scale = 1 / 10. * torch.arange(
            start=1, end=self.embed_num + 1).view(1, 1, self.embed_num)
        embed = (torch.exp(
            taus.view(batch_size, N, 1) * scale
        ).view(batch_size, N, self.embed_num)) / 100.

        x, a = xs
        ego = torch.cat([x, a], 1)
        out1 = self.fc1(ego)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.relu(out1)
        # out1 = self.fc3(out1)
        # out1 = self.relu(out1)

        # w1 = torch.abs(self.hyper_tau_w1(ego))
        # b1 = self.hyper_tau_b1(ego)
        # w1 = w1.view(-1, self.embed_num, self.tau_embed_dim)
        # b1 = b1.view(-1, 1, self.tau_embed_dim)
        # tau_embeddings = (torch.bmm(embed, w1) + b1)

        # w2 = torch.abs((self.hyper_tau_w2(ego)))
        # b2 = self.hyper_tau_b2(ego)
        # w2 = w2.view(-1, self.tau_embed_dim, 1)
        # b2 = b2.view(-1, 1, 1)
        tau_embeddings = self.fc_tau_embedding(embed)
        for n in range(self.quant_num):
            feat_rand = out1 + tau_embeddings[:, n, :]
            q = self.fc_quantile(feat_rand.view(-1, 1, self.tau_embed_dim))
            # q = torch.bmm(feat_rand.view(-1, 1, self.tau_embed_dim), w2) + b2
            if n == 0:
                Q = q
            else:
                Q = torch.cat([Q, q], 1)
        Q_sorted, indices = torch.sort(Q,dim=1,descending=False)
        taus_sorted = taus[indices]
        return Q_sorted, taus_sorted


class Agent():
    def __init__(self, state_dim, name, seed, quant_num=16, n_stops=22, buslist=None, mode='n'):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0
        self.mode = mode
        self.critic = Critic(state_dim, quant_num=quant_num, n_stops=n_stops, action_dim=1, seed=seed)
        self.critic_target = Critic(state_dim, n_stops=n_stops, quant_num=quant_num, action_dim=1, seed=seed)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim, seed=seed)
        self.actor_target = Actor(self.state_dim, seed=seed)
        self.actor_for_ml = Actor(self.state_dim, seed=seed)
        self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0.0001)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_for_ml.load_state_dict(self.actor.state_dict())
        self.mc = MC(state_dim=state_dim, tau_num=quant_num)
        self.mc_optim = torch.optim.Adam(self.mc.parameters(), lr=0.0001)
        self.meta_weight_record = {}
        self.meta_learn_freq = 1
        self.learn_count = 0

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        a = self.actor(state).squeeze(0).detach().numpy()
        return a

    def get_sample(self, memory):
        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp, batch_done = [], [], [], [], [], [], []
        batch_mask = []
        batch_mask_n = []
        batch_fp_critic_t = []
        batch_actor_a = []
        for s, fp, a, r, ns, nfp, d in memory:
            batch_s.append(s)
            _fp_ = copy.deepcopy(fp)
            _fp_ = torch.tensor(_fp_, dtype=torch.float32)
            _fp_[0, 4] = self.actor(torch.tensor(s, dtype=torch.float32)).detach()
            batch_fp_critic_t.append(_fp_)
            batch_actor_a.append(self.actor(torch.tensor(s, dtype=torch.float32)))
            batch_fp.append(torch.FloatTensor(fp))

            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)
            batch_done.append(d)
            batch_nfp.append(torch.FloatTensor(nfp))
        b_fp_pad = batch_fp
        b_nfp_pad = batch_nfp
        batch_actor_a = torch.stack(batch_actor_a, 0)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_d = torch.tensor(batch_done, dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_s_ = torch.tensor(batch_ns, dtype=torch.float)


        return b_s, b_a, b_r, b_d, b_s_, b_fp_pad, b_nfp_pad

    def learn(self, memories, batch=16, bus_id=None):
        self.learn_count+=1
        n_samples = 0
        batch_s, batch_a, batch_r, batch_ns, batch_d = [], [], [], [], []
        batch = min(len(memories), batch)
        memory = random.sample(memories, batch)

        b_s, b_a, b_r, b_d, b_s_, b_fp_pad, b_nfp_pad = self.get_sample(memory)

        # update critic
        # Q [batch_size, quant_num]
        q, tau_is = self.critic([b_s, b_a])
        q_next, tau_js = self.critic_target([b_s_, self.actor_target(b_s_).detach()])
        q_target = b_r + self.gamma * q_next.view(batch, -1).detach() * (1. - b_d)
        q_target = q_target.unsqueeze(1).detach()
        q_target_sorted, indices = torch.sort(q_target, dim=1, descending=False)
        td_errors = q_target_sorted - q  # (m, quant_num, quant_num)

        kappa = 1.0

        def huber(x, k=kappa):
            return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

        element_wise_huber_loss = huber(td_errors)
        element_wise_quantile_huber_loss = torch.abs(
            tau_is[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa

        qloss = element_wise_quantile_huber_loss.sum(
            dim=1).mean(dim=1, keepdim=True)

        self.critic_optim.zero_grad()
        qloss.mean().backward()
        self.critic_optim.step()

        # update actor
        tau = torch.Tensor((2 * np.arange(tau_is.size(1)) + 1) / (2.0 * tau_is.size(1))).view(1, -1)
        tau = tau.repeat([batch, 1])
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.

        meta_risk_weights = self.mc(b_fp_pad)
        policy_loss, _ = self.critic([b_s, self.actor(b_s)], tau_hat)
        policy_loss = policy_loss.squeeze()
        policy_loss = policy_loss  * meta_risk_weights.detach()
        self.actor_optim.zero_grad()
        policy_loss = -policy_loss.mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        # if np.isnan(policy_loss.data.numpy()).any():
        # print(meta_risk_weights)
        self.actor_optim.step()

        # update meta network

        meta_risk_weights = self.mc(b_fp_pad)
        policy_loss, _ = self.critic([b_s, self.actor_for_ml(b_s)], tau_hat)
        policy_loss = policy_loss.squeeze()
        policy_loss = policy_loss * meta_risk_weights
        policy_loss = -policy_loss.mean()

        w1 = self.actor_for_ml.w1 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.w1, create_graph=True, retain_graph=True)[0]
        b1 = self.actor_for_ml.b1 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.b1, retain_graph=True)[0]
        w2 = self.actor_for_ml.w2 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.w2, retain_graph=True)[0]
        b2 = self.actor_for_ml.b2 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.b2, retain_graph=True)[0]
        w3 = self.actor_for_ml.w3 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.w3, retain_graph=True)[0]
        b3 = self.actor_for_ml.b3 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.b3, retain_graph=True)[0]
        w4 = self.actor_for_ml.w4 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.w4, retain_graph=True)[0]
        b4 = self.actor_for_ml.b4 - self.actor_optim.param_groups[0]['lr'] * torch.autograd.grad(policy_loss, self.actor_for_ml.b4, retain_graph=True)[0]

        memory = random.sample(memories, batch)
        m_b_s, m_b_a, m_b_r, m_b_d, m_b_s_, m_b_fp_pad, m_b_nfp_pad = self.get_sample(memory)
        x = F.elu(m_b_s @ w1 + b1)
        x = F.elu(x @ w2 + b2)
        x = F.elu(x @ w3 + b3)
        eval_a = F.elu(x @ w4 + b4)


        meta_loss, _ = self.critic([m_b_s, eval_a], tau_hat)
        meta_loss = meta_loss.squeeze()
        self.mc_optim.zero_grad()
        meta_loss = -meta_loss.mean()
        if self.learn_count % self.meta_learn_freq == 0:
            meta_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.mc.parameters(), 5.)
            self.mc_optim.step()
        # for name, param in self.mc.named_parameters():
        #     if param.grad!=None:
        #         print(param.grad )
        # for name, param in self.mc.named_parameters():
        #     if param.grad != None and np.isnan(policy_loss.data.numpy()).any():
        #         print(param.grad)

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)
        self.actor_for_ml.load_state_dict(self.actor.state_dict())
        print('meta loss',meta_loss.mean().data.numpy())
        return policy_loss.data.numpy(), qloss.data.numpy(),meta_loss.data.numpy()

    def record_meta(self,fp,bus_id):
        n = len(fp)
        fp = torch.tensor([fp],dtype=torch.float32)
        meta_risk_weights = self.mc(fp).detach()
        if bus_id in self.meta_weight_record:
            self.meta_weight_record[bus_id].append(np.array(meta_risk_weights).reshape(-1,).tolist()+[n])
        else:
            self.meta_weight_record[bus_id] = [np.array(meta_risk_weights).reshape(-1,).tolist()+[n]]


    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))
        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_meta.pth"
        torch.save(self.mc.state_dict(),path)
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
            # path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_meta.pth"
            # state_dict = torch.load(path)
            # self.mc.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)

            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
            state_dict = torch.load(path)
            self.critic.load_state_dict(state_dict)

            # path = abspath +"\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_meta.pth"
            # state_dict = torch.load(path)
            # self.mc.load_state_dict(state_dict)


