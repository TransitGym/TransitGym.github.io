import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler
from model.confidence import *
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

class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_dim=1, noisy_net=False):
        super(CosineEmbeddingNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        self.net = nn.Sequential(
            linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings

class NondecreasingEmbeddingNetwork(nn.Module):
    def __init__(self, num_exps=64, embedding_dim=12, noisy_net=False):
        super(NondecreasingEmbeddingNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        r1 = torch.tensor(torch.sqrt(torch.tensor(3./num_exps,dtype=torch.float32)))
        x = (r1 - 0.)* torch.rand( num_exps, embedding_dim) + 0.
        self.mo_weight =torch.nn.Parameter( torch.tensor(torch.log(x),requires_grad=True).view(1,num_exps, embedding_dim))

        self.bias = torch.nn.Parameter(torch.normal(0, 0.01, [embedding_dim]), requires_grad=True)
        self.relu = nn.ReLU()
        self.num_exps = num_exps
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        scale = 1/10. * torch.arange(
            start=1, end=self.num_exps+1).view(1, 1, self.num_exps)

        # Calculate cos(i * \pi * \tau).
        exps = (torch.exp(
            taus.view(batch_size, N, 1) * scale
            ).view(batch_size * N, self.num_exps)-1)/100.

        # Calculate embeddings of taus.

        tau_embeddings = torch.bmm(exps.view(-1, 1, self.num_exps),torch.exp(self.mo_weight).repeat(batch_size * N,1,1).view(-1,self.num_exps,self.embedding_dim))+self.bias
        tau_embeddings = self.relu(tau_embeddings).view(batch_size,N,-1)
        return tau_embeddings

class Critic(nn.Module):
    def __init__(self, state_dim, n_stops=22, action_dim=1, seed=1,tau_embed_dim=64, quant_num=16,embed_num=32):
        super(Critic, self).__init__()
        hidden1 = 400

        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim + 1 , hidden1)
        self.fc2 = nn.Linear(hidden1, tau_embed_dim)
        self.fc3 = nn.Linear(tau_embed_dim, tau_embed_dim)
        self.fc_tau_embedding = nn.Linear(embed_num, tau_embed_dim)
        self.fc_quantile = nn.Linear(tau_embed_dim, 1)
        # self.hyper_tau_w1 = nn.Sequential(nn.Linear(state_dim + 1,hidden1),
        #                               nn.ReLU(),
        #                               nn.Linear(hidden1, tau_embed_dim * embed_num) )
        # self.hyper_tau_b1 = nn.Linear(state_dim + 1, tau_embed_dim)
        #
        # self.hyper_tau_w2 = nn.Sequential(nn.Linear(state_dim + 1, tau_embed_dim)  )
        # self.hyper_tau_b2 = nn.Linear(state_dim + 1, 1)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.n_stops = n_stops
        self.tau_embed_dim =tau_embed_dim
        self.quant_num = quant_num
        self.embed_num = embed_num
        # r1 = torch.tensor(torch.sqrt(torch.tensor(3./tau_embed_dim,dtype=torch.float32)))
        # x = (r1 - 0.)* torch.rand(tau_embed_dim,1) + 0.
        # self.mo_weight = torch.nn.Parameter(torch.tensor(torch.log(x),requires_grad=True))
        # self.bias = torch.nn.Parameter(torch.normal(0, 0.01, [ 1]), requires_grad=True)
        # self.cosine_net = CosineEmbeddingNetwork(
        #     num_cosines=quant_num, embedding_dim=tau_embed_dim)
        # self.exp_net = NondecreasingEmbeddingNetwork(num_exps=quant_num, embedding_dim=tau_embed_dim)
    def forward(self, xs,taus=None):
        m = xs[0].size(0)
        if taus == None:
            presum_tau = torch.rand(1, self.quant_num) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
            taus = torch.cumsum(presum_tau, dim=1)
            taus = taus.repeat([m, 1])

        batch_size = taus.shape[0]
        N = taus.shape[1]
        # Calculate i * \pi (i=1,...,N).
        scale = 1 / 10. * torch.arange(
            start=1, end=self.embed_num + 1).view(1, 1, self.embed_num)
        # Calculate cos(i * \pi * \tau).
        embed = (torch.exp(
            taus.view(batch_size, N, 1) * scale
        ).view(batch_size,N, self.embed_num )) / 100.


        # for i in range(100):
        #     print(i,self.exp_net(torch.tensor(i/100,dtype=torch.float32).view(-1,1)).data.numpy().reshape(-1,))
        x, a = xs
        ego = torch.cat([x, a], 1)
        out1 = self.fc1(ego)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        # out1 = self.relu(out1)
        # out1 = self.fc3(out1)

        # w1 = torch.abs(self.hyper_tau_w1(ego))
        # b1 = self.hyper_tau_b1(ego)
        # w1 = w1.view(-1, self.embed_num, self.tau_embed_dim)
        # b1 = b1.view(-1, 1, self.tau_embed_dim)
        # tau_embeddings = (torch.bmm(embed, w1) + b1)
        #
        # w2 = torch.abs((self.hyper_tau_w2(ego)))
        # b2 = self.hyper_tau_b2(ego)
        # w2 = w2.view(-1,  self.tau_embed_dim,1)
        # b2 = b2.view(-1, 1, 1)
        tau_embeddings = self.fc_tau_embedding(embed)
        for n in range(self.quant_num):
            feat_rand = out1 + tau_embeddings[:, n, :]
            # print(tau_embeddings[:,n,:])
            q = self.fc_quantile(feat_rand.view(-1, 1, self.tau_embed_dim))
            # q =  torch.bmm(feat_rand.view(-1,1,self.tau_embed_dim), w2) + b2
            if n==0:
                Q = q
            else:
                Q = torch.cat([Q,q],1)
        Q_sorted, indices = torch.sort(Q, dim=1, descending=False)
        taus_sorted = taus[indices]
        return Q_sorted,taus_sorted


class Agent():
    def __init__(self, state_dim, name, seed, n_stops=22, buslist=None,mode='n'):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0
        self.mode = mode
        self.critic = Critic(state_dim, n_stops=n_stops, action_dim=1, seed=seed)
        self.critic_target = Critic(state_dim, n_stops=n_stops, action_dim=1, seed=seed)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim, seed=seed)
        self.actor_target = Actor(self.state_dim, seed=seed)
        self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0.0001)
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
        #Q [batch_size, quant_num]
        q, tau_is = self.critic([b_s, b_a])
        q_next, tau_js = self.critic_target([b_s_, self.actor_target(b_s_).detach()])
        q_target = b_r + self.gamma * q_next.view(batch,-1).detach() * (1. - b_d)
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
        # print(q.data.numpy().reshape(batch,-1)[0])
        # for name, param in self.critic.named_parameters():
        #     if param.grad!=None:
        #         print(name )

        # update actor
        tau = torch.Tensor((2 * np.arange(tau_is.size(1)) + 1) / (2.0 * tau_is.size(1))).view(1, -1)
        tau  = tau.repeat([batch,1])
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        policy_loss, _ = self.critic([b_s, self.actor(b_s)],tau_hat)
        with torch.no_grad():
            risk_weights = distortion_de(tau_hat, mode=self.mode)
        policy_loss = policy_loss.squeeze()
        policy_loss = policy_loss*risk_weights#*(torch.cat([tau[:,0].view(-1,1), tau[:,1:]-tau[:,:-1]],1))
        # take gradient step
        self.actor_optim.zero_grad()
        policy_loss = -policy_loss.mean(dim=1)
        policy_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.actor_optim.step()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)


        return policy_loss.data.numpy(), qloss.data.numpy(),None

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


