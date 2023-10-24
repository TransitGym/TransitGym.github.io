import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
import os
from torch.nn import utils as nn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as f
from model import layers
import scipy.sparse as sp
import copy


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def prepare_eg(fp, state_dim):
    adjs = []
    u_features = []
    d_features = []
    u_adjs = []
    d_adjs = []
    for i in range(len(fp)):
        fp_ = fp[i][(fp[i][:, -3] <= 0)]

        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            shape=(edges.shape[0], edges.shape[0]))
        # Do not consider ego event in marginal contribution
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = normalize_adj(adj )
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj)  # no direction

        u_adjs.append(adj)
        u_features.append(fp_[:, :state_dim + 1 + 2])

        fp_ = fp[i][(fp[i][:, -3] >= 0)]
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            shape=(edges.shape[0], edges.shape[0]))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = normalize_adj(adj )
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj)  # no direction
        d_adjs.append(adj)
        d_features.append(fp_[:, :state_dim + 1 + 2])

    return u_adjs, d_adjs, u_features, d_features


def stop_embedding(self, bus, stop, d_model):
    position = np.arange(0, len(bus.stop_list))[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    v1 = np.sin(position * div_term)[bus.stop_list.index(stop.id), :]
    v2 = np.cos(position * div_term)[bus.stop_list.index(stop.id), :]
    return np.concatenate([v1, v2])


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1, is_stop_embedding=False):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        self.is_stop_embedding = is_stop_embedding

    def forward(self, s):
        try:
            s, stop_embedding = s
        except:
            pass
        if self.is_stop_embedding == 1:
            x = self.elu(self.linear1(s))  # +stop_embedding
        else:
            x = self.elu(self.linear1(s))
        if self.is_stop_embedding == 1:
            x = self.elu(self.linear2(x))
            x = self.elu(self.linear3(x))
            x = self.elu(self.linear4(x))
        else:
            x = self.elu(self.linear2(x))
            x = self.elu(self.linear3(x))
            x = self.elu(self.linear4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, n_stops=22, action_dim=1, is_stop_embedding=False):

        super(Critic, self).__init__()
        self.hidden1 = 400
        self.state_dim = state_dim

        self.is_stop_embedding = is_stop_embedding

        # for ego critic
        self.fc0 = nn.Linear(state_dim + 1, self.hidden1)
        self.fc1 = nn.Linear(self.hidden1, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, 1)
        self.fc3 = nn.Linear(self.hidden1, 1)

        self.u_attentions = [
            layers.GraphAttentionLayer(state_dim + 1 + 2, self.hidden1, dropout=False, alpha=0.2, concat=True) for _ in
            range(1)]
        for i, attention in enumerate(self.u_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.u_out_att = layers.GraphAttentionLayer(self.hidden1 * 1, self.hidden1, dropout=False, alpha=0.2,
                                                    concat=False)

        self.d_attentions = [
            layers.GraphAttentionLayer(state_dim + 1 + 2, self.hidden1, dropout=False, alpha=0.2, concat=True) for _ in
            range(1)]
        for i, attention in enumerate(self.d_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.d_out_att = layers.GraphAttentionLayer(self.hidden1 * 1, self.hidden1, dropout=False, alpha=0.2,
                                                    concat=False)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.n_stops = n_stops

    def d_egat(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.d_attentions], dim=1)
        x = self.d_out_att(x, adj)
        x = torch.sum(x, 0)
        return x

    def u_egat(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.u_attentions], dim=1)
        x = self.u_out_att(x, adj)
        x = torch.sum(x, 0)
        return x

    def event_critic(self, fp):
        u_adjs, d_adjs, u_features, d_features = prepare_eg(fp, self.state_dim)
        a = []
        reg = []
        for i in range(len(u_adjs)):
            u_x = u_features[i]
            u_adj = u_adjs[i]
            d_x = d_features[i]
            d_adj = d_adjs[i]

            if u_adj.size(0) >= 2:
                u_x = self.u_egat(u_x, u_adj)
            else:
                u_x = self.u_egat(u_x, u_adj)
                reg.append(torch.square(u_x))
                u_x = torch.zeros_like(u_x)

            if d_adj.size(0) >= 2:
                d_x = self.d_egat(d_x, d_adj)
            else:
                d_x = self.d_egat(d_x, d_adj)
                reg.append(torch.square(d_x))
                d_x = torch.zeros_like(d_x)
            u_x = u_x.view(-1, self.hidden1)
            d_x = d_x.view(-1, self.hidden1)
            a.append(self.fc3(u_x + d_x))

        a = torch.stack(a, 0).view(-1, 1)
        if len(reg) > 0:
            reg = torch.stack(reg, 0).view(-1, 1)
        else:
            reg = torch.zeros(1)
        return a, reg

    def ego_critic(self, ego):
        out1 = self.fc0(ego)
        out1 = self.relu(out1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        Q = self.fc2(out1)
        return Q

    def ego_critic_with_embed(self, ego, embed):
        out1 = self.fc0(ego)
        out1 = self.relu(out1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        Q = self.fc2(out1)
        return Q

    def forward(self, xs):
        x, a, fp, embed = xs
        ego = torch.cat([x, a], 1)
        if self.is_stop_embedding == 0:
            Q = self.ego_critic(ego)
        else:
            Q = self.ego_critic_with_embed(ego, embed)
        A, reg = self.event_critic(fp)
        G = Q + A
        return Q, A, G.view(-1, 1), reg


class Agent():
    def __init__(self, state_dim, name, seed=123, n_stops=22, buslist=None, is_stop_embedding=False):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0
        self.critic = Critic(state_dim, n_stops=n_stops, action_dim=1, is_stop_embedding=is_stop_embedding)
        self.critic_target = Critic(state_dim, n_stops=n_stops, action_dim=1, is_stop_embedding=is_stop_embedding)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim, is_stop_embedding=is_stop_embedding)
        self.actor_target = Actor(self.state_dim, is_stop_embedding=is_stop_embedding)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.is_stop_embedding = is_stop_embedding

    def choose_action(self, state):
        if self.is_stop_embedding == 1:
            s, stop_embed = state
            state = [torch.tensor(s, dtype=torch.float).unsqueeze(0),
                     torch.tensor(stop_embed, dtype=torch.float).unsqueeze(0)]
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        a = self.actor(state).squeeze(0).detach().numpy()
        return a

    def learn(self, memories, batch=16, bus_id=None):
        if len(memories) < 1000:
            return 0, 0

        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp, batch_embed, batch_next_embed = [], [], [], [], [], [], [], []
        memory = random.sample(memories, batch)

        batch_mask = []
        batch_mask_n = []
        batch_fp_critic_t = []
        batch_actor_a = []
        for s, fp, a, r, ns, nfp, embed, next_embed in memory:
            batch_s.append(s)
            fp_ = copy.deepcopy(fp)
            fp_ = torch.tensor(fp_, dtype=torch.float32)
            if self.is_stop_embedding == 1:
                fp_[0, self.state_dim + 1] = self.actor(
                    [torch.tensor(s, dtype=torch.float32), torch.tensor(embed, dtype=torch.float32)]).detach()
            else:
                fp_[0, self.state_dim + 1] = self.actor(torch.tensor(s, dtype=torch.float32)).detach()
            batch_fp_critic_t.append(fp_)
            batch_actor_a.append(
                self.actor([torch.tensor(s, dtype=torch.float32), torch.tensor(embed, dtype=torch.float32)]))
            batch_fp.append(torch.FloatTensor(fp))

            batch_a.append(a)

            w = np.array(s[-3:])
            r = np.dot(w, np.array(r))

            batch_r.append(r)
            batch_ns.append(ns)
            batch_embed.append(embed)
            batch_next_embed.append(next_embed)
            batch_nfp.append(torch.FloatTensor(nfp))
        b_fp_pad = batch_fp
        b_nfp_pad = batch_nfp

        batch_actor_a = torch.stack(batch_actor_a, 0)

        b_s = torch.tensor(batch_s, dtype=torch.float)
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)
        b_d = torch.tensor(batch_embed, dtype=torch.float)
        b_nd = torch.tensor(batch_next_embed, dtype=torch.float)
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)
        b_ns = torch.tensor(batch_ns, dtype=torch.float)

        # b_m = torch.tensor(batch_mask, dtype=torch.float).view(-1, 1)
        # b_nm = torch.tensor(batch_mask_n, dtype=torch.float).view(-1, 1)

        # if self.is_stop_embedding==1:
        b_s_comb = [b_s, b_d]
        b_ns_comb = [b_s, b_nd]

        def critic_learn():

            Q, A, G, reg = self.critic([b_s, b_a, b_fp_pad, b_d])
            Q_, A_, G_, _ = self.critic_target(
                [b_ns, self.actor_target(b_ns_comb).detach(), b_nfp_pad,
                 b_nd])  # detach from graph, don't backpropagate

            q_target = b_r + self.gamma * (G_.detach()).view(-1, 1)
            loss_fn = nn.MSELoss()

            qloss = loss_fn(G, q_target) + 0.1 * reg.mean()

            self.critic_optim.zero_grad()
            qloss.backward()
            self.critic_optim.step()

            # print('Q:',Q.mean().item(),'A:',A.mean().item())
            return qloss.item()

        def actor_learn():

            policy_loss, _, _, _ = self.critic([b_s, batch_actor_a, batch_fp_critic_t, b_s_comb])
            loss_fn = nn.MSELoss()
            policy_loss = -torch.mean(policy_loss)
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            return policy_loss.item()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        qloss = critic_learn()
        policy_loss = actor_learn()

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)
        self.learn_step_counter += 1

        return policy_loss, qloss

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))

        path = abspath + "/save/" + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

    def load(self, model):

        abspath = os.path.abspath(os.path.dirname(__file__))
        # if self.is_lstm == True:
        #     model = str(model) + 'Lstm'
        print('Load: ' + abspath + "/save/" + str(model))
        path = abspath + "/save/" + str(model) + str(self.seed) + "_actor.pth"
        state_dict = torch.load(path)
        self.actor.load_state_dict(state_dict)

        # path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        # state_dict = torch.load(path)
        # self.critic.load_state_dict(state_dict)

        # path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        # state_dict = torch.load(path)
        # self.critic.load_state_dict(state_dict)