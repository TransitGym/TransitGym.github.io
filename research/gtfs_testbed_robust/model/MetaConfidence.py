import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal
from torch.optim import lr_scheduler
import os
from torch.nn import utils as nn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM
import torch.nn.functional as f
from model import layers
import copy
import scipy.sparse as sp
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def prepare_eg(fp ):
    adjs = []
    u_features = []
    d_features = []
    u_adjs = []
    d_adjs = []
    for i in range(len(fp)):
        fp_ = fp[i][(fp[i][:, -3] <=0)]

        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0,:] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges==1)[0], np.where(edges==1)[1])), shape=(edges.shape[0], edges.shape[0]))
        # Do not consider ego event in marginal contribution
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = normalize_adj(adj )
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj) # no direction

        u_adjs.append(adj)
        u_features.append(fp_[ :,:3+1+1+2])

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
        d_features.append(fp_[:, :3+1+1+2])

    return u_adjs,d_adjs,u_features,d_features

class MC(nn.Module):
    def __init__(self, state_dim,tau_num, n_stops=22, action_dim=1, seed=1 ):

        super(MC, self).__init__()
        self.hidden1 = 64
        self.fc0 = nn.Linear(state_dim + 1, self.hidden1)
        self.fc1 = nn.Linear(self.hidden1, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, tau_num)


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
        self.softmax = nn.Softmax()

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

    def meta_network(self,fp):
        u_adjs, d_adjs, u_features, d_features = prepare_eg(fp)
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
            a.append(self.relu(self.fc2(u_x + d_x)))

        a = torch.stack(a, 0).view(len(u_adjs), -1)

        return a

    def forward(self, fp):
        W = self.meta_network(fp)
        W_sorted, indices = torch.sort(W, dim=1, descending=False)

        return W.clamp(0.,10.)