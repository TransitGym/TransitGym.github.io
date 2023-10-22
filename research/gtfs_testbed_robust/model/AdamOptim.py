import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import numpy as np
import torch
import torch
import torch.nn as nn
import math


class AdamOptim():
    def __init__(self, eta=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, b, dw, db):
        if t == 1:
            self.m_dw = torch.zeros_like(torch.tensor(w), memory_format=torch.preserve_format, dtype=torch.float)
            self.v_dw = torch.zeros_like(torch.tensor(w), memory_format=torch.preserve_format, dtype=torch.float)
            self.m_db = torch.zeros_like(torch.tensor(b), memory_format=torch.preserve_format, dtype=torch.float)
            self.v_db = torch.zeros_like(torch.tensor(b), memory_format=torch.preserve_format, dtype=torch.float)
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw.detach() + (1-self.beta1)*dw
        # self.m_dw.mul_(self.beta1).add_(dw, alpha=1 - self.beta1)
        # *** biases *** #
        self.m_db = self.beta1*self.m_db.detach() + (1-self.beta1)*db
        # self.m_db.mul_(self.beta1).add_(db, alpha=1 - self.beta1)

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw.detach() + (1-self.beta2)*(dw**2)
        # self.v_dw.mul_(self.beta2).addcmul_(dw, dw.conj(), value=1 - self.beta2)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db.detach() + (1-self.beta2)*(db**2)
        # self.v_db.mul_(self.beta2).addcmul_(db, db.conj(), value=1 - self.beta2)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw.sqrt() / math.sqrt(1 - self.beta2 ** t)
        v_db_corr = self.v_db.sqrt() / math.sqrt(1 - self.beta2 ** t)

        ## update weights and biases
        return w.detach() - self.eta * (m_dw_corr / (v_dw_corr + self.epsilon)), b.detach() - self.eta * (m_db_corr / (v_db_corr + self.epsilon))