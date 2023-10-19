import numpy as np
from collections import deque
import random
from scipy.optimize import minimize
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


class weight_nn(nn.Module):
    def __init__(self, ):
        super(weight_nn, self).__init__()
        self.fc0 = nn.Linear(11, 6)
        self.fc1 = nn.Linear(6, 3)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        r, s = inputs
        x = self.fc0(s)
        x = self.relu(x)
        w = self.fc1(x)
        w = self.softmax(w)
        traj_gain = r * w  # torch.matmul(r, torch.squeeze(w))
        self.w = w
        return traj_gain

    def get_weights(self, s):
        x = self.fc0(s)
        x = self.relu(x)
        w = self.fc1(x)
        w = self.softmax(w)
        return torch.squeeze(w)


class PWorker():
    def __init__(self, members, num_traj=10):
        self.traj = {}
        self.num_traj = 0
        self.num_traj_store = 0
        self.total_traj = 3

        for m in members:
            self.traj[m] = [[] for _ in range(self.total_traj)]
        self.init = np.array([0.6, 0.3, 0.1])
        self.hist_eva = {}
        for m in members:
            self.hist_eva[m] = []
        self.weight_nn = weight_nn()
        self.optimizer = torch.optim.Adam(self.weight_nn.parameters(), lr=0.004)
        self.debug = {"loss": [], "succ": []}

    def reset(self):
        self.num_traj = 0
        for m in list(self.traj.keys()):
            self.traj[m] = [[] for _ in range(self.total_traj)]
            self.hist_eva[m] = []

    def update(self, member_id, r1, r2, r3, s):
        self.traj[member_id][self.num_traj].append(np.concatenate([np.array([r1, r2, r3]).reshape(3), s.reshape(-1)]))

    def eval(self, eval, member_id, traj_id):
        self.traj[member_id][traj_id]["eval"] = eval

    def get_train_data(self, index1, index2):
        g1 = 0.
        g2 = 0.
        reg = 0.
        for k, traj_of_bus in self.traj.items():
            if len(traj_of_bus[index1]) == 0:
                continue

            traj1_reward = np.array(traj_of_bus[index1]).reshape(-1, 3 + 11)[:, :3]
            traj1_state = np.array(traj_of_bus[index1]).reshape(-1, 3 + 11)[:, 3:]
            traj2_reward = np.array(traj_of_bus[index2]).reshape(-1, 3 + 11)[:, :3]
            traj2_state = np.array(traj_of_bus[index2]).reshape(-1, 3 + 11)[:, 3:]

            traj1_reward = torch.tensor(traj1_reward, dtype=torch.float32)
            traj2_reward = torch.tensor(traj2_reward, dtype=torch.float32)
            traj1_state = torch.tensor(traj1_state, dtype=torch.float32)
            traj2_state = torch.tensor(traj2_state, dtype=torch.float32)

            eval1 = -self.hist_eva[k][index1]
            eval2 = -self.hist_eva[k][index2]

            q1 = self.weight_nn([traj1_reward, traj1_state])
            g1 += torch.sum(q1) / 1000.
            reg += torch.mean((self.weight_nn.w - torch.tensor(self.init)) ** 2)

            q2 = self.weight_nn([traj2_reward, traj2_state])
            g2 += torch.sum(q2) / 1000.
            reg += torch.mean((self.weight_nn.w - torch.tensor(self.init)) ** 2)
        succ = 0
        if eval1 > eval2:
            # g = g2 - g1
            if g1 > g2:
                succ = 1
                print(-torch.exp(g1) / (torch.exp(g1) + torch.exp(g2)))
            g = torch.minimum(-torch.exp(g1) / (torch.exp(g1) + torch.exp(g2)), torch.tensor(-0.3)) + 0. * reg
        else:
            # g = g1 - g2
            if g1 < g2:
                succ = 1

            g = torch.minimum(-torch.exp(g2) / (torch.exp(g1) + torch.exp(g2)), torch.tensor(-0.3)) + 0. * reg
        return g, succ

    def learnw(self, ):
        batch_size = 2
        learn_step = 200
        to_vis = []
        succ_count_vis = []
        for k in range(learn_step):
            loss = 0.
            begin_loss = 1000
            succ_count = 0
            for _ in range(batch_size):
                index_set = [i for i in range(self.total_traj)]
                traj_i = random.sample(index_set, 1)[0]
                index_set.pop(index_set.index(traj_i))
                traj_j = random.sample(index_set, 1)[0]
                g, succ = self.get_train_data(traj_i, traj_j)
                succ_count += succ

                loss += g
            succ_count = succ_count * 1.0 / batch_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_ = loss.detach().numpy()
            to_vis.append(loss_)
            succ_count_vis.append(succ_count)
            print("epoch {}: succ:{}".format(k,  succ_count))

            if abs(begin_loss - loss_) / abs(loss_) * 1.0 < 0.05:
                break
            begin_loss = loss_
            # for name, param in self.weight_nn.named_parameters():
            #     if param.grad!=None:
            #         print(name,param.grad )
        # plt.plot(to_vis)
        # plt.show()
        # plt.plot(succ_count_vis)
        # plt.show()
        #
        # self.debug["loss"] = to_vis
        # self.debug["succ"] = succ_count_vis
        # df = pd.DataFrame(self.debug)
        # df.to_csv("preference_train_log.csv")
        return loss_, None, None

    def updatew(self, line1, line2, ep=0):
        def cal_traj_gain(line, w):
            traj_gain = []
            for i in range(self.total_traj):
                gain = 0
                for bus in line:
                    gain += np.sum(np.matmul(np.array(bus[i]).reshape(-1, 3), w))
                traj_gain.append(gain)
            return np.array(traj_gain).reshape(-1, )

        def coopear_eval(g1, g2):
            return np.corrcoef(g1, g2)[0, 1]

        def obj(x):
            def cal_traj_gain(line, w=x):
                traj_gain = []
                for i in range(self.total_traj):
                    gain = 0
                    for bus in line:
                        gain += np.sum(np.matmul(np.array(bus[i]).reshape(-1, 3), np.array(w).reshape(3, 1)))
                    traj_gain.append(gain)
                return np.array(traj_gain).reshape(-1, )

            g1 = cal_traj_gain(line1)
            g2 = cal_traj_gain(line2)
            return -(np.corrcoef(cal_traj_gain(line1), cal_traj_gain(line2))[0, 1])

        def eq(x):
            return np.sum(x) - 1

        if ep > 100:
            res = minimize(obj, self.w, bounds=[(0.6, 0.6), (0.4, 0.4), (0.2, 1.)], method="Powell")
            self.w = res.x
        g1 = cal_traj_gain(line1, self.w)
        g2 = cal_traj_gain(line2, self.w)

        return coopear_eval(g1, g2), self.w
        import logging


if __name__ == '__main__':
    import torch
    import numpy as np

    np.random.seed(1700)
    line1_bus1_sample1 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line1_bus1_sample2 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line1_bus1_sample3 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]

    line1_bus2_sample1 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line1_bus2_sample2 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line1_bus2_sample3 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]

    line2_bus1_sample1 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line2_bus1_sample2 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line2_bus1_sample3 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]

    line2_bus2_sample1 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line2_bus2_sample2 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]
    line2_bus2_sample3 = [np.random.normal(0., 1., size=3) * 5000 for _ in range(6)]


    def coopear_eval(line1, line2, w):
        traj1 = []
        for s in line1:
            s_arr = np.array(s)
            traj1.append(np.sum(np.matmul(s_arr, w)))
        traj2 = []
        for s in line2:
            s_arr = np.array(s)
            traj2.append(np.sum(np.matmul(s_arr, w)))
        print(traj1)
        print(traj2)
        return np.corrcoef(traj1, traj2)[0, 1]


    w = np.array([0.8, 0.2, 0.5]).reshape(3, 1)
    line1 = [[line1_bus1_sample1, line1_bus1_sample2, line1_bus1_sample3],
             [line1_bus2_sample1, line1_bus2_sample2, line1_bus2_sample3]]
    line2 = [[line2_bus1_sample1, line2_bus1_sample2, line2_bus1_sample3],
             [line2_bus2_sample1, line2_bus2_sample2, line2_bus2_sample3]]


    def cal_traj_gain(line, w):
        traj_gain = []
        for i in range(3):
            gain = 0
            for bus in line:
                x = np.array(bus[i]).reshape(-1, 3)
                a = np.matmul(x, np.array(w).reshape(3, 1))
                gain += np.sum(a)
            traj_gain.append(gain)
        return traj_gain


    def coopear_eval(g1, g2):
        return np.corrcoef(g1, g2)[0, 1]


    def obj(x):
        def cal_traj_gain(line, w=x):
            traj_gain = []
            for i in range(3):
                gain = 0
                for bus in line:
                    gain += np.sum(np.matmul(np.array(bus[i]).reshape(-1, 3), w.reshape(3, 1)))
                traj_gain.append(gain)
            return traj_gain

        return -(np.corrcoef(cal_traj_gain(line1), cal_traj_gain(line2))[0, 1])


    def eq(x):
        return np.sum(x) - 1


    #
    # g1 = [-2023.9431, - 2297.797, - 2297.797]
    # g2 = [-3419.819, - 2331.9534, - 2297.797]
    # c = np.corrcoef(g1, g2)
    # print(coopear_eval(g1, g2))
    g1 = cal_traj_gain(line1, w)
    g2 = cal_traj_gain(line2, w)
    print(coopear_eval(g1, g2))
    # res = minimize(obj, w, constraints={'type': 'eq', 'fun': eq}, bounds=[(0., 1.) for _ in range(w.shape[0])])
    res = minimize(obj, w, constraints={'type': 'eq', 'fun': eq}, bounds=[(0.2, 0.8), (0.2, 0.2), (-0., 1.)],
                   method="Powell")

    g1 = cal_traj_gain(line1, res.x)
    g2 = cal_traj_gain(line2, res.x)
    print(coopear_eval(g1, g2))
    print(res.x)
    import logging

    logging.basicConfig(filename='w.txt', filemode='a', format='%(message)s')
    logging.warning(str(np.array(w).reshape(-1)))
