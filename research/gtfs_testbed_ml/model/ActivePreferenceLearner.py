import numpy as np
import math
import scipy.stats as st
import random
import matplotlib.pyplot as plt


class PreferenceLearner:
    def __init__(self, members, n_iter=15000, warmup=5000, d=3):

        self.n_iter = n_iter
        self.warmup = warmup
        self.d = d
        self.accept_rates = None
        self.deltas = []
        self.prefs = []

        self.traj = {}
        self.num_traj = 0
        self.num_traj_store = 0
        self.total_traj = 1000
        self.volume_buffer = VolumeBuffer()
        self.w_curr_pos = self.sample_w_prior(10000)
        self.w_curr_mean = self.w_curr_pos.mean(axis=0)
        for m in members:
            self.traj[m] = [[] for _ in range(self.total_traj)]
        self.hist_eva = {}
        for m in members:
            self.hist_eva[m] = []

    def reset(self):
        self.volume_buffer.reset()

    def update(self, member_id, r1, r2, r3):
        # self.traj[member_id][self.num_traj].append(np.concatenate([np.array([r1, r2, r3]).reshape(3), s.reshape(-1)]))
        self.traj[member_id][self.num_traj].append(np.concatenate([np.array([r1, r2, r3]).reshape(3)]))

    # add delta and preference
    def log_preferences(self, delta, pref):
        self.deltas.append(delta)
        self.prefs.append(pref)

    def w_prior(self, w):
        if np.linalg.norm(w) <= 1 and np.all(np.array(w) >= 0):
            return (2 ** self.d) / (math.pi ** (self.d / 2) / math.gamma(self.d / 2 + 1))
        else:
            return 0

    def sample_w_prior(self, n):
        sample = np.random.randn(n, self.d)
        w_out = []
        for w in sample:
            w_out.append(list(np.exp(w) / np.sum(np.exp(w))))
        return np.array(w_out)

    def propose_w(self, w_curr):
        w_new = st.multivariate_normal(mean=w_curr,
                                       cov=0.05).rvs()  # Random Variates sampled from new posterior distribution
        return w_new

    def propose_w_prob(self, w1, w2):
        q = st.multivariate_normal(mean=w1, cov=0.05).pdf(w2)
        return q

    # def propose_w_prob(self, w1, w2):
    #     return st.beta.pdf(x=w1[0], a=w2[0]+0.1, b=1.) * st.beta.pdf(x=w1[1], a=w2[1]+0.1, b=1.) * st.beta.pdf(x=w1[2], a=w2[2]+0.1, b=1.)
    #
    #
    # def propose_w(self, w_curr):
    #     return st.beta(a=np.array(w_curr)+0.1, b=1.).rvs(self.d)



    # def w_prior(self, w):
    #     try:
    #         return st.dirichlet.pdf(x=w, alpha=[1, 1, 1])
    #     except:
    #         return 0
    #
    # def sample_w_prior(self, n):
    #     sample = st.dirichlet(alpha=[1, 1, 1]).rvs(n)
    #     w_out = []
    #     for w in sample:
    #         w_out.append(list(w / np.linalg.norm(w)))
    #     return np.array(w_out)

    # def propose_w_prob(self, w1, w2):
    #     try:
    #         return st.dirichlet.pdf(x=w1, alpha=w2)
    #     except:
    #         return 0
    #
    # def propose_w(self, w_curr):
    #     return st.dirichlet(alpha=w_curr).rvs()

    def f_logliks(self, w, delta, pref):
        return np.log(np.minimum(1, np.exp(pref * np.dot(w, delta)) + 1e-5))

    def posterior_log_prob(self, deltas, prefs, w):
        f_logliks = []
        for i in range(len(prefs)):
            try:
                f_logliks.append(self.f_logliks(w, deltas[i], prefs[i]))
            except:
                pass

        logliks = np.sum(f_logliks)
        log_prior = np.log(self.w_prior(w) + 1e-5)

        return logliks + log_prior

    def update_volume(self,):
        rand_idx = np.random.choice(np.arange(self.num_traj), 2, replace=False)
        index1 = rand_idx[0]
        index2 = rand_idx[1]

        new_returns_1 = np.zeros([self.d, ])
        new_returns_2 = np.zeros([self.d, ])
        for k, traj_of_bus in self.traj.items():
            if len(traj_of_bus[index1]) == 0:
                continue
            traj1_rewards = np.array(traj_of_bus[index1]).reshape(-1, 3)[:, :self.d]
            traj2_rewards = np.array(traj_of_bus[index2]).reshape(-1, 3)[:, :self.d]

            new_returns_1 += np.sum(traj1_rewards, axis=0)
            new_returns_2 += np.sum(traj2_rewards, axis=0)
            if self.hist_eva[k][index1] > self.hist_eva[k][index2]:
                preference = -1.
            else:
                preference = 1.
        # return new_returns_1 - new_returns_2, preference
        if self.compare_delta(w_posterior=self.w_curr_pos, new_returns_a=new_returns_1,
                              new_returns_b=new_returns_2, preference=preference):
            self.volume_buffer.log_statistics([self.hist_eva[k][index1], self.hist_eva[k][index2]])

    # mcmc use propose distribution to sample and approximate the posterior (sicne posterior is hard to sample)
    def mcmc_vanilla(self, w_init="mode"):
        if w_init == "mode":
            w_init = [0. for _ in range(self.d)]

        w_arr = []
        w_curr = np.array(w_init)
        w_curr = np.exp(w_curr) / np.sum(np.exp(w_curr))
        accept_num = 0

        for i in range(1, self.warmup + self.n_iter + 1):
            # sample from current posterior
            w_curr = np.array(w_curr).reshape([self.d])
            w_new = self.propose_w(w_curr=w_curr) # use proposed distribution to sample
            w_new = np.array(w_new).reshape([self.d])

            w_new = np.exp(w_new) / np.sum(np.exp(w_new))

            if np.any(np.isnan(w_new)):
                print(w_arr)
                assert 1 == 0

            pos_prob_curr = self.posterior_log_prob(self.deltas, self.prefs, w_curr) # evaluate current w with posterior distribution
            pos_prob_new = self.posterior_log_prob(self.deltas, self.prefs, w_new) # evaluate newly sampled w with posterior distribution
 
            if pos_prob_new > pos_prob_curr: # determined whether accept new sampled w
                accept_ratio = 1
            else:
                # i.e., g(x|x')/g(x'|x)
                try:
                    qr = self.propose_w_prob(w_curr, w_new) / self.propose_w_prob(w_new, w_curr)
                    accept_ratio = np.exp(pos_prob_new - pos_prob_curr) * qr
                except:
                    accept_ratio = -1.
            accept_prob = min(1., accept_ratio)
            if accept_prob > st.uniform(0., 1.).rvs():
                w_curr = w_new
                accept_num = accept_num + 1
                w_arr.append(w_curr)
            else:
                w_arr.append(w_curr)
        self.accept_rates = accept_num / (self.warmup + self.n_iter + 1)
        self.w_curr_pos = np.array(w_arr)[self.warmup:]
        self.w_curr_mean = np.array(w_arr)[self.warmup:].mean(axis=0)
        return self.w_curr_mean, self.accept_rates

    def volume_removal(self, w_posterior, delta):
        expected_volume_a = 0
        expected_volume_b = 0
        for w in w_posterior:
            expected_volume_a += (1 - self.f_logliks(w=w, delta=delta, pref=1))
            expected_volume_b += (1 - self.f_logliks(w=w, delta=delta, pref=-1))
        return min(expected_volume_a / len(w_posterior), expected_volume_b / len(w_posterior))

    def compare_delta(self, w_posterior, new_returns_a, new_returns_b, preference, random=False):
        delta = new_returns_a - new_returns_b
        volume = self.volume_removal(w_posterior, delta)
        if volume > self.volume_buffer.best_volume or random:
            self.volume_buffer.best_volume = volume
            self.volume_buffer.best_delta = delta
            self.volume_buffer.best_observed_returns = (new_returns_a, new_returns_b)
            self.volume_buffer.preference = preference
            return True
        return False


class VolumeBuffer:
    def __init__(self, auto_pref=True):
        self.auto_pref = auto_pref
        self.best_volume = -np.inf
        self.best_delta = None  # the difference of cumulative reward between two trajactories
        self.preference = None
        self.best_observed_returns = (None, None)
        self.observed_logs = []
        self.objective_logs = []

    def log_statistics(self, statistics):
        self.objective_logs.append(statistics)

    def log_rewards(self, rewards):
        self.observed_logs.append(rewards)

    def reset(self):
        self.best_volume = -np.inf
        self.best_delta = None
        self.best_observed_returns = (None, None)
