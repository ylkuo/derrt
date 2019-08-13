from planners.rrtstar import RRTStar
from planners.util import argmax, theta_from_vecs
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

import copy
import numpy as np


class DeRRTStar(RRTStar):
    def __init__(self, state_space, d_steer, n_samples, model,
                 k_nn=30, n_resample=20):
        super(DeRRTStar, self).__init__(state_space, d_steer, n_samples, k_nn=k_nn)
        self.n_resample = n_resample
        self.model = model

    # TODO: replace the steering function to sample from the predicted distribution
    #       instead of selecting the max likelihood one of a few nodes
    def steer(self, nn, rand):
        mu = super(DeRRTStar, self).steer(nn, rand)
        mu.parent = nn
        d = self.ss.dist(nn, mu)
        if d > self.d_steer:
            d = self.d_steer
        if nn.parent is None:
            prior_vec = np.array([1, 0])
        else:
            prior_vec = nn.value[:self.ss.dim] - nn.parent.value[:self.ss.dim]
        mu_new_vec = mu.value[:self.ss.dim] - nn.value[:self.ss.dim]
        mu_move = theta_from_vecs(prior_vec, mu_new_vec)
        thetas = np.linspace(0., 360., num=self.n_resample, endpoint=False)
        norm_probs = []; rnn_probs = []; candidates = []
        for theta in thetas:
            if theta >= 180:
                theta = theta - 360
            theta = theta / 180. * np.pi
            dx = mu_new_vec[0]*np.cos(theta) - mu_new_vec[1]*np.sin(theta)
            dy = mu_new_vec[0]*np.sin(theta) + mu_new_vec[1]*np.cos(theta)
            value = nn.value + np.array([dx, dy])
            value = np.concatenate((value, rand.value[self.ss.dim:]))
            state = self.ss.state_type(self.ss, value)
            if value[0] >= self.ss.components[0][1] or value[1] >= self.ss.components[1][1] or \
                value[0] < self.ss.components[0][0] or value[1] < self.ss.components[1][0]:
                continue
            state.parent = nn; state.update(self.model, do_forward=False)
            norm_probs.append(multivariate_normal.logpdf(theta, mean=0, cov=1))
            rnn_probs.append(state.log_prob)
            candidates.append(state)
        # add the original random sampled point to candidate too
        norm_probs.append(multivariate_normal.logpdf(0, mean=0, cov=1))
        mu.update(self.model, do_forward=False)
        rnn_probs.append(mu.log_prob)
        candidates.append(mu)
        # sample new move
        norm_probs = norm_probs - logsumexp(norm_probs)
        rnn_probs = rnn_probs - logsumexp(rnn_probs)
        sum_probs = [logsumexp([rnn_probs[i], norm_probs[i]], b=[0.5, 0.5]) for i in range(len(norm_probs))]
        sample = sum_probs.index(max(sum_probs))
        candidates[sample].update(self.model)
        return candidates[sample]

    def choose_parent(self, x_nears, x_new, x_min):
        super(DeRRTStar, self).choose_parent(x_nears, x_new, x_min)
        x_new.update(self.model)
        return

    def rewire(self, x_nears, x_new):
        for n in x_nears:
            cost = x_new.cost + self.ss.dist(x_new, n)
            if cost < n.cost and not x_new.check_collision(n):
                n.parent.children = [x for x in n.parent.children if x != n]
                n.parent = x_new
                x_new.children.append(n)
                n.update(self.model)
                n.update_children(self.model)
