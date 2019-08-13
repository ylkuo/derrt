#-*- coding: utf-8 -*-

import hmmlearn.hmm as hmmlearn
import numpy as np

from collections import deque

from scipy.special import logsumexp
from sklearn import cluster
from scipy import linalg

'''HMM implementation in log space.
Implementation adopted from:
    Mann, Tobias P. "Numerically stable hidden Markov model implementation."
    An HMM scaling tutorial (2006).
'''

"""Utility functions"""
def eexp(x):
    if np.isneginf(x):
        return 0
    else:
        return np.exp(x)

def eln(x):
    if x == 0:
        return -np.inf
    elif x > 0:
        return np.log(x)
    else:
        raise ValueError('cannot take log of negative number')

def elnsum(eln_x, eln_y):
    if np.isneginf(eln_x):
        return eln_y
    elif np.isneginf(eln_y):
        return eln_x
    else:
        if eln_x > eln_y:
            return eln_x + np.log1p(eexp(eln_y-eln_x))
        else:
            return eln_y + np.log1p(eexp(eln_x-eln_y))

def normalize(a, axis=None):
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape
    a /= a_sum

def log_normalize(a, axis=None):
    a_lse = logsumexp(a, axis)
    a -= a_lse[:, np.newaxis]

def log_mask_zero(a):
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        a_log = np.log(a)
        return a_log

def log_multivariate_normal_density_diag(obs, means, covars):
    """Compute Gaussian log-density at obs for a diagonal model."""
    n_samples, n_dim = obs.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(obs, (means / covars).T)
                  + np.dot(obs ** 2, (1.0 / covars).T))
    return lpr

def log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob

def distribute_covar_matrix_to_match_covariance_type(
    tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template
    """
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv

class ConvergenceMonitor(object):
    """Convergence monitor"""
    def __init__(self, threshold, n_iter):
        self.threshold = threshold
        self.n_iter = n_iter
        self.history = deque(maxlen=2)
        self.iter = 0

    def report(self, logprob):
        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        return (self.iter == self.n_iter or
                (len(self.history) == 2 and
                 self.history[1] - self.history[0] < self.threshold))


class _BaseHMM(object):
    def __init__(self, n, m, is_left_right=False, precision=np.double):
        # state and feature size of HMM
        self.n = n
        self.m = m
        self.precision = precision
        # model probabilities
        init = 1. / self.n
        if not hasattr(self, "pi"):
            self.pi = np.full(self.n, init)
        if not hasattr(self, "A"):
            self.A = np.full((self.n, self.n), init)
        # log probabilities
        self.logA = None
        self.logpi = None
        # init for left-to-right HMM
        if is_left_right:
            for i in range(n):
                for j in range(n):
                    if i == n - 1 and j == 0:
                        self.A[i][j] = 0.5
                    elif i == n - 1 and j == n - 1:
                        self.A[i][j] = 0.5
                    elif i > j: self.A[i][j] = 0
                    else: self.A[i][j] = 1. / (self.n - i)
            self.pi = np.zeros(self.n, dtype=self.precision)
            self.pi[0] = 1.

    def train(self, all_observations, iterations=10, epsilon=1e-2):
        self.monitor_ = ConvergenceMonitor(epsilon, iterations)
        for i in range(iterations):
            curr_logprob = 0
            stats = self._init_stats()
            self._compute_logprob()
            # run E-step on all training examples
            for observations in all_observations:
                self._mapB(observations)
                stats['logprob'], stats['logalpha'] = self._forward(observations)
                curr_logprob += stats['logprob']
                stats['logbeta'] = self._backward(observations)
                stats['gamma'] = self._compute_gamma(stats['logalpha'], stats['logbeta'])
                self._accumulate_stats(observations, stats)
            # run M-step after updating stats from all training examples
            self._reestimate(stats, observations)
            # record log probability and check if converged
            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break
        # update the final log probabilities after training
        self._compute_logprob()

    def viterbi(self, observations, return_lattice=False):
        n_obs = len(observations)
        self._mapB(observations)
        delta = np.full((n_obs, self.n), -np.inf, dtype=self.precision)
        psi = np.zeros((n_obs, self.n), dtype=self.precision)
        # init, in log space
        for i in range(self.n):
            delta[0][i] = self.logpi[i] + self.logB_map[0][i]
        # induction
        for t in range(1, n_obs):
            for j in range(self.n):
                for i in range(self.n):
                    if delta[t][j] < delta[t-1][i] + self.logA[i][j]:
                        delta[t][j] = delta[t-1][i] + self.logA[i][j]
                        psi[t][j] = i
                delta[t][j] = delta[t][j] + self.logB_map[t][j]
        # termination: find the sequence with max probability
        p_max = -np.inf
        state_seq = np.empty(n_obs, dtype=np.int32)
        for i in range(self.n):
            if p_max < delta[n_obs-1][i]:
                p_max = delta[n_obs-1][i]
                state_seq[n_obs-1] = i
        for i in range(1, n_obs):
            state_seq[n_obs-i-1] = psi[n_obs-i][state_seq[n_obs-i]]
        if return_lattice:
            return p_max, state_seq, delta, (delta - self.logB_map)
        else:
            return p_max, state_seq

    def viterbi_with_prefix(self, prefix_delta, observations):
        n_obs = len(observations)
        self._mapB(observations)
        delta = np.full((n_obs, self.n), -np.inf, dtype=self.precision)
        psi = np.zeros((n_obs, self.n), dtype=self.precision)
        # init from prefix
        for j in range(self.n):
            for i in range(self.n):
                if delta[0][j] < prefix_delta[-1][i] + self.logA[i][j]:
                    psi[0][j] = i
                    delta[0][j] = prefix_delta[-1][i] + self.logA[i][j]
            delta[0][j] = delta[0][j] + self.logB_map[0][j]
        # induction, same as regular viterbi
        for t in range(1, n_obs):
            for j in range(self.n):
                for i in range(self.n):
                    if delta[t][j] < delta[t-1][i] + self.logA[i][j]:
                        delta[t][j] = delta[t-1][i] + self.logA[i][j]
                        psi[t][j] = i
                delta[t][j] = delta[t][j] + self.logB_map[t][j]
        # termination: get max likelihood
        p_max = -np.inf
        state_seq = np.empty(n_obs, dtype=np.int32)
        for i in range(self.n):
            if p_max < delta[n_obs-1][i]:
                p_max = delta[n_obs-1][i]
                state_seq[n_obs-1] = i
        for i in range(1, n_obs):
            state_seq[n_obs-i-1] = psi[n_obs-i][state_seq[n_obs-i]]
        return p_max, state_seq, delta, (delta - self.logB_map)


    def _init_stats(self):
        stats = {'pi': np.zeros(self.n),
                 'A': np.zeros((self.n, self.n))}
        return stats

    def _compute_logprob(self):
        self.logA = log_mask_zero(self.A)
        self.logpi = log_mask_zero(self.pi)

    def _compute_alpha(self, n_obs, logalpha, prefix_logalpha=None):
        work_buffer = np.zeros(self.n, dtype=self.precision)
        # init: alpha_1(x) = pi(x)b_x(O1)
        if prefix_logalpha is None:
            for i in range(self.n):
                logalpha[0][i] = self.logpi[i] + self.logB_map[0][i]
        else:
            for j in range(self.n):
                for i in range(self.n):
                    work_buffer[i] = prefix_logalpha[-1][i] + self.logA[i][j]
                logalpha[0][j] = logsumexp(work_buffer) + self.logB_map[0][j]
        # induction
        for t in range(1, n_obs):
            for j in range(self.n):
                for i in range(self.n):
                    work_buffer[i] = logalpha[t-1][i] + self.logA[i][j]
                logalpha[t][j] = logsumexp(work_buffer) + self.logB_map[t][j]

    def _compute_beta(self, n_obs, logbeta):
        work_buffer = np.zeros(self.n, dtype=self.precision)
        # init: beta_i(T) = 1
        for i in range(self.n):
            logbeta[n_obs-1][i] = 0.0
        # induction
        for t in range(n_obs-2, -1, -1):
            for i in range(self.n):
                for j in range(self.n):
                    work_buffer[j] = self.logA[i][j] + self.logB_map[t+1][j] + logbeta[t+1][j]
                logbeta[t][i] = logsumexp(work_buffer)

    def _forward(self, observations):
        n_obs = len(observations)
        self._mapB(observations)
        # initialize forward lattice
        logalpha = np.zeros((n_obs, self.n), dtype=self.precision)
        # compute alpha
        self._compute_alpha(n_obs, logalpha)
        return logsumexp(logalpha[n_obs-1]), logalpha

    def _forward_with_prefix(self, observations, prefix_logalpha):
        n_obs = len(observations)
        self._mapB(observations)
        logalpha = np.zeros((n_obs, self.n), dtype=self.precision)
        self._compute_alpha(n_obs, logalpha, prefix_logalpha)
        return logsumexp(logalpha[n_obs-1]), logalpha

    def _backward(self, observations):
        n_obs = len(observations)
        # initialize backward lattice
        logbeta = np.zeros((n_obs, self.n), dtype=self.precision)
        # compute beta
        self._compute_beta(n_obs, logbeta)
        return logbeta

    def _compute_gamma(self, logalpha, logbeta):
        gamma = logalpha + logbeta
        log_normalize(gamma, axis=1)
        gamma_exp = np.exp(gamma)
        return gamma_exp

    def _accumulate_stats(self, observations, stats):
        stats['pi'] += stats['gamma'][0]
        logxi = np.full((self.n, self.n), -np.inf, dtype=self.precision)
        self._compute_xi(observations, stats['logalpha'], stats['logbeta'], logxi)
        stats['A'] += np.exp(logxi)

    def _compute_xi(self, observations, logalpha, logbeta, logxi):
        normalizer = logsumexp(logalpha[len(observations)-1])
        for t in range(len(observations)-1):
            for i in range(self.n):
                for j in range(self.n):
                    work_buffer = (logalpha[t][i]
                                   + self.logA[i][j]
                                   + self.logB_map[t+1][j]
                                   + logbeta[t+1][j]
                                   - normalizer)
                    logxi[i][j] = elnsum(logxi[i][j], work_buffer)

    def _reestimate(self, stats, observations):
        self.pi = np.where(self.pi == 0.0, self.pi, stats['pi'])
        normalize(self.pi)
        self.A = np.where(self.A == 0.0, self.A, stats['A'])
        normalize(self.A, axis=1)

    def _mapB(self, observations):
        pass


class HMM(_BaseHMM):
    """Discrete HMM
    n: size of hidden states
    m: size of observed symbols
    A: trainsition probability
    B: emission probability
    pi: start probability
    """
    def __init__(self, n, m, A=None, B=None, pi=None,
                 random_state=None, precision=np.double, uniform_init=False,
                 is_left_right=False):
        _BaseHMM.__init__(self, n, m, is_left_right, precision)
        init_val = 1. / self.n
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0)
        if A is not None:
            self.A = A
        else:
            self.A = np.full((self.n, self.n), init_val, dtype=self.precision)
        if B is not None:
            self.B = B
        else:
            if uniform_init:
                self.B = np.full((self.n, self.m), 1. / self.m, dtype=self.precision)
            else:
                self.B = self.random_state.rand(self.n, self.m)
                normalize(self.B, axis=1)
        if pi is not None:
            self.pi = pi
        else:
            self.pi = np.full(self.n, init_val, dtype=self.precision)

    def _init_stats(self):
        stats = _BaseHMM._init_stats(self)
        stats['obs'] = np.zeros((self.n, self.m))
        return stats

    def _accumulate_stats(self, observations, stats):
        _BaseHMM._accumulate_stats(self, observations, stats)
        for t, symbol in enumerate(observations):
            stats['obs'][:, symbol] += stats['gamma'][t]
    
    def _reestimate(self, stats, observations):
        _BaseHMM._reestimate(self, stats, observations)
        self.B = (stats['obs'] / stats['obs'].sum(axis=1)[:, np.newaxis])

    def _compute_logprob(self):
        _BaseHMM._compute_logprob(self)
        self.logB = log_mask_zero(self.B)

    def _mapB(self, observations):
        self.logB_map = self.logB[:, observations].T

class GaussianHMM(_BaseHMM):
    """Gaussian HMM
    n: size of hidden states
    m: number of features
    """
    def __init__(self, n, m, is_left_right=False, random_state=None,
                 covariance_type='diag', min_covar=1e-3, covars_prior=1e-2,
                 covars_weight=1, uniform_init=False, precision=np.double):
        _BaseHMM.__init__(self, n, m, is_left_right, precision)
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0)
        self.uniform_init = uniform_init
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.min_covar = min_covar
        self.covariance_type = covariance_type
 
    def train(self, all_observations, iterations=10, epsilon=1e-2):
        # pack observations together
        features = []
        for obs in all_observations: features.extend(obs)
        features = np.asarray(features)
        kmeans = cluster.KMeans(n_clusters=self.n,
                                random_state=self.random_state)
        kmeans.fit(features)
        self.means = kmeans.cluster_centers_
        cv = np.cov(features.T) + self.min_covar * np.eye(features.shape[1])
        if not cv.shape:
            cv.shape = (1, 1)
        self.covars = distribute_covar_matrix_to_match_covariance_type(
            cv, self.covariance_type, self.n).copy()
        _BaseHMM.train(self, all_observations, iterations, epsilon)

    def _init_stats(self):
        stats = _BaseHMM._init_stats(self)
        stats['posterior'] = np.zeros(self.n)
        stats['obs'] = np.zeros((self.n, self.m))
        stats['obs**2'] = np.zeros((self.n, self.m))
        stats['obs*obs.T'] = np.zeros((self.n, self.m, self.m))
        return stats

    def _accumulate_stats(self, observations, stats):
        observations = np.asarray(observations)
        _BaseHMM._accumulate_stats(self, observations, stats)
        stats['posterior'] += stats['gamma'].sum(axis=0)
        stats['obs'] += np.dot(stats['gamma'].T, observations)
        if self.covariance_type in ('spherical', 'diag'):
            stats['obs**2'] += np.dot(stats['gamma'].T, observations ** 2)
        elif self.covariance_type in ('tied', 'full'):
            # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
            # -> (nc, nf, nf)
            stats['obs*obs.T'] += np.einsum(
                'ij,ik,il->jkl', stats['gamma'], observations, observations)

    def _reestimate(self, stats, observations):
        observations = np.asarray(observations)
        _BaseHMM._reestimate(self, stats, observations)
        denom = stats['posterior'][:, np.newaxis]
        self.means = stats['obs'] / denom
        if self.covariance_type in ('spherical', 'diag'):
            cv_num = (stats['obs**2']
                      - 2 * self.means * stats['obs']
                      + self.means**2 * denom)
            cv_den = max(self.covars_weight - 1, 0) + denom
            self.covars = \
                (self.covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
            if self.covariance_type == 'spherical':
                self.covars = np.tile(
                    self.covars.mean(1)[:, np.newaxis],
                    (1, self.covars.shape[1]))
        elif self.covariance_type in ('tied', 'full'):
            cv_num = np.empty((self.n, self.m, self.m))
            for c in range(self.n):
                obsmean = np.outer(stats['obs'][c], self.means[c])
                cv_num[c] = (stats['obs*obs.T'][c]
                             - obsmean - obsmean.T
                             + np.outer(self.means[c], self.means[c])
                             * stats['posterior'][c])
            cvweight = max(self.covars_weight - self.m, 0)
            if self.covariance_type == 'tied':
                self.covars = ((self.covars_prior + cv_num.sum(axis=0)) /
                                 (cvweight + stats['posterior'].sum()))
            elif self.covariance_type == 'full':
                self.covars = ((self.covars_prior + cv_num) /
                               (cvweight + stats['posterior'][:, None, None]))

    def _mapB(self, observations):
        observations = np.asarray(observations)
        if self.uniform_init:
            self.logB_map = np.full((len(observations), self.n),
                                    np.log(1. / self.m), dtype=self.precision)
        else:
            # TODO: add other covariance types
            if self.covariance_type == 'diag':
                self.logB_map = log_multivariate_normal_density_diag(
                    np.asarray(observations), self.means, self.covars)
            elif self.covariance_type == 'full':
                self.logB_map = log_multivariate_normal_density_full(
                    np.asarray(observations), self.means, self.covars)


if __name__ == '__main__':
    obs = (3,1,2,1,0,1,2,3,1,2,0,0,0,1,1,2,1,3,0)
 
    # test HMM implementation, should output the same state sequence as hmmlearn
    rndst = np.random.RandomState(0)
    hmm = HMM(n=4, m=4, random_state=rndst, precision=np.longdouble)
    hmm.train([obs*10], iterations=20)
    print("pi: ", hmm.pi)
    print("A: ", hmm.A)
    print("B: ", hmm.B)
    print(hmm.viterbi([3,1,2,0,1,2,3]))
    print("--------------------")
    # test hmmlearn
    rndst = np.random.RandomState(0)
    hmm = hmmlearn.MultinomialHMM(n_components=4, n_iter=20, random_state=rndst)
    hmm.fit(np.reshape(obs*10, (-1, 1)), [len(obs)*10])
    print("pi: ", hmm.startprob_)
    print("A: ", hmm.transmat_)
    print("B: ", hmm.emissionprob_)
    print(hmm.decode(np.reshape([3,1,2,0,1,2,3], (-1, 1)), [7]))
    print("--------------------")

    # test Gaussian HMM
    obs = np.asarray([
        [7.15000000e+02, 5.85000000e+02, 0.00000000e+00, 0.00000000e+00],
        [7.15000000e+02, 5.20000000e+02, 1.04705811e+00, -6.03696289e+01],
        [7.15000000e+02, 4.55000000e+02, 7.20886230e-01, -5.27055664e+01],
        [7.15000000e+02, 3.90000000e+02, -4.57946777e-01, -7.80605469e+01],
        [7.15000000e+02, 3.25000000e+02, -6.43127441e+00, -5.59954834e+01],
        [7.15000000e+02, 2.60000000e+02, -2.90063477e+00, -7.80220947e+01],
        [7.15000000e+02, 1.95000000e+02, 8.45532227e+00, -7.03294373e+01],
        [7.15000000e+02, 1.30000000e+02, 4.09387207e+00, -5.83621216e+01],
        [7.15000000e+02, 6.50000000e+01, -1.21667480e+00, -4.48131409e+01]
    ])
    rndst = np.random.RandomState(0)
    hmm = GaussianHMM(3, 4, covariance_type='diag', random_state=rndst)
    hmm.train([obs], iterations=10)
    print("pi: ", hmm.pi)
    print("A: ", hmm.A)
    print("means: ", hmm.means)
    print("covariance: ", hmm.covars)
    print("--------------------")
    # test hmmlearn Gaussian HMM
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        rndst = np.random.RandomState(0)
        hmm = hmmlearn.GaussianHMM(3, random_state=rndst)
        hmm.fit(obs)
        print("pi: ", hmm.startprob_)
        print("A: ", hmm.transmat_)
        print("means: ", hmm.means_)
        print("covariance: ", hmm._covars_)
