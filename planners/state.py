from enum import Enum

import numpy as np
import random
import torch


class StateSpace(object):
    def __init__(self, components=None, state_type=None):
        self.components = components  # each component is (lower, upper)
        self.state_type = State if state_type is None else state_type

    @property
    def dim(self):
        return len(self.components)

    @property
    def volume(self):
        m = 1.0
        for i in range(self.dim):
            m *= self.components[i][1] - self.components[i][0]
        return m

    def sample(self):
        value = []
        for c in self.components:
            value.append(random.uniform(c[0], c[1]))
        return self.state_type(self, np.asarray(value))

    def dist(self, s1, s2):
        return np.linalg.norm(s2.value-s1.value)

    def __str__(self):
        pass
    __repr__ = __str__


class State(object):
    def __init__(self, state_space, value, parent=None):
        self.ss = state_space
        self.value = np.asarray(value)
        for d in range(self.ss.dim):
            self.value[d] = np.clip(self.value[d], self.ss.components[d][0], self.ss.components[d][1])
        if parent is not None:
            self._parent = parent
            self.cost = parent.cost + ss.dist(self.parent, self)
        else:
            self._parent = None
            self.cost = 0
        self.children = []
        # for model
        self.log_prob = -np.inf
        self.hidden_state = None
        self.proposal = None
        self.sample = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    def check_collision(self, new_state):
        pass

    def update(self, model=None):
        if self.parent is not None:
            self.cost = self.parent.cost + self.ss.dist(self.parent, self)

    def update_children(self, model=None):
        if model is None:
            for n in self.children:
                n.cost = self.cost + self.ss.dist(self, n)
                n.update_children()
        else:
            for n in self.children:
                n.cost = self.cost + self.ss.dist(self, n)
                n.update(model)
                n.update_children(model)
