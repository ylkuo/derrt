from scipy.special import gamma as gamma_fn
from sklearn.neighbors import KDTree
from planners.state import State
from planners.util import argmin

import matplotlib.pyplot as plt
import numpy as np


class RRTStar(object):
    def __init__(self, state_space, d_steer, n_samples, k_nn=30):
        self.ss = state_space
        self.d_steer = d_steer
        self.n_samples = n_samples
        self.k_nn = k_nn
        self.kd_tree = None

        self.states = []
        self.found_goals = []

    def sample_free(self):
        while True:
            x_rand = self.ss.sample()
            if self.states[0].env.is_obstacle_free(x_rand):
                return x_rand

    def find_radius(self):
        n = len(self.states)
        unit_ball_volume = np.sqrt(np.pi)**self.ss.dim / gamma_fn(self.ss.dim / 2.0 + 1.0)
        x_free_volume = self.ss.volume
        gamma = (2*(1.0 + 1.0/self.ss.dim) * \
            (x_free_volume/unit_ball_volume))**(1.0/self.ss.dim)
        ball_radius = min(gamma * ((np.log(n+1) / (n+1))**(1.0/self.ss.dim)),
                          self.d_steer)
        return ball_radius

    def steer(self, nn, rand):
        dist = self.ss.dist(nn, rand)
        if dist > self.d_steer:
            ratio = self.d_steer / dist
            # only interpolate the x,y coordinate
            vec = rand.value[:self.ss.dim] - nn.value[:self.ss.dim]
            val = nn.value[:self.ss.dim] + vec*ratio
            val = np.concatenate((val, rand.value[self.ss.dim:]))
            return self.ss.state_type(self.ss, val)
        return rand

    def near(self, s):
        radius = self.find_radius()
        ind = self.kd_tree.query_radius(np.array([s.value]), r=radius)
        nn = [state for i, state in enumerate(self.states) if i in ind[0]]
        return nn

    def choose_parent(self, x_nears, x_new, x_min):
        cost_min = x_min.cost + self.ss.dist(x_min, x_new)
        for n in x_nears:
            cost = n.cost + self.ss.dist(n, x_new)
            if cost < cost_min and not n.check_collision(x_new):
                x_min = n
                cost_min = cost
        x_new.cost = cost_min
        x_new.parent = x_min
        x_min.children.append(x_new)

    def rewire(self, x_nears, x_new):
        for n in x_nears:
            cost = x_new.cost + self.ss.dist(x_new, n)
            if cost < n.cost and not x_new.check_collision(n):
                n.parent.children = [x for x in n.parent.children if x != n]
                n.parent = x_new
                x_new.children.append(n)
                n.update()
                n.update_children()

    def get_solution_path(self):
        if len(self.found_goals) == 0:
            return []
        goal = argmin(lambda n: n.cost, self.found_goals)
        end_node = goal
        final_path = [end_node]
        while end_node.parent is not None:
            final_path.insert(0, end_node.parent)
            end_node = end_node.parent
        return final_path

    def rrt_iter(self):
        # build kd tree for nearest neighbor queries
        self.kd_tree = KDTree(np.array([state.value for state in self.states]))
        x_rand = self.sample_free()
        dist, ind = self.kd_tree.query(np.array([x_rand.value]), k=1)
        x_nearest = self.states[ind[0][0]]
        x_new = self.steer(x_nearest, x_rand)
        if not x_nearest.check_collision(x_new):
            # extend
            x_nears = self.near(x_new)
            self.choose_parent(x_nears, x_new, x_nearest)
            self.states.append(x_new)
            # rewire
            x_nears = set([x for x in x_nears if x != x_new.parent])
            self.rewire(x_nears, x_new)

    def solve(self, start, goal_fn=None):
        if start is None:
            raise ValueError('Need to set start nodes first')
            return []
        self.states = [start]
        for i in range(self.n_samples):
            print('Iter', i)
            self.rrt_iter()
            if goal_fn is not None and goal_fn(self.states[-1]):
                self.found_goals.append(self.states[-1])
        solution = self.get_solution_path()
        return solution

    def clear(self):
        self.states = []
        self.found_goals = []

    def draw(self, draw_path=True, draw_tree=True, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.axes(xlim=(self.ss.components[0][0],self.ss.components[0][1]),
                          ylim=(self.ss.components[1][0],self.ss.components[1][1]))
            ax.set_aspect('equal')
        if draw_tree:
            leaves = []
            for node in self.states:
                if node.parent is None:
                    leaves.append(node)
            while len(leaves) > 0:
                node =  leaves.pop(0)
                for child in node.children:
                    leaves.append(child)
                    xs = np.asarray([child.value[0], child.parent.value[0]])
                    ys = np.asarray([child.value[1], child.parent.value[1]])
                    plt.plot(xs, ys, 'k-', lw=1.)
        if draw_path:
            path = self.get_solution_path()
            for i in range(len(path)-1):
                xs = np.asarray([path[i].value[0], path[i+1].value[0]])
                ys = np.asarray([path[i].value[1], path[i+1].value[1]])
                plt.plot(xs, ys, 'r-', lw=2.)
        return ax
