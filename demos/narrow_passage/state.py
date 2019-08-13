from planners.nn import Sample
from planners.state import *
from planners.util import theta_from_vecs
from math import sqrt
from torch import FloatTensor as Tensor
from torch.distributions import Normal

import copy
import numpy as np
import pygame
import torch

def prior_dist():
    return [Normal(0, 1)]

class PassageStateSpace(StateSpace):
    def __init__(self, width, height):
        components = [(-width/2.,width/2.), (-height/2.,height/2.)]
        super(PassageStateSpace, self).__init__(components=components, state_type=PassageState)

    @property
    def dim(self):
        return 2

class PassageState(State):
    def __init__(self, state_space, value, parent=None, env=None):
        super(PassageState, self).__init__(state_space, value, parent=None)
        self.env = env

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p
        self.update_env()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'env': continue  # don't copy env
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def extract_feature(self):
        walls = []
        for b in self.env.world.bodies:
            if b.userData['name'] == 'obs_wall':
                walls.append(b)
        top_y = walls[0].position[1] - walls[0].userData['h']/2.
        bot_y = walls[1].position[1] + walls[1].userData['h']/2.
        left_x = walls[0].position[0] - walls[0].userData['w']/2.
        right_x = walls[0].position[0] + walls[0].userData['w']/2.
        landmarks = [PassageState(self.ss, [left_x,top_y]),
                     PassageState(self.ss, [right_x, top_y]),
                     PassageState(self.ss, [left_x, bot_y]),
                     PassageState(self.ss, [right_x, bot_y])]
        normalizer = sqrt(self.env.width**2 + self.env.height**2)
        feature = [float(self.ss.dist(self, landmark) / normalizer) for landmark in landmarks]
        for landmark in landmarks:
            feature.append(float((self.value[0]-landmark.value[0]) / self.env.width))
            feature.append(float((self.value[1]-landmark.value[1]) / self.env.height))
        feature = np.asarray(feature) * 10.  # make it more stable
        return feature

    def init_model_params(self, model):
        if self.hidden_state is None:
            self.hidden_state = None
            self.proposal = None        

    def update(self, model=None, do_forward=True):
        super(PassageState, self).update()
        if model is None: return
        # init variables based on number of models
        self.init_model_params(model)
        if do_forward or self.parent.proposal is None:
            feature = self.extract_feature()
        if self.parent is None:
            rel_pos = np.zeros(self.ss.dim)
        else:
            # update root
            if self.parent.proposal is None:
                rel_pos = np.zeros(self.ss.dim)
                feature = self.parent.extract_feature()
                self.parent.sample = Sample(Tensor(feature), prior_dist(), pos=rel_pos)
                self.parent.init_model_params(model)
                self.parent.proposal, self.parent.hidden_state = \
                    model.forward(self.parent.sample, prev_hidden_state=self.parent.hidden_state)
            # update else
            prior_vec = [1, 0]
            if self.parent.parent is not None:
                prior_vec = np.asarray([self.parent.value[0]-self.parent.parent.value[0],
                                        self.parent.value[1]-self.parent.parent.value[1]])
            new_vec = np.asarray([self.value[0]-self.parent.value[0],
                                  self.value[1]-self.parent.value[1]])
            move = theta_from_vecs(prior_vec, new_vec)
            rel_pos = self.value - self.parent.value
            val = Tensor([move])
            if model.on_cuda: val = val.cuda()
            self.log_prob = self.parent.proposal[0].log_prob(val).item()
        if do_forward:
            self.sample = Sample(Tensor(feature), prior_dist(), pos=rel_pos)
            hidden = None if self.parent is None else self.parent.hidden_state
            self.proposal, self.hidden_state = \
                 model.forward(self.sample, prev_hidden_state=hidden)

    def check_collision(self, new_state):
        pos = new_state.value
        return self.env.check_collision(pos, 0)

    def update_env(self):
        pos = self.value
        _, self.env = self.parent.env.check_collision(pos, 0, return_map=True)

    def get_img(self):
        self.env.draw()
        screen = self.env.gui_world.screen
        img_str = pygame.image.tostring(screen, 'RGB')
        img = np.array(Image.frombytes('RGB', screen.get_size(), img_str).transpose(Image.FLIP_TOP_BOTTOM))
        return img

    def save_img(self, filename):
        self.env.draw()
        pygame.image.save(self.env.gui_world.screen, filename)
