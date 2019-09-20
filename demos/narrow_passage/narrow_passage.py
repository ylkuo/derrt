# -*- coding: utf-8 -*-

from demos.narrow_passage.generate_map import *
from demos.narrow_passage.state import *
from planners.derrtstar import DeRRTStar
from planners.hmm import GaussianHMM
from planners.nn import *
from planners.rrtstar import RRTStar
from shapely.geometry import Point
from tensorboardX import SummaryWriter
from utils.drawer import *
from utils.vector import *
from torch import FloatTensor as Tensor

import argparse
import matplotlib
import numpy as np
import shapely.geometry as geom
import torch
import torch.optim as optim


parser = argparse.ArgumentParser(description='DeRRT* for narrow passage example.')
# for debugging
parser.add_argument('--draw', action='store_true', help='Draw debug tree or not.')
parser.add_argument('--draw_map', action='store_true', help='Draw debug maps or not.')
parser.add_argument('--draw_train_map', action='store_true', help='Draw debug maps or not for training data.')
# for training
parser.add_argument('--n_train_samples', type=int, default=300, help='Number of training examples.')
parser.add_argument('--n_states', type=int, default=3, help='Number of HMM states.')
parser.add_argument('--uniform_hmm', action='store_true', help='Set to a uniform HMM or not.')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs for training.')
parser.add_argument('--use_cuda', action='store_true', help='Use GPU to train and evaluate the model.')
parser.add_argument('--device_id', type=int, default=0, help='Cuda device id.')
# for DeRRTstar
parser.add_argument('--n_samples', type=int, default=600, help='Number of samples drawn in DeRRTstar.')
parser.add_argument('--steer_dist', type=int, default=6, help='Distance to steer in expanding the tree.')
parser.add_argument('--use_rnn', action='store_true', help='Use RNN in DeRRTstar.')
parser.add_argument('--use_hmm', action='store_true', help='Use HMM in DeRRTstar.')
parser.add_argument('--load_model', action='store_true', help='Use saved model.')
parser.add_argument('--model_prefix', type=str, default='models/', help='Path to save and load models.')
parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation of random samples.')
# for testing
parser.add_argument('--n_test_rounds', type=int, default=10, help='Number of testing runs.')
parser.add_argument('--result_prefix', type=str, default='results/', help='Path to save results.')
args = parser.parse_args()


matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_batch(obs, pos, batch_size=100):
    batches = []
    sequences = []
    for i in range(len(obs)):
        seq = Sequence()
        prior_vec = np.asarray([1, 0])
        for j in range(len(obs[i])):
            done = False
            if j == len(obs[i])-1:
                theta = 0
                done = True
            else:
                new_vec = np.asarray([pos[i][j+1][0]-pos[i][j][0], pos[i][j+1][1]-pos[i][j][1]])
                theta = theta_from_vecs(prior_vec, new_vec)
            seq.add_sample(Sample(Tensor(obs[i][j]), prior_dist(),
                                  pos=np.asarray([pos[i][j][0], pos[i][j][1]]),
                                  values=[theta], is_end=done))
            prior_vec = new_vec
        sequences.append(seq)
        if len(sequences) >= batch_size:
            batches.append(Batch(sequences))
            sequences = []
    if len(sequences) > 0:
        batches.append(Batch(sequences))
    for batch in batches:
        batch.pad_sequences()
    return batches

def train(uniform_hmm=False):
    rndst = np.random.RandomState(1)
    # generate training data and extract features
    ss = PassageStateSpace(width=30, height=30)
    features = []; maps = []; pos = []
    for i in range(args.n_train_samples):
        y = rndst.uniform(-15, 15)
        width = rndst.uniform(1.0, 1.5)
        length = rndst.uniform(7, 26)
        map2d, start, goal, sol = one_passage(y, width, length,
                                              map_width=30, map_height=30,
                                              draw=args.draw_map)
        feature = []; parent = None; xs = []; ys = []
        for j in range(len(sol)):
            state = PassageState(ss, sol[j], env=map2d)
            if parent is not None: state.parent = parent
            obs = state.extract_feature()
            feature.append(obs)
            parent = state
            xs.append(state.value[0])
            ys.append(state.value[1])
        features.append(feature)
        maps.append((map2d, start, goal))
        pos.append(sol)
        if args.draw_train_map:
            fig = plt.figure()
            ax = plt.axes(xlim=(ss.components[0][0],ss.components[0][1]),
                          ylim=(ss.components[1][0],ss.components[1][1]))
            ax.set_aspect('equal')
            for b in map2d.world.bodies:
                if b.userData['name'] != 'obs_wall': continue
                polygon = geom.box(b.position[0]-b.userData['w']/2, b.position[1]-b.userData['h']/2,
                                   b.position[0]+b.userData['w']/2, b.position[1]+b.userData['h']/2)
                plot_poly(ax, polygon, 'blue')
            plt.plot(xs, ys, 'r-', lw=2.)
            plt.savefig('tmp_images/train_sol_{}.png'.format(i))
    # train the model
    features = np.asarray(features)
    n_features = len(features[0][0])
    if args.use_rnn:
        batches = get_batch(features, pos)
        model = EvalNetwork(example_feature=Tensor(features[0]),
                            proposal_types=['normal'],
                            rnn_dim=32,
                            on_cuda=args.use_cuda,
                            device_id=args.device_id)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        writer = SummaryWriter()
        for i in range(args.n_epochs):
            batch_loss = 0
            for batch in batches:
                success, loss = model.loss(batch, optimizer=optimizer)
                batch_loss += float(loss)
                if not success: continue
            writer.add_scalar('loss/channel', batch_loss/float(len(batches)), i)
            print('Epoch', i, ', loss: ', batch_loss/float(len(batches)))
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), i)
                writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i)
    else:
        model = GaussianHMM(n=args.n_states, m=n_features, uniform_init=uniform_hmm)
        if not uniform_hmm:
            model.train(features)
        else:
            model._compute_logprob()
        print('pi: ', model.pi)
        print('A: ', model.A)
    return model

def post_log(filename, mode, run_id, success):
    with open(filename, 'a') as log_file:
        logs = [mode, str(args.n_samples), str(run_id)]
        if success: logs.append('1')
        else: logs.append('0')
        log_file.write(','.join(logs) + '\n')

def main():
    map2d, start, goal, _ = one_passage(11, 1, 22, map_width=60, x_left=2,
                                        start=(-12.8,-10.3), goal=(29.5, 4.2),
                                        draw=args.draw_map)
    if (args.use_hmm or args.use_rnn) and not args.load_model:
        model = train(uniform_hmm=args.uniform_hmm)
        if args.use_rnn:
            torch.save(model, args.model_prefix + 'narrow_channel_rnn.model')
    elif args.use_rnn and args.load_model:
        model = torch.load(args.model_prefix + 'narrow_channel_rnn.model')
    else:
        model = None
    if args.use_hmm:
        mode = 'DeRRT*/HMM'
    elif args.use_rnn:
        mode = 'DeRRT*/RNN'
    else:
        mode = 'RRT*'
    success_count = 0
    ss = PassageStateSpace(width=60, height=30)
    start_val = list(start)
    goal = PassageState(ss, list(goal), env=map2d)
    for i in range(args.n_test_rounds):
        start = PassageState(ss, start_val, env=map2d)
        if args.use_hmm or args.use_rnn:
            planner = DeRRTStar(ss, args.steer_dist, args.n_samples, model, n_resample=40)
        else:
            planner = RRTStar(ss, args.steer_dist, args.n_samples)
        goal_fn = lambda state: ss.dist(state, goal) < 1.5
        plan = planner.solve(start, goal_fn)
        if len(plan) > 0:
            success_count += 1
            print('Found a solution')
        else:
            print('Cannot find a solution')
        post_log(args.result_prefix + 'narrow_passage.log', mode, i, len(plan) > 0)
        if args.draw:
            ax = planner.draw()
            goal_region = Point((goal.value[0], goal.value[1])).buffer(0.7, 3)
            plot_poly(ax, goal_region, 'green', 0.7)
            plt.savefig('tmp_images/tree_narrow_passage.png')
            plt.close()
        del planner
    print('# of success:', success_count)

if __name__ == '__main__':
    main()
