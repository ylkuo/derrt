import copy
import numpy as np
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F

from planners.util import epsilon
from torch import FloatTensor as Tensor
from torch.distributions import Normal

def has_nan_or_inf(value):
    if torch.is_tensor(value):
        value = np.sum(value.cpu().detach().numpy())
        return np.isnan(value) or np.isinf(value)
    else:
        value = float(value)
        return (value == float('inf')) or (value == float('-inf')) or (value == float('NaN'))

class Sample(object):
    def __init__(self, obs, dists, values=[], pos=None, is_pad=False, is_end=False):
        self.dists = dists
        self.values = Tensor(values)  # direction
        self.obs = Tensor(obs)  # observed features
        self.is_pad = is_pad
        self.is_end = is_end
        if pos is not None: self.pos = Tensor(pos)
        self.rnn_input = None
        self.rnn_output = None
    def cuda(self, device=None):
        for i in range(len(self.dists)):
            if 'loc' in self.dists[i].__dict__:
                self.dists[i].loc = self.dists[i].loc.cuda(device)
            if 'scale' in self.dists[i].__dict__:
                self.dists[i].scale = self.dists[i].scale.cuda(device)
            if 'prob' in self.dists[i].__dict__:
                self.dists[i].prob = self.dists[i].prob.cuda(device)
        self.values = self.values.cuda(device)
        self.obs = self.obs.cuda(device)
        if self.pos is not None:
            self.pos = self.pos.cuda(device)
    def cpu(self):
        for i in range(len(self.values)):
            if 'loc' in self.dists[i].__dict__:
                self.dists[i].loc = self.dists[i].loc.cpu()
            if 'scale' in self.dists[i].__dict__:
                self.dists[i].scale = self.dists[i].scale.cpu()
            if 'prob' in self.dists[i].__dict__:
                self.dists[i].prob = self.dists[i].prob.cpu()
        self.values = self.values.cpu()
        self.obs = self.obs.cpu()
        if self.pos is not None:
            self.pos = self.pos.cpu()

class Sequence(object):
    def __init__(self):
        self.samples = []
    def add_sample(self, s):
        self.samples.append(s)
    def __len__(self):
        return len(self.samples)
    def cuda(self, device=None):
        for i in range(len(self.samples)):
            self.samples[i].cuda(device)
    def cpu(self):
        for i in range(len(self.samples)):
            self.samples[i].cpu()

class Batch(object):
    def __init__(self, sequences):
        self.sequences = sequences
        self.sequence_max_length = 0
        for sequence in sequences:
            if len(sequence) > self.sequence_max_length:
                self.sequence_max_length = len(sequence)
    def pad_sequences(self):
        for b in range(len(self)):
            last_sample = copy.deepcopy(self.sequences[b].samples[-1])
            last_sample.is_pad = True
            while len(self.sequences[b].samples) < self.sequence_max_length:
                self.sequences[b].add_sample(last_sample)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, key):
        return self.sequences[key]
    def cuda(self, device_id=None):
        for sequence in self.sequences:
            sequence.cuda(device_id)
    def cpu(self):
        for sequence in self.sequences:
            sequence.cpu()

class ProposalNormalNormal(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProposalNormalNormal, self).__init__()
        self._output_dim = output_dim
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, self._output_dim * 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))
    def forward(self, x, samples, dist_idx=0):
        x = torch.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:, :self._output_dim].view(len(samples))
        stddevs = torch.exp(x[:, self._output_dim:].view(len(samples)))
        prior_means = torch.stack([s.dists[dist_idx].mean for s in samples])
        prior_stddevs = torch.stack([s.dists[dist_idx].stddev for s in samples])
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        return Normal(means, stddevs)

class EmbeddingFC(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(EmbeddingFC, self).__init__()
        input_dim = input_shape[1]
        self.lin1 = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        return torch.relu(self.lin1(x))

class EmbeddingCNN2D2C(nn.Module):
    def __init__(self, input_shape, output_dim=16,
                 n_out_channels=16):
        super(EmbeddingCNN2D2C, self).__init__()
        self.input_channels = input_shape[2]
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, padding=2, out_channels=n_out_channels, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n_out_channels, padding=2, out_channels=n_out_channels*2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.cnn_output_shape = self.forward_cnn(torch.zeros(input_shape).unsqueeze(0)).shape
        cnn_output_dim = self.forward_cnn(torch.zeros(input_shape).unsqueeze(0)).nelement()
        self.lin1 = nn.Linear(cnn_output_dim, self.output_dim)
        self.lin2 = nn.Linear(output_dim, self.output_dim)
    def forward_cnn(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # out size: torch.Size([1, 32, 50, 50])
        return x
    def forward(self, x):
        x = self.forward_cnn(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        return x


class EvalNetwork(nn.Module):
    def __init__(self, on_cuda=False, device_id=0,
                 example_feature=None, embedding_type='fc', n_channels=16,
                 pos_dim=0, pos_idxs=[], rnn_dim=512,
                 rnn_depth=2, proposal_types=['normal'], dropout=0.2):
        super(EvalNetwork, self).__init__()

        self.dropout = dropout
        self.pos_dim = pos_dim
        self.pos_idxs = pos_idxs
        self.on_cuda = on_cuda
        self.device_id = 0

        # set up layers
        self.embedding_type = embedding_type
        self.set_sample_embedding(n_channels, example_feature, smp_emb_dim=rnn_dim)
        self.set_rnn(rnn_dim, rnn_depth)
        self.add_proposal_layers(proposal_types)

    def cuda(self, device_id):
        self.on_cuda = True
        self.device_id = device_id
        super(EvalNetwork, self).cuda(device_id)

    def cpu(self):
        self.on_cuda = False
        super(EvalNetwork, self).cpu()

    def get_parameter_str(self):
        ret = ''
        for p in self.parameters():
            ret = ret + '{0} {1}\n'.format(type(p.data), list(p.size()))
        return ret

    def set_sample_embedding(self, n_channels, example_feature, smp_emb_dim):
        input_shape = example_feature.size()
        if self.embedding_type == 'fc':
            self.smp_emb_dim = smp_emb_dim
            self.sample_layer = EmbeddingFC(input_shape, smp_emb_dim)
        elif self.embedding_type == 'cnn2d2c':
            self.n_channels = n_channels
            self.sample_layer = EmbeddingCNN2D2C(input_shape,
                                                 n_out_channels=n_channels,
                                                 has_fc_out=False)
            self.smp_emb_dim = self.sample_layer.cnn_output_shape[1]

    def set_rnn(self, rnn_dim, rnn_depth):
        self.rnn_dim = rnn_dim
        self.rnn_depth = rnn_depth
        self.rnn_input_dim = self.smp_emb_dim + self.pos_dim
        self.rnn = nn.GRU(self.rnn_input_dim, rnn_dim, rnn_depth,
                          dropout=self.dropout)
        self.rnn.reset_parameters()
        # orthogonal initialization of recurrent weights
        for _, hh, _, _ in self.rnn.all_weights:
            for i in range(0, hh.size(0), self.rnn.hidden_size):
                nn.init.orthogonal_(hh[i:i + self.rnn.hidden_size])

    def add_proposal_layers(self, proposal_types):
        self.proposal_layers = []
        for i, proposal_type in enumerate(proposal_types):
            if proposal_type == 'normal':
                proposal = ProposalNormalNormal(self.rnn_dim, 1)
            self.proposal_layers.append(proposal)
            self.add_module('proposal_'+str(i), proposal)

    def forward(self, smp=None, prev_hidden_state=None, is_batch=False):
        if not is_batch:
            smp = [smp]
        # get low level embeddings
        if self.on_cuda:
            for s in smp: s.cuda()
        smp_input = torch.stack([s.obs for s in smp])
        rnn_input = self.sample_layer(smp_input)
        if self.pos_dim > 0:
            pos = []; smp_emb_new = []
            for s in smp:
                pos.append([s.pos[i] for i in self.pos_idxs])
            pos = torch.tensor(pos)
            if self.on_cuda:
                pos = pos.cuda(self.device_id)
            for b in range(len(smp)):
                smp_emb_new.append(torch.cat((rnn_input[b], pos[b])))
            rnn_input = torch.stack(smp_emb_new)
        rnn_input = rnn_input.unsqueeze(0)
        if prev_hidden_state is None:
            rnn_output, hidden_state = self.rnn(rnn_input)
        else:
            rnn_output, hidden_state = self.rnn(rnn_input, prev_hidden_state)
        proposal_input = rnn_output[0]  # only has one time step, gets that one
        proposals = []
        for proposal_layer in self.proposal_layers:
            proposal_output = proposal_layer.forward(proposal_input, smp)
            proposals.append(proposal_output)
        return proposals, hidden_state

    def loss_proposal_step(self, proposal, samples, idx=0):
        samples_values = torch.stack([s.values[idx] for s in samples])
        mask = torch.stack([Tensor([1]) if not s.is_pad else Tensor([0]) for s in samples]).view(len(samples))
        if self.on_cuda:
            mask = mask.cuda(self.device_id)
            samples_values = samples_values.cuda(self.device_id)
        # only consider the non-padded time-step for computing loss
        l = proposal.log_prob(samples_values) * mask
        if has_nan_or_inf(l):
            print('Warning: NaN or Inf encountered proposal log_prob.')
            return False, torch.zeros(l.shape), mask
        return True, l, mask

    def loss(self, batch, optimizer=None):
        hidden_state = None
        log_prob = 0
        seq_len = torch.zeros([len(batch)])
        if self.on_cuda:
            seq_len = seq_len.cuda(self.device_id)
        for time_step in range(batch.sequence_max_length):
            smp = [batch[b].samples[time_step] for b in range(len(batch))]
            proposals, hidden_state = \
                self.forward(smp=smp, prev_hidden_state=hidden_state, is_batch=True)
            # compute loss for proposals
            for i in range(len(proposals)):
                success, tmp_prob, mask = self.loss_proposal_step(proposals[i], smp, idx=i)
                if not success: return False, 0
                log_prob += tmp_prob
            seq_len += mask
        # aggregate loss
        total_loss = -torch.sum(log_prob/seq_len) / len(batch)
        # backpropagate
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return True, total_loss
