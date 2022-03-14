import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from rspo.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def likelihoods(self, actions):
        return self.log_probs(actions)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)
        # _log_probs = (super().log_prob(actions) + torch.log(self.scale) + math.log(math.sqrt(2 * math.pi))) / 100.
        # return _log_probs.sum(-1, keepdim=True)

    def likelihoods(self, actions):
        log_probs = (super().log_prob(actions) + torch.log(self.scale) + math.log(math.sqrt(2 * math.pi))) * (self.scale ** 2)
        # self.scale *= 2
        # log_probs2 = (super().log_prob(actions) + torch.log(self.scale) + math.log(math.sqrt(2 * math.pi))) * (
        #             self.scale ** 2)
        # print(log_probs - log_probs2)
        # self.scale /= 2
        return log_probs.sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def likelihoods(self, actions):
        return self.log_probs(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_ref=False):
        super(Categorical, self).__init__()

        if is_ref:
            init_ = lambda m: m
        else:
            init_ = lambda m: init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        # self.linear = nn.Linear(num_inputs, num_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        # print(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, activation=None):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        if activation is None:
            self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        elif activation == "tanh":
            # print("!!!")
            self.fc_mean = nn.Sequential(init_(nn.Linear(num_inputs, num_outputs)),
                                         nn.Tanh())
        else:
            raise NotImplementedError
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        # print(action_mean)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        # print(action_mean)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
