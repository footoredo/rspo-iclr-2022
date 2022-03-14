import glob
import os

import torch
import torch.nn as nn

# from a2c_ppo_acktr.envs import VecNormalize

# Magic box function for DiCE
def magic_box(x):
    """Magic Box Operator. x are log-probabilities"""
    return torch.exp(x - x.detach())


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    # if isinstance(venv, VecNormalize):
    #     return venv
    # elif hasattr(venv, 'venv'):
    #     return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=None):
    # print("init A", module, weight_init)
    if gain is not None:
        weight_init(module.weight.data, gain=gain)
    else:
        weight_init(module.weight.data)
    # print("init B", module)
    bias_init(module.bias.data)
    # print("init C", module)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
