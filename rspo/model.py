import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rspo.distributions import Bernoulli, Categorical, DiagGaussian, FixedNormal
from rspo.utils import init
from copy import deepcopy
import math


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, action_activation=None, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        # print("21312312")

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n

            def action_embedding(actions):
                return torch.nn.functional.one_hot(actions, num_classes=num_outputs).squeeze(-2)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            def action_embedding(actions):
                return actions
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]

            raise NotImplementedError
        else:
            raise NotImplementedError

        base_kwargs["num_actions"] = num_outputs
        base_kwargs["action_embedding"] = action_embedding

        self.ob_dim = obs_shape[0]
        self.h_dim = base_kwargs["hidden_size"]
        self.ac_dim = num_outputs


        # print("start")
        self.base = base(obs_shape[0], **base_kwargs)
        # print('finish')

        if action_space.__class__.__name__ == "Discrete":
            self.dist = Categorical(self.base.output_size, num_outputs, is_ref=base_kwargs["is_ref"])
        elif action_space.__class__.__name__ == "Box":
            self.dist = DiagGaussian(self.base.output_size, num_outputs, activation=action_activation)
        elif action_space.__class__.__name__ == "MultiBinary":
            self.dist = Bernoulli(self.base.output_size, num_outputs)

        self.obs_rms = None

    def _get_weights(self, weights, sizes):
        hiddens = []
        biases = []
        cur = 0
        for x, y in sizes:
            hiddens.append(torch.tensor(weights[cur: cur + x * y].reshape(x, y), dtype=torch.float))
            cur += x * y
        for x, _ in sizes[:-1]:
            biases.append(torch.tensor(weights[cur: cur + x].reshape(x), dtype=torch.float))
            cur += x
        return hiddens, biases

    def load_DvD_weight(self, weight_path):
        weights = np.load(weight_path)
        mu = weights[-self.ob_dim * 2: -self.ob_dim]
        std = weights[-self.ob_dim:]
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        self.obs_rms = RunningMeanStd(shape=(self.ob_dim,))
        self.obs_rms.mean = mu
        self.obs_rms.var = np.square(std)
        # print(mu, std)
        weights = weights[:-self.ob_dim * 2]
        sizes = [(self.h_dim, self.ob_dim), (self.h_dim, self.h_dim), (self.ac_dim, self.h_dim)]
        hiddens, weights = self._get_weights(weights, sizes)
        h1, h2, h999 = hiddens
        b1, b2 = weights
        self.base.actor[0].weight.data = h1
        self.base.actor[0].bias.data = b1
        self.base.actor[2].weight.data = h2
        self.base.actor[2].bias.data = b2
        self.dist.fc_mean[0].weight.data = h999
        self.dist.fc_mean[0].bias.data.fill_(0.)

        # print("!!!")

    def normalize_obs(self, obs):
        obs_rms = self.obs_rms
        if obs_rms is not None:
            if obs_rms.mean.dtype == np.float64:
                obs_rms.mean = obs_rms.mean.astype(np.float32)
                obs_rms.var = obs_rms.var.astype(np.float32)
            epsilon = 1e-8
            clip_obs = 10.
            obs = torch.clamp((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)
            # print(obs)
            return obs
        else:
            return obs

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def process_base(self, inputs, *args, **kwargs):
        inputs = self.normalize_obs(inputs)
        return self.base(inputs, *args, **kwargs)

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, _, rnn_hxs = self.process_base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        # torch.set_printoptions(precision=3, sci_mode=False)
        # print(dist.probs.detach())

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _, _ = self.process_base(inputs, rnn_hxs, masks)
        return value

    def get_strategy(self, inputs, rnn_hxs, masks):
        value, actor_features, _, rnn_hxs = self.process_base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        return dist
        # if isinstance(dist, FixedNormal):
        #     return dist.mean, dist.stddev
        # else:
        #     return dist.probs

    def get_probs(self, inputs, rnn_hxs, masks, action):
        value, actor_features, _, rnn_hxs = self.process_base(inputs, rnn_hxs, masks)
        # print(actor_features)
        dist = self.dist(actor_features)
        # print(dist)

        action_log_probs = dist.log_probs(action)
        # print(f"      {action_log_probs + torch.log(dist.scale).sum() + math.log(math.sqrt(2. * math.pi)) * dist.scale.size()[-1]}")
        # print(f"      {torch.log(dist.scale).sum()}")

        # print(action_log_probs)

        return torch.exp(action_log_probs)

    def get_likelihoods(self, inputs, rnn_hxs, masks, action):
        value, actor_features, _, rnn_hxs = self.process_base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_likelihoods = dist.likelihoods(action)

        return action_likelihoods

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, _, rnn_hxs = self.process_base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist

    def get_reward_prediction(self, inputs, rnn_hxs, masks, action):
        _, _, prediction, _ = self.process_base(inputs, rnn_hxs, masks, action)

        return prediction


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            # print(x.shape)
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class AttentionTransform(nn.Module):
    def __init__(self, num_input, hidden_size):
        super(AttentionTransform, self).__init__()
        self.k = nn.Linear(num_input, hidden_size)
        self.v = nn.Linear(num_input, hidden_size)

    def forward(self, _input):
        k = self.k(_input)
        v = self.v(_input)
        return k, v


class DenseLayer(nn.Module):
    def __init__(self, num_input, hidden_size):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(num_input, hidden_size)
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.layer_normself.activation(self.linear(x))


def make_dense(num_input, hidden_size):
    return nn.Sequential(nn.Linear(num_input, hidden_size), nn.Tanh(), nn.LayerNorm(hidden_size))


class AttentionBase(NNBase):
    def __init__(self, num_input, input_structure, recurrent=False, hidden_size=64, num_heads=4):
        super(AttentionBase, self).__init__(recurrent, hidden_size, hidden_size)

        self._input_structure = input_structure
        self._hidden_size = hidden_size
        self._num_heads = num_heads

        self.entity_index = dict()
        self.n_entities = 0
        self.entity_num_input = []
        transforms = []

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), np.sqrt(2))

        self_num_input = 0
        self.self_input_part = (0, 0)

        sum_length = 0
        for name, length in input_structure:
            if name == "self":
                self_num_input = length
                self.self_input_part = (sum_length, sum_length + length)
            else:
                if name not in self.entity_index:
                    self.entity_index[name] = self.n_entities
                    self.entity_num_input.append(length)
                    self.n_entities += 1
            sum_length += length

        for num_input in self.entity_num_input:
            transforms.append(make_dense(num_input + self_num_input, hidden_size))

        self.transforms = nn.ModuleList(transforms)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.attention_dense1 = make_dense(hidden_size, hidden_size)
        self.attention_dense2 = make_dense(hidden_size + self_num_input, hidden_size)

        self.actor = nn.Sequential(
            make_dense(hidden_size, hidden_size),
            make_dense(hidden_size, hidden_size))

        self.critic = nn.Sequential(
            make_dense(hidden_size, hidden_size),
            make_dense(hidden_size, hidden_size))

        self.critic_linear = nn.Linear(hidden_size, 1)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        self_input = inputs[:, self.self_input_part[0]: self.self_input_part[1]]
        sum_length = 0
        entity_inputs = [[] for _ in range(self.n_entities)]
        hs = []
        for name, length in self._input_structure:
            if name != "self":
                entity_inputs[self.entity_index[name]].append(
                    torch.cat([self_input, inputs[:, sum_length: sum_length + length]], dim=1))
            sum_length += length
        for i in range(self.n_entities):
            _inputs = torch.stack(entity_inputs[i], dim=1)
            h = self.transforms[i](_inputs)
            hs.append(h)
        x = torch.cat(hs, dim=1)
        bs = x.size()[0]
        n = x.size()[1]
        nh = self._num_heads
        hs = self._hidden_size // nh
        q = self.q(x).view(bs, n, nh, hs).transpose(1, 2).reshape(-1, n, hs)
        k = self.k(x).view(bs, n, nh, hs).transpose(1, 2).reshape(-1, n, hs)
        v = self.v(x).view(bs, n, nh, hs).transpose(1, 2).reshape(-1, n, hs)

        w = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(hs), dim=-1)
        z = torch.bmm(w, v).view(bs, nh, n, hs).transpose(1, 2).reshape(bs, n, nh * hs)
        y = self.attention_dense1(z) + x
        y = torch.mean(y, dim=1)
        y = self.attention_dense2(torch.cat([y, self_input], dim=-1))

        if self.is_recurrent:
            y, rnn_hxs = self._forward_gru(y, rnn_hxs, masks)

        hidden_critic = self.critic(y)
        hidden_actor = self.actor(y)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


ACTIVATION_FN = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU
}


class LinearBase(NNBase):
    def __init__(self, num_inputs):
        super(LinearBase, self).__init__(False, num_inputs, num_inputs)

        self.num_inputs = num_inputs

        self.actor = nn.Identity()

        self.critic = nn.Identity()

        self.critic_linear = nn.Linear(num_inputs, 1)

        self.train()

    @property
    def output_size(self):
        return self.num_inputs

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, activation="tanh", critic_dim=1, is_ref=False,
                 predict_reward=False, num_actions=1, action_embedding=None, timestep_mask=False, rnd=False):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        # print("start")

        if recurrent:
            num_inputs = hidden_size

        self.timestep_mask = timestep_mask and not recurrent

        # print(is_ref)

        if is_ref:
            init_ = lambda m:  m
        else:
            # print(111)
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), np.sqrt(2))
        # init_ = lambda m: init(m, lambda x: nn.init.constant_(x, 0), lambda x: nn.init.constant_(x, 0), None)
        # init_ = lambda m:  m

        self.num_inputs = num_inputs
        activation_fn = ACTIVATION_FN[activation]
        # if not is_ref:
        #     print("B", critic_dim, num_inputs, hidden_size)
        # print(init_)
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), activation_fn(),
            init_(nn.Linear(hidden_size, hidden_size)), activation_fn())

        self.action_input_mask = torch.ones(num_inputs)
        if self.timestep_mask:
            self.action_input_mask[0] = 0.

        # print("finish", num_inputs, hidden_size)
        # if not is_ref:
        #     print("C", critic_dim, num_inputs, hidden_size)

        # self.actor = nn.Identity()

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), activation_fn(),
            init_(nn.Linear(hidden_size, hidden_size)), activation_fn())

        self.critic_linear = init_(nn.Linear(hidden_size, critic_dim))
        self.critic_dim = critic_dim

        self.predict_reward = predict_reward
        if predict_reward:
            self.num_actions = num_actions
            self.action_embedding = action_embedding
            self.predictor = nn.Sequential(
                nn.Linear(num_inputs + num_actions, hidden_size), activation_fn(),
                nn.Linear(hidden_size, 1)
            )
        self.rnd = rnd
        if rnd:
            self.random_net = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), activation_fn(),
                nn.Linear(hidden_size, 32)
            )
            self.random_net_predictor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), activation_fn(),
                nn.Linear(hidden_size, 32)
            )

        self.train()

    # @property
    # def output_size(self):
    #     return self.num_inputs

    def forward(self, inputs, rnn_hxs, masks, actions=None):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x * self.action_input_mask)

        if self.predict_reward and actions is not None:
            # print(inputs.size(), actions.size())
            # print(actions.dtype)
            actions = self.action_embedding(actions)
            _inputs = torch.cat((inputs, actions), dim=-1)
            reward_prediction = self.predictor(_inputs)
        else:
            reward_prediction = None
        if self.rnd:
            random_net_value = self.random_net(inputs).detach()
            random_net_prediction = self.random_net_predictor(inputs)
        else:
            random_net_value = None
            random_net_prediction = None
        prediction = (reward_prediction, random_net_value, random_net_prediction)

        return self.critic_linear(hidden_critic), hidden_actor, prediction, rnn_hxs


if __name__ == "__main__":
    import time
    from pettingzoo.mpe import simple_tag_v1
    env = simple_tag_v1.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2, max_frames=50)

    mlp = MLPBase(env.observation_spaces["agent_0"].shape[0])
    att = AttentionBase(env.observation_spaces["agent_0"].shape[0], env.input_structures["agent_0"])

    inputs = torch.randn((64, env.observation_spaces["agent_0"].shape[0]))

    st = time.time()
    for _ in range(1000):
        mlp.zero_grad()
        c, _, _ = mlp(inputs, None, None)
        c.mean().backward(create_graph=True)
    print("mlp:", time.time() - st)

    torch.set_num_threads(1)
    st = time.time()
    # while True:
    for _ in range(1000):
        att.zero_grad()
        c, _, _ = att(inputs, None, None)
        c.mean().backward(create_graph=True)
    print("att:", time.time() - st)
