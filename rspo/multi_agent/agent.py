import os
import gym
import numpy as np
import torch
import joblib
import time
from collections import deque

import threading
from copy import deepcopy

import multiprocessing as mp

from argparse import Namespace

from stable_baselines3.common.running_mean_std import RunningMeanStd

from rspo.storage import RolloutStorage
from rspo.distributions import FixedNormal, FixedCategorical
from .utils import get_agent, release_all_locks, acquire_all_locks, mkdir2, load_actor_critic, ts, reseed, make_env, \
    get_action_repr_fn, dipg_ker


def true_func(n_iter):
    return True


class RefAgent(mp.Process):
    def __init__(self, agent, agent_id, ref_id, num_refs, num_actions, args: Namespace, obs_shm,
                 buffer_start, buffer_end, ref_shm, obs_locks, act_locks, ref_locks):
        super(RefAgent, self).__init__()
        self.agent = agent
        self.agent_id = agent_id
        self.ref_id = ref_id
        self.num_refs = num_refs
        self.num_actions = num_actions

        self.seed = reseed(args.seed, "agent-{}-ref-{}".format(self.agent_id, self.ref_id))
        self.num_steps = args.num_steps
        self.num_envs = args.num_processes
        self.num_updates = args.num_env_steps // args.num_steps // args.num_processes
        self.num_episodes = args.num_steps // args.episode_steps
        self.episode_length = args.episode_steps

        self.obs_shm = obs_shm
        self.buffer_start = buffer_start
        self.buffer_end = buffer_end
        self.ref_shm = ref_shm
        self.obs_locks = obs_locks
        self.act_locks = act_locks
        self.ref_locks = ref_locks

        # print(len(self.obs_locks))

        self.dtype = np.float32
        item_size = np.zeros(1, dtype=self.dtype).nbytes
        buffer_length = item_size * self.num_envs * num_actions
        offset = buffer_length * self.ref_id
        self.place = np.frombuffer(self.ref_shm.buf[offset: offset + buffer_length], dtype=self.dtype).reshape(
            self.num_envs, self.num_actions)

    def get_obs(self):
        acquire_all_locks(self.obs_locks)
        # print(1)
        data = np.frombuffer(self.obs_shm.buf[self.buffer_start: self.buffer_end], dtype=self.dtype).reshape(
            self.num_envs, -1)
        obs = data[:, :-4]
        reward = data[:, -4:-3]
        normalized_reward = data[:, -3:-2]
        done = data[:, -2:-1]
        bad_mask = data[:, -1:]
        # if any(reward > 0.):
        #     print(data, obs, reward)
        # self.log("release obs locks")
        # release_all_locks(self.obs_locks)
        return ts(obs), ts(reward), ts(normalized_reward), ts(done), ts(bad_mask)

    def write(self, probs):
        # num_envs * float
        probs = probs.numpy()
        assert probs.dtype == np.float32
        np.copyto(self.place, probs)

    def run(self):
        torch.set_num_threads(1)
        np_random = np.random.RandomState(self.seed)
        for i in range(self.num_updates):
            if self.ref_id == 0:
                print("update", i)
            for j in range(self.num_episodes):
                obs, _, _, _, _ = self.get_obs()
                for k in range(self.episode_length):
                    # print(k)
                    probs = self.agent.get_strategy(obs, None, None).detach()
                    release_all_locks(self.act_locks)
                    # print(probs)
                    self.write(probs)
                    release_all_locks(self.ref_locks)
                    obs, _, _, _, _ = self.get_obs()
                release_all_locks(self.act_locks)


class Agent(mp.Process):
    def __init__(self, agent_id, agent_name, thread_limit, logger, args: Namespace, obs_space, input_structure,
                 act_space, act_sizes, main_conn, obs_shm, buffer_start, buffer_end, act_shm,
                 ref_shm, obs_locks, act_locks, ref_locks, use_attention=False, save_dir=None, train=true_func,
                 num_refs=None, reference_agent=None, data_queue=None, norm_obs=True, norm_reward=True):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.thread_limit = thread_limit
        self.logger = logger

        self.args = args
        self.seed = reseed(args.seed, "agent-{}".format(self.agent_id))
        self.num_steps = args.num_steps
        self.num_envs = args.num_processes
        self.num_agents = args.num_agents
        self.batch_size = self.num_steps * self.num_envs
        self.num_updates = args.num_env_steps // args.num_steps // args.num_processes
        self.save_interval = args.save_interval
        self.save_dir = mkdir2(save_dir or args.save_dir, str(agent_name))

        self.obs_space = obs_space
        # print(obs_space)
        self.input_structure = input_structure
        self.act_space = act_space

        self.main_conn = main_conn
        self.obs_shm = obs_shm
        self.buffer_start = buffer_start
        self.buffer_end = buffer_end
        self.act_shm = act_shm
        self.ref_shm = ref_shm
        self.obs_locks = obs_locks
        self.act_locks = act_locks
        self.ref_locks = ref_locks

        self.use_attention = use_attention
        self.train = train

        self.num_refs = num_refs
        self.reference_agent = reference_agent

        self.env = make_env(args.env_name, args.episode_steps, args.env_config)
        self.num_sym = self.env.env.num_sym if args.use_symmetry_for_reference else 1

        self.act_sizes = act_sizes
        self.act_place_env = []

        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.obs_rms = None
        self.ret_rms = None
        self.ret = None

        self.epsilon = 1e-8
        self.clip_obs = 10.
        self.clip_reward = 10.

        self.trajectories = None

        if args.gail:
            raise ValueError("gail is not supported")

        self.data_queue = data_queue

    def log(self, msg):
        self.logger("{}: {}".format(self.agent_name, msg))

    def get_obs(self):
        # self.log("acquire obs locks")
        acquire_all_locks(self.obs_locks)
        data = np.frombuffer(self.obs_shm.buf[self.buffer_start: self.buffer_end], dtype=np.float32).reshape(self.num_envs, -1)
        obs = data[:, :-4]
        reward = data[:, -4:-3]
        normalized_reward = data[:, -3:-2]
        done = data[:, -2:-1]
        bad_mask = data[:, -1:]
        # if any(reward > 0.):
        #     print(data, obs, reward)
        # self.log("release obs locks")
        # release_all_locks(self.obs_locks)
        return obs, reward, ts(normalized_reward), ts(done), ts(bad_mask)

    def get_ref_strategies(self, num_refs):
        acquire_all_locks(self.ref_locks)
        data = np.frombuffer(self.ref_shm.buf, dtype=np.float32).reshape(num_refs, self.num_envs, -1)
        return data

    def put_act(self, i_env, act):
        # self.log("{}: {}".format(i_env, act))
        self.act_place_env[i_env][:] = act

    @staticmethod
    def obs_distance(obs1, obs2):
        return np.linalg.norm((obs1 - obs2)[-4:])

    @staticmethod
    def obs_distance_all(obs, obs_list):
        return min([Agent.obs_distance(obs, other_obs) for other_obs in obs_list])

    def _normalize_obs(self, obs, obs_rms):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_obs(self, obs, update=True):
        obs_ = deepcopy(obs)
        obs_rms = self.obs_rms
        if self.norm_obs and obs_rms is not None:
            if update:
                obs_rms.update(obs_)
            obs_ = self._normalize_obs(obs_, obs_rms)
        # print(obs, new_obs)
        return obs_

    def process_obs(self, obs, update=True):
        return ts(obs), ts(self.normalize_obs(obs, update))

    def _update_reward(self, reward):
        if self.norm_reward and self.ret_rms is not None:
            # print(self.ret.shape, reward.shape)
            self.ret = self.ret * self.args.gamma + reward
            self.ret_rms.update(self.ret)

    def _normalize_reward(self, reward, ret_rms, center=False):
        reward_ = deepcopy(reward)
        if ret_rms is not None:
            if center:
                reward_ -= ret_rms.mean
            reward_ = np.clip(reward_ / np.sqrt(ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward_

    def normalize_reward(self, reward):
        ret_rms = self.ret_rms
        return self._normalize_reward(reward, ret_rms)

    def run(self):
        if self.thread_limit is not None:
            torch.set_num_threads(self.thread_limit)

        self.obs_rms = RunningMeanStd(shape=self.obs_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = np.zeros(self.num_envs)

        act_sizes = self.act_sizes
        agent_id = self.agent_id
        act_size = act_sizes[agent_id]
        sum_act_size = sum(act_sizes)
        act_pos = sum(act_sizes[:agent_id])
        dtype = np.float32
        item_size = np.zeros(1, dtype=dtype).nbytes
        self.act_place_env = []
        cur_act_pos = act_pos * item_size
        for i in range(self.num_envs):
            self.act_place_env.append(
                np.frombuffer(self.act_shm.buf[cur_act_pos: cur_act_pos + item_size * act_size], dtype=dtype))
            cur_act_pos += sum_act_size * item_size

        args = self.args
        use_dice = args.algo == "loaded-dice"
        # dice_lambda = args.dice_lambda if use_dice else None
        np_random = np.random.RandomState(self.seed)
        if args.reseed_step is not None and args.reseed_step < 0:
            seed = reseed(self.seed, "reseed-{}".format(args.reseed_z))
            # seed = self.seed
            # for iz in range(args.reseed_z):
            #     seed = np_random.randint(10000)
            torch.manual_seed(seed)
        else:
            torch.manual_seed(self.seed)
        # self.log(self.obs_space)
        ref = self.args.use_reference
        ref_agents = self.reference_agent
        # self.log("111")
        if ref and type(ref_agents) != list:
            ref_agents = [ref_agents]
        n_ref = self.num_refs if self.num_refs is not None else len(ref_agents) * self.num_sym if ref else 0
        sample_n_ref = len(ref_agents) * self.num_sym if ref else 0
        # self.log(12312312312)
        actor_critic, agent = get_agent(self.agent_name, self.args, self.obs_space, self.input_structure,
                                        self.act_space, self.save_dir, n_ref=n_ref, is_ref=False)
        # print(self.num_updates)
        likelihood_rms = RunningMeanStd(shape=sample_n_ref)
        reward_prediction_rms = RunningMeanStd(shape=sample_n_ref)
        rnd_rms = RunningMeanStd(shape=())
        # self.log("13123123")
        agent.ref_agent = self.reference_agent
        # print(self.num_steps)
        # self.log("123123")
        if args.load_dir is not None and args.load:
            self.log("Loading model from {}".format(args.load_dir))
            obs_rms, _ = load_actor_critic(actor_critic, args.load_dir, self.agent_name, args.load_step)
            if obs_rms is not None:
                self.obs_rms = obs_rms
                self.norm_obs = True
            else:
                self.norm_obs = False
            # print(obs_rms)
            self.log("Done.")

        if args.load_dvd_weights_dir is not None:
            print("loading dvd weights...")
            actor_critic.load_DvD_weight(args.load_dvd_weights_dir)
            self.obs_rms = None

        episode_steps = self.args.episode_steps
        num_envs = self.num_envs
        num_episodes = self.num_steps // episode_steps

        rollouts = RolloutStorage(episode_steps, num_envs,
                                  self.obs_space.shape, self.act_space,
                                  actor_critic.recurrent_hidden_state_size,
                                  num_refs=sample_n_ref, num_value_refs=n_ref, use_rnd=args.use_rnd)

        valid_rollouts = RolloutStorage(self.num_steps, num_envs,
                                        self.obs_space.shape, self.act_space,
                                        actor_critic.recurrent_hidden_state_size,
                                        num_refs=sample_n_ref, num_value_refs=n_ref, use_rnd=args.use_rnd)

        self.main_conn.recv()

        def merge_ref_strat(_obs):
            ref_strat = [ra.get_strategy(_obs, None, None) for ra in ref_agents]
            ref_strat = torch.stack(ref_strat, dim=2).max(dim=2)
            return torch.cat([_obs, ref_strat], dim=1)

        observations = [[[] for _ in range(num_episodes)] for _ in range(num_envs)]

        statistics = dict(reward=[], grad_norm=[], value_loss=[], action_loss=[], rnd_loss=[], dist_entropy=[], dist_penalty=[],
                          likelihood=[], total_episodes=[], accepted_episodes=[], efficiency=[], reward_prediction_loss=[])
        sum_reward = 0.
        sum_dist_penalty = 0.
        actual_use_ref = ref_agents is not None and ref_agents[0] is not None

        if actual_use_ref:
            sum_likelihood = np.zeros(sample_n_ref)
            sum_likelihood_capped = np.zeros((num_envs, sample_n_ref))
            likelihood_threshold = args.likelihood_threshold
            if type(likelihood_threshold) != list:
                likelihood_threshold = [likelihood_threshold] * len(ref_agents)
            likelihood_threshold = np.array(likelihood_threshold).repeat(self.num_sym)
        else:
            sum_likelihood = np.zeros(1)
            sum_likelihood_capped = 0.
            likelihood_threshold = 0.
        # print(sum_likelihood)
        # last_update_iter = -1
        update_counter = 0

        episode_rewards = deque(maxlen=10)

        if args.auto_threshold is not None:
            threshold_selected = False
        else:
            threshold_selected = True

        trajectories = []
        current_trajectory = []
        collect_trajectories = args.collect_trajectories or args.dipg

        action_repr = get_action_repr_fn(self.act_space)

        dipg_g = None
        rnd_size = 1 if args.use_rnd else 0

        for it in range(self.num_updates):
            if args.use_linear_lr_decay:
                progress = 1. - it / self.num_updates
                for param_group in agent.optimizer.param_groups:
                    param_group["lr"] = args.lr * progress

            report_data = {"iteration": it, "agent": self.agent_name}
            omega = 1.
            omega_exploration_reward = 1.
            progress = min(1., it / self.num_updates)
            if args.threshold_annealing_schedule == "linear":
                omega = 1. - progress
            elif args.threshold_annealing_schedule == "cosine":
                omega = np.cos(progress)
            if args.exploration_reward_annealing_schedule == "linear":
                omega_exploration_reward = 1. - progress
            elif args.exploration_reward_annealing_schedule == "cosine":
                omega_exploration_reward = np.cos(progress)

            report_data["omega"] = omega
            report_data["omega_exploration_reward"] = omega_exploration_reward

            self.log(f"threshold omega: {omega}")
            self.log(f"exploration reward omega: {omega_exploration_reward}")

            sampling_st = time.time()
            total_samples = 0
            total_episodes = 0
            accepted_episodes = 0
            accepted_episodes_ref = np.zeros(sample_n_ref)
            cur_agent = actor_critic
            # self.log(it)
            # self.log("iter: {}".format(it))
            remain_episodes = num_episodes * num_envs
            tmp_likelihood = []
            ref_time = 0.
            copy_time = 0.

            while True:
                total_episodes += num_envs
                decay = 1.
                # self.log(total_episodes)
                original_obs, _, _, _, _ = self.get_obs()
                original_obs, obs = self.process_obs(original_obs)

                if collect_trajectories:
                    current_trajectory.append(original_obs.detach().numpy())

                # self.log("step {} - received {}".format(0, obs))
                # self.log(obs.shape)
                rollouts.obs[0].copy_(obs)
                rollouts.original_obs[0].copy_(original_obs)
                for step in range(episode_steps):
                    # self.log(step)
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = cur_agent.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step], deterministic=args.deterministic)
                        # size = tuple(action.size())
                        # int_type = action.dtype
                        # random_action = torch.tensor(np_random.randint(self.act_space.n, size=size), dtype=int_type)
                        # random_filter = np_random.rand(*size)
                        # mask = torch.tensor(random_filter < 0.05, dtype=int_type)
                        # action = mask * random_action + (1 - mask) * action
                        # self.log("step {} - act {}".format(it * self.num_steps + step, action))
                    action = action.data
                    for i_env in range(self.num_envs):
                        self.put_act(i_env, action[i_env].numpy())
                    # self.log("release act locks")
                    release_all_locks(self.act_locks)

                    if collect_trajectories:
                        repr_action = action_repr(action.detach().numpy())
                        current_trajectory.append(repr_action)

                    # self.log(step)
                    original_obs, reward, _, done, bad_mask = self.get_obs()
                    original_obs, obs = self.process_obs(original_obs)

                    if collect_trajectories:
                        current_trajectory.append(original_obs.detach().numpy())

                    # obs = self.normalize_obs(obs)
                    # obs = ts(obs)
                    # self.log("{}, {}, {}".format(obs[0], reward[0], done[0]))
                    __reward = reward.reshape(-1)
                    self._update_reward(__reward)
                    normalized_reward = ts(self.normalize_reward(__reward)).unsqueeze(-1)
                    # print(_reward.shape, normalized_reward.shape, self.ret.shape)
                    _reward = normalized_reward
                    reward = ts(reward)
                    bd = done.numpy().astype(np.bool).reshape(-1)
                    # print(bd)
                    self.ret[bd] = 0.
                    # print(reward)
                    if step == episode_steps - 1:
                        # self.log(f"step: {step}, done: {done[4]}, obs: {obs[4]}")
                        assert all(done)
                        if collect_trajectories:
                            trajectory = np.concatenate(current_trajectory, axis=1)
                            trajectories.append(trajectory)
                            current_trajectory = []
                    else:
                        if any(done):
                            self.log(f"step: {step}, done: {done[4]}, obs: {obs[4]}")
                        assert not any(done)
                    # self.log("{}, {}".format(step, done))

                    st = time.time()
                    if ref:
                        # def ref_get_prob(_agent):
                        #     return _agent.get_probs(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                             rollouts.masks[step], action).detach().squeeze(dim=-1)
                        # ref_threads = []
                        # for ref_agent in ref_agents:
                        #     ref_thread = threading.Thread(target=ref_get_prob, args=(ref_agent,))
                        #     ref_thread.start()
                        #     ref_threads.append(ref_thread)
                        # with mp.pool.ThreadPool(processes=1) as pool:
                        #     _dis = list(pool.map(ref_get_prob, ref_agents))
                        # _dis = [ref_agent.get_probs(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                             rollouts.masks[step], action).detach().squeeze(dim=-1)
                        #         for ref_agent in ref_agents]
                        # _dis2 = [ref_agent.get_value(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                              rollouts.masks[step]).detach().squeeze()
                        #          for ref_agent in ref_agents]
                        # print(len(_dis), _dis[1])
                        # for ref_conn in ref_conns:
                        #     ref_conn.send((rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                   rollouts.masks[step], action))
                        # self.log("{}, requested".format(step))
                        # ref_strategies = self.get_ref_strategies(len(ref_agents))  # [n_ref, n_env, n_act]
                        # self.log("{}, received".format(step))
                        # self.log("{}, {}, {}".format(action.shape, ref_strategies.shape, action.dtype))
                        # _action = action.squeeze().numpy().tolist()
                        # _dis = [np.choose(action, ref_strategies[_i]) for _i in range(len(ref_agents))]
                        # _dis = [ts(ref_strategies[_i][range(num_envs), _action]) for _i in range(len(ref_agents))]

                        # for ref_conn in ref_conns:
                        #     _dis.append(ref_conn.recv())
                        # self.log("123123")
                        # dis = torch.stack(_dis, dim=0).transpose(1, 0)
                        # # dis2 = torch.stack(_dis2, dim=0).transpose(1, 0)
                        # t_dis = (-torch.log(dis)).clamp(max=5000.)
                        # sum_likelihood += t_dis.sum(dim=0).numpy()
                        # old_sum_likelihood_capped = sum_likelihood_capped.copy()
                        # sum_likelihood_capped += t_dis.numpy() * decay
                        # decay *= args.likelihood_gamma
                        # if args.use_likelihood_reward_cap:
                        #     sum_likelihood_capped = sum_likelihood_capped.clip(max=likelihood_threshold)
                        # print(t_dis)

                        # _dis = [ref_agent.get_value(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                             rollouts.masks[step]).detach().squeeze()
                        #         for ref_agent in ref_agents]
                        # dis = torch.stack(_dis, dim=0).transpose(1, 0)
                        # t_dis = dis

                        # sum_dist_penalty += t_dis.sum().item()
                        # print(sum_likelihood_capped)
                        # _reward = torch.cat([reward, t_dis * 0.01, t_dis], dim=1)
                        # _reward[:, 0] += args.likelihood_alpha * (sum_likelihood_capped - old_sum_likelihood_capped).mean(axis=1)
                        _reward = torch.cat([normalized_reward, torch.zeros(num_envs, sample_n_ref * 3 + rnd_size)], dim=1)
                    ref_time += time.time() - st

                    # print(reward)
                    total_samples += 1
                    sum_reward += reward.sum().item()
                    # self.log("step {} - received {}, {}, {}".format(it * self.num_steps + step + 1, obs, reward, done))
                    # self.log("acquire act locks")
                    # acquire_all_locks(self.act_locks)

                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    # if step == episode_steps - 1:
                    #     self.log("{}, {}".format(step, done[0]))
                    # sum_likelihood_capped *= masks.numpy()
                    bad_masks = bad_mask
                    # if any(reward > 0.):
                    #     print(reward)
                    # print(_reward.size())
                    rollouts.insert(obs, original_obs, recurrent_hidden_states, action,
                                    action_log_prob.detach(), value.detach(), _reward, reward, masks, bad_masks)

                # available = np.zeros(num_envs, dtype=int)
                # self.log(sum_likelihood_capped)
                st = time.time()

                assert not args.reject_sampling
                keywords0 = ["original_obs", "obs", "actions", "action_log_probs", "value_preds", "rewards", "original_rewards"]
                keywords1 = ["recurrent_hidden_states", "masks", "bad_masks"]
                step = valid_rollouts.step
                # self.log(step)
                for k in keywords0:
                    valid_rollouts.__getattribute__(k)[step: step + episode_steps, :] = \
                        rollouts.__getattribute__(k)[:episode_steps, :]
                for k in keywords1:
                    valid_rollouts.__getattribute__(k)[step + 1: step + episode_steps + 1, :] = \
                        rollouts.__getattribute__(k)[1: episode_steps + 1, :]
                valid_rollouts.step += episode_steps

                if actual_use_ref:
                    obs = rollouts.original_obs[:episode_steps, :].reshape(episode_steps * num_envs, -1)
                    actions = rollouts.actions[:episode_steps, :].reshape(episode_steps * num_envs, -1)

                    if args.use_symmetry_for_reference:
                        new_obs = self.env.env.obs_to_all_symmetries_agent(obs.numpy(), self.agent_id)
                        new_actions = self.env.env.actions_to_all_symmetries_agent(actions.numpy(), self.agent_id)
                        num_sym = len(new_obs)
                        assert num_sym == len(new_actions) and num_sym == self.num_sym
                        obs = ts(np.concatenate(new_obs, axis=0))
                        actions = torch.LongTensor(np.concatenate(new_actions, axis=0))
                    else:
                        num_sym = 1

                    # _dis = [ref_agent.get_probs(obs, None, None, actions).detach() for ref_agent in ref_agents]
                    _dis = [ref_agent.get_likelihoods(obs, None, None, actions).detach() for ref_agent in ref_agents]
                    # _dis_self = actor_critic.get_probs(obs, None, None, actions).detach()
                    # _dis = [__dis / (__dis + dis_self) for __dis in _dis]
                    # print(_dis[0][])
                    # [ref, sym * step * env]
                    dis = torch.stack(_dis, dim=0).reshape(len(ref_agents), num_sym, episode_steps, num_envs)
                    # [ref, sym, step, env]
                    t_dis = -dis
                    # t_dis_self = -torch.log(_dis_self)
                    # print(t_dis)
                    t_dis *= args.likelihood_alpha
                    if args.likelihood_cap is not None:
                        t_dis = t_dis.clamp(max=args.likelihood_cap)
                    t_dis = t_dis.transpose(0, 2).transpose(1, 3).reshape(episode_steps, num_envs, len(ref_agents) * num_sym).detach().numpy()
                    t_dis_ret = np.zeros((num_envs, sample_n_ref))
                    for i in reversed(range(episode_steps)):
                        t_dis_ret = t_dis_ret * args.likelihood_gamma + t_dis[i]
                    # likelihood_rms.update(t_dis_ret)
                    likelihood_rms.update(t_dis.reshape(-1, sample_n_ref))
                    # normalized_t_dis = self._normalize_reward(t_dis, likelihood_rms, center=True)
                    # if args.likelihood_cap is not None:
                    #     normalized_t_dis = t_dis - args.likelihood_cap
                    # else:
                    normalized_t_dis = t_dis * 0.01
                    # print(normalized_t_dis.sum(axis=0)[0])
                    # print(normalized_t_dis)
                    # normalized_t_dis_ret = np.zeros((num_envs, sample_n_ref))
                    # for i in range(episode_steps):
                    #     normalized_t_dis_ret = normalized_t_dis_ret * args.likelihood_gamma + normalized_t_dis[i]
                    # print(t_dis)
                    # [step, env, ref * sym]
                    if args.use_rnd:
                        _, rnd_values, rnd_predictions = actor_critic.get_reward_prediction(obs, None, None, actions)
                        rnd_errors = (rnd_values.detach() - rnd_predictions.detach()).pow(2).sum(-1) / 2
                        rnd_error_ret = np.zeros(num_envs)
                        for i in reversed(range(episode_steps)):
                            rnd_error_ret = rnd_error_ret * args.gamma + rnd_errors[i].detach().numpy()
                        rnd_rms.update(rnd_error_ret)
                        rnd_errors /= np.sqrt(rnd_rms.var)
                    else:
                        rnd_errors = 0.
                    if args.use_reward_predictor:
                        with torch.no_grad():
                            next_value = actor_critic.get_value(
                                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                rollouts.masks[-1]).detach()
                        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.likelihood_gamma, args.use_proper_time_limits)
                        actual_returns = rollouts.returns[:episode_steps, :, :1]
                        actual_rewards = valid_rollouts.rewards[step: step + episode_steps, :, :1]
                        reward_predictions = [ref_agent.get_reward_prediction(obs, None, None, actions)[0].detach() for ref_agent in ref_agents]
                        reward_predictions = torch.stack(reward_predictions, dim=0).reshape(len(ref_agents), num_sym, episode_steps, num_envs)
                        reward_predictions = reward_predictions.transpose(0, 2).transpose(1, 3).reshape(episode_steps, num_envs, len(ref_agents) * num_sym)
                        # print(reward_predictions.size(), valid_rollouts.rewards[step: step + episode_steps, :, 0].size())
                        multiplier = 1. / args.reward_prediction_multiplier
                        reward_prediction_errors = (reward_predictions * multiplier - actual_rewards).square()
                        # reward_prediction_errors = (reward_predictions * 0.2 - actual_returns).square()
                        # self_reward_prediction = agent.actor_critic.get_reward_prediction(obs, None, None, actions)[0].detach()
                        # self_reward_prediction = self_reward_prediction.reshape(episode_steps, num_envs, 1)
                        # self_reward_prediction_error = (self_reward_prediction * multiplier - actual_rewards).square()
                        # self_reward_prediction_error = (self_reward_prediction * 0.2 - actual_returns).square()
                        # reward_prediction_reward = torch.minimum(reward_prediction_errors, self_reward_prediction_error)
                        reward_prediction_reward = reward_prediction_errors.detach()
                        if args.likelihood_cap is not None:
                            reward_prediction_reward = reward_prediction_reward.clamp(max=args.likelihood_cap)
                        reward_prediction_ret = np.zeros((num_envs, sample_n_ref))
                        for i in range(episode_steps):
                            reward_prediction_ret = reward_prediction_ret * args.gamma + reward_prediction_reward.numpy()[i]
                        reward_prediction_rms.update(reward_prediction_ret)
                        reward_prediction_reward /= np.maximum(reward_prediction_rms.mean, np.sqrt(reward_prediction_rms.var))
                            # print(reward_prediction_reward.sum(dim=0)[0])
                        # reward_prediction_reward = reward_prediction_reward * t_dis * 0.1
                        # t_dis = t_dis * torch.gt(reward_prediction_reward, 0.01)
                    else:
                        reward_prediction_reward = torch.zeros((episode_steps, num_envs, sample_n_ref))

                    # if args.use_reward_predictor:
                        # ijs = []
                        # for i in range(episode_steps):
                        #     for j in range(num_envs):
                        #         if valid_rollouts.rewards[step + i, j, 0] > 0.25 and valid_rollouts.rewards[step + i, j, 0] < 0.75:
                        #             print(valid_rollouts.rewards[step + i, j, 0], reward_prediction_reward[i, j, 0])
                        #             ijs.append((i, j))

                        # if any(valid_rollouts.rewards[step: step + episode_steps, 0, 0] > 0. and ):
                        #     print(valid_rollouts.rewards[step: step + episode_steps, 0, 0], reward_prediction_reward[:, 0, 0])
                        # for i in range(sample_n_ref):
                        #     valid_rollouts.rewards[step: step + episode_steps, :, 0] *= torch.minimum(reward_prediction_reward[:, :, i] * 10., torch.tensor(2.))
                        # for i, j in ijs:
                        #     print("after", valid_rollouts.rewards[step + i, j, 0])
                    # print(reward_prediction_reward)
                    if args.use_rnd:
                        valid_rollouts.rewards[step: step + episode_steps, :, 1] = rnd_errors
                    valid_rollouts.rewards[step: step + episode_steps, :, 1 + rnd_size: sample_n_ref + 1] = reward_prediction_reward
                    valid_rollouts.rewards[step: step + episode_steps, :, sample_n_ref + 1 + rnd_size: sample_n_ref * 2 + 1] = ts(normalized_t_dis)
                    valid_rollouts.rewards[step: step + episode_steps, :, sample_n_ref * 2 + 1 + rnd_size:] = ts(t_dis)

                    sum_likelihood += t_dis_ret.sum(axis=0)

                if not threshold_selected:
                    mean_likelihood = sum_likelihood / total_episodes
                    likelihood_threshold = mean_likelihood * args.auto_threshold
                    threshold_selected = True
                    self.log("selected threshold: {}".format(likelihood_threshold))

                for i_env in range(num_envs):
                    # self.log(args.reject_sampling)
                    # if not args.reject_sampling or all(sum_likelihood_capped[i_env] > likelihood_threshold):
                    #     # self.log(sum_likelihood_capped[i_env])
                    #     if ref:
                    #         tmp_likelihood.append(np.copy(sum_likelihood_capped[i_env]))
                    remain_episodes -= 1
                    if actual_use_ref:
                        sl = t_dis_ret[i_env]
                        accepted_episodes += int(all(sl > omega * likelihood_threshold))
                        accepted_episodes_ref += (sl > omega * likelihood_threshold)
                        # print(sl, likelihood_threshold)
                    else:
                        accepted_episodes += 1
                # print("sss:", t_dis_ret[0])
                copy_time += time.time() - st
                if actual_use_ref:
                    sum_likelihood_capped.fill(0.)

                # self.log("remain {}, {}".format(remain_episodes, remain_episodes <= 0))
                if args.reject_sampling:
                    self.main_conn.send(remain_episodes <= 0 or not self.train(it))
                    # self.log(remain_episodes)
                    if self.main_conn.recv():
                        break
                else:
                    release_all_locks(self.act_locks)
                    if remain_episodes <= 0:
                        break

            self.log("ref time {}, copy time {}".format(ref_time, copy_time))
            ref_time = 0.
            copy_time = 0.
            efficiency = accepted_episodes / total_episodes
            efficiency_ref = accepted_episodes_ref / total_episodes
            self.log("Total explored episodes: {}. Accepted episodes: {}. Efficiency: {:.2%}. Efficiency per ref: {}".
                     format(total_episodes, accepted_episodes, efficiency, efficiency_ref))

            # valid_rollouts.rewards[step: step + episode_steps, :, 1: 1 + sample_n_ref] *=

            report_data["total_episodes"] = total_episodes
            report_data["accepted_episodes"] = accepted_episodes
            report_data["efficiency"] = efficiency

            # print(valid_rollouts.returns[0, 0, 1:])
            # self.log("{}, {}".format(tmp_likelihood[0], valid_rollouts.returns[0, 0, 1:]))
            # diff = []
            # for i in range(num_episodes * num_envs):
            #     diff.append(np.absolute(tmp_likelihood[i] - valid_rollouts.returns[episode_steps * i, 0, 1:].numpy()).sum())
            # self.log(np.max(diff))
            # sum_dist_penalty += valid_rollouts.returns[:, :, 1:].mean().item()

            if args.plot_joint_plot:
                if self.agent_id == 0:
                    valid_rollouts.plot_joint_plot(episode_steps, self.agent_name)

            sampling_time = time.time() - sampling_st
            self.log("Sampling time: {}".format(sampling_time))
            report_data["sampling_time"] = sampling_time

            if self.train(it):
                report_data["train"] = 1
                st = time.time()

                if actual_use_ref and args.dipg:
                    average_mmd = 0.
                    average_q = 0.
                    trajectories = np.concatenate(trajectories, axis=0)
                    n, m = trajectories.shape
                    assert n == total_episodes
                    if dipg_g is None:
                        k = args.dipg_k
                        # dipg_g = np.random.randn(m, k) / np.sqrt(m * k)
                        dipg_g = np.identity(m)
                        dipg_g = dipg_g[:, np.random.choice(m, k, replace=False)]
                    tg = np.matmul(trajectories, dipg_g)
                    num_samples = args.dipg_num_samples
                    for i in range(n):
                        tpi = np.random.choice(n, min(n, num_samples), replace=False)
                        ker_p = dipg_ker(tg[i] - tg[tpi])
                        ker_q = np.zeros(len(ref_agents))
                        for j, ref_agent in enumerate(ref_agents):
                            tqi = np.random.choice(ref_agent.trajectories.shape[0], num_samples, replace=False)
                            tq = ref_agent.trajectories[tqi]
                            tqg = np.matmul(tq, dipg_g)
                            ker_q[j] = dipg_ker(tg[i] - tqg)
                        # mmd = ker_p - np.max(ker_q)
                        mmd = np.max(ker_q)
                        average_mmd += mmd / n
                        average_q += np.max(ker_q) / n
                        pos_i = i // num_envs
                        pos_j = i % num_envs
                        valid_rollouts.mmds[pos_i * episode_steps: (pos_i + 1) * episode_steps, pos_j, 0] = mmd
                    trajectories = []
                    self.log("average mmd: {}, p: {}, q: {}".format(average_mmd, average_mmd + average_q, average_q))

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        valid_rollouts.obs[-1], valid_rollouts.recurrent_hidden_states[-1],
                        valid_rollouts.masks[-1]).detach()

                valid_rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                               args.gae_lambda, args.likelihood_gamma, args.use_proper_time_limits)
                # print("computed:", valid_rollouts.returns[0, 0, 1 + sample_n_ref * 2:])
                if args.interpolate_rewards:
                    if args.no_exploration_rewards:
                        exploration_rewards = False
                    elif args.exploration_threshold is not None:
                        exploration_rewards = efficiency < args.exploration_threshold
                    else:
                        exploration_rewards = True
                    self.log(f"use exploration rewards? {exploration_rewards}")
                    report_data["use_exploration_rewards"] = int(exploration_rewards)
                    threshold = omega * torch.tensor(likelihood_threshold, dtype=torch.float)
                    # threshold = threshold.repeat_interleave(self.num_sym)
                    fine_tune = args.fine_tune_start_iter is not None and it >= args.fine_tune_start_iter

                    alphas = [1., 0., 1.]
                    use_filters = [True, False, True]

                    if args.use_reward_predictor and sample_n_ref > 0:
                        # sample_acceptance = valid_rollouts.returns[:, :, -sample_n_ref:]
                        # alphas[1] = args.prediction_reward_alpha * (1 - efficiency) * 2. / sample_n_ref
                        if args.use_dynamic_prediction_alpha:
                            # alphas[1] = torch.maximum(torch.tensor(1 - efficiency_ref * (1. / 1.0)), torch.tensor(0.0))
                            alphas[1] = torch.tensor(1 - efficiency_ref)
                            # topk = torch.topk(alphas[1], k=min(sample_n_ref, 3))
                            # topk_mask = torch.zeros(sample_n_ref)
                            # topk_mask[topk.indices] = 1.
                            # print(alphas[1], topk_mask)
                            # alphas[1] *= topk_mask
                        else:
                            alphas[1] = 1.
                        alphas[1] *= args.prediction_reward_alpha
                        # alphas[1] *= 2. / sample_n_ref
                        use_filters[1] = False

                    if sample_n_ref > 0:
                        if args.use_dynamic_prediction_alpha:
                            alphas[2] = torch.tensor(1 - efficiency_ref)
                            use_filters[2] = False
                        else:
                            alphas[2] = 1.
                            # alphas[2] = torch.tensor(efficiency_ref < 0.9, dtype=torch.float32)
                        alphas[2] *= args.exploration_reward_alpha

                    # if not exploration_rewards:
                    #     alphas[2] = 0.

                    if fine_tune:
                        alphas = [1., 0., 0.]
                        use_filters = [False, False, False]

                    if args.full_intrinsic:
                        alphas = [1., args.prediction_reward_alpha, args.exploration_reward_alpha]
                        use_filters = [False, False, False]

                    valid_rollouts.compute_interpolate_masks(threshold, alphas, use_filters, args.use_rnd, args.rnd_alpha)
                # self.log("ready to update")
                # valid_rollouts.action_loss_coef = 0.0 if efficiency < 0.1 else 1.0
                # self.log("before update")
                value_loss, action_loss, rnd_loss, dist_entropy, grad_norm, reward_prediction_loss = agent.update(valid_rollouts)
                update_time = time.time() - st
                self.log("update time {}".format(update_time))
                report_data["update_time"] = update_time
                self.log("Update #{}, reward {}, likelihood {}, value_loss {}, action_loss {}, rnd_loss {}, dist_entropy {}, grad_norm {}, reward_prediction_loss {}"
                         .format(it, sum_reward / total_episodes,
                                 sum_likelihood / total_episodes,
                                 value_loss, action_loss, rnd_loss, dist_entropy, grad_norm, reward_prediction_loss))
                episode_rewards.append(sum_reward / total_episodes)
                if len(episode_rewards) > 1:
                    self.log("mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}".format(
                        np.mean(episode_rewards), np.median(episode_rewards),
                        np.min(episode_rewards), np.max(episode_rewards)
                    ))
                statistics["value_loss"].append((it, value_loss))
                statistics["action_loss"].append((it, action_loss))
                statistics["rnd_loss"].append((it, rnd_loss))
                statistics["dist_entropy"].append((it, dist_entropy))
                statistics["grad_norm"].append((it, grad_norm))
                statistics["reward_prediction_loss"].append((it, reward_prediction_loss))
                report_data["value_loss"] = value_loss
                report_data["action_loss"] = action_loss
                report_data["dist_entropy"] = dist_entropy
                report_data["grad_norm"] = grad_norm
                if args.use_reward_predictor:
                    report_data["reward_prediction_loss"] = reward_prediction_loss
                update_counter += 1
                if self.save_interval > 0 and update_counter % self.save_interval == 0:
                    current_save_dir = mkdir2(self.save_dir, "update-{}".format(update_counter))
                    torch.save((actor_critic.state_dict(), self.obs_rms),
                               os.path.join(current_save_dir, "model.obj"))
            else:
                report_data["train"] = 0

            valid_rollouts.step = 0
            statistics["reward"].append((it, sum_reward / total_episodes))
            report_data["reward"] = sum_reward / total_episodes
            if ref:
                statistics["dist_penalty"].append((it, sum_dist_penalty))
                statistics["likelihood"].append((it, sum_likelihood / total_episodes))
                statistics["total_episodes"].append((it, total_episodes))
                statistics["accepted_episodes"].append((it, accepted_episodes))
                statistics["efficiency"].append((it, accepted_episodes / total_episodes))
                report_data["dist_penalty"] = sum_dist_penalty
                if actual_use_ref:
                    for i in range(sample_n_ref):
                        report_data["likelihood-{}".format(i)] = sum_likelihood[i] / total_episodes
            # last_update_iter = it
            sum_reward = 0.
            sum_dist_penalty = 0.
            sum_likelihood.fill(0.)

            valid_rollouts.after_update()

            joblib.dump(statistics, os.path.join(self.save_dir, "statistics.obj"))

            if self.data_queue is not None:
                # self.log("data put!")
                self.data_queue.put(report_data)

        if self.data_queue is not None:
            self.data_queue.put(None)

        # for ref_conn in ref_conns:
        #     ref_conn.send(None)
        #
        # for ref in ref_processes:
        #     ref.join()

        obs_rms = self.obs_rms
        if actor_critic.obs_rms is not None:
            obs_rms = actor_critic.obs_rms
        torch.save((actor_critic.state_dict(), obs_rms), os.path.join(self.save_dir, "model.obj"))

        if args.collect_trajectories:
            trajectories = np.concatenate(trajectories, axis=0)
            np.save(os.path.join(self.save_dir, "trajectories"), trajectories)

        recurrent_hidden_states = torch.zeros((1, actor_critic.recurrent_hidden_state_size))
        self.main_conn.send(statistics)
        # print(recurrent_hidden_states)
        if actual_use_ref:
            neg_likelihoods = np.zeros(len(ref_agents))
        else:
            neg_likelihoods = None
        command = self.main_conn.recv()
        if command is not None:
            verbose = command
            while True:
                command = self.main_conn.recv()
                # st = time.time()
                if command is None:
                    break
                obs, done = command
                if self.norm_obs:
                    obs = self.normalize_obs(obs, update=False)
                # _, action, action_log_prob, recurrent_hidden_states = actor_critic.act(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]]))
                # if obs[2] != 0 or obs[3] != 0:
                #     # if obs[0] > 0:
                #     #     action = 2
                #     # elif obs[1] > 0:
                #     #     action = 0
                #     # else:
                #     #     action = 4
                #     if obs[0] < 2:
                #         action = 3
                #     elif obs[0] > 2:
                #         action = 2
                #     elif obs[1] < 2:
                #         action = 1
                #     elif obs[1] > 2:
                #         action = 0
                #     else:
                #         action = 4
                #     # if abs(obs[2]) > 0:
                #     #     if obs[2] < -1:
                #     #         action = 2
                #     #     elif obs[2] == -1:
                #     #         action = 4
                #     #     else:
                #     #         action = 3
                #     # elif abs(obs[3]) > 0:
                #     #     if obs[3] < -1:
                #     #         action = 0
                #     #     elif obs[3] == -1:
                #     #         action = 4
                #     #     else:
                #     #         action = 1
                # else:
                #     # action = 4
                #     if abs(obs[4]) + abs(obs[5]) > 1:
                #         # if obs[4] <= -1 and obs[0] >= 2:
                #         #     action = 2
                #         # elif obs[4] >= 1 and obs[0] <= 2:
                #         #     action = 3
                #         # elif obs[5] <= -1 and obs[1] >= 2:
                #         #     action = 0
                #         # elif obs[5] >= 1 and obs[1] <= 2:
                #         #     action = 1
                #         # else:
                #         #     action = 4
                #
                #         if abs(obs[4]) > abs(obs[5]) or obs[4] == obs[5] and np_random.rand() < 0.5:
                #             if obs[4] <= -1:
                #                 action = 2
                #             else:
                #                 action = 3
                #         else:
                #             if obs[5] <= -1:
                #                 action = 0
                #             else:
                #                 action = 1
                #     else:
                #         action = 4
                strategy = actor_critic.get_strategy(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]]))
                strategy.likelihoods(strategy.sample())
                # print(strategy)
                if isinstance(strategy, FixedCategorical):
                    strategy = FixedCategorical(logits=strategy.logits[0].detach())
                    action = strategy.sample().item()
                elif isinstance(strategy, FixedNormal):
                    strategy = FixedNormal(loc=strategy.loc[0].detach(), scale=strategy.scale[0].detach())
                    action = strategy.mode().numpy()
                else:
                    raise NotImplementedError
                # c = 0.0
                # strategy = (1 - c) * strategy + c * np.ones_like(strategy) / strategy.shape[0]
                # action = np_random.choice(strategy.shape[0], p=strategy)
                # action = strategy.sample().numpy()
                # print(strategy.loc, strategy.scale)
                # print(action, torch.log(actor_critic.get_probs(ts([obs]), recurrent_hidden_states, ts([[1.0]]), ts([action + 0.2]))))
                # action = np.argmax(strategy)
                if verbose:
                    np.set_printoptions(precision=3, suppress=True)
                    # print(self.name, strategy, strategy[action])
                    # if ref_agents is not None:
                    #     ref_strats = [ref_agent.get_strategy(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]])).detach().numpy() for ref_agent in ref_agents]
                    #     ref_strats = np.array(ref_strats)
                    #     print("Mean strat:", ref_strats.mean(axis=0))
                    #     ref_probs = [ref_agent.get_probs(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]]),
                    #                                      ts([action])).detach().squeeze().item() for ref_agent in ref_agents]
                    #     neg_likelihoods += -np.log(np.array(ref_probs))
                    #     print(neg_likelihoods)
                    # print(ref_probs[8], ref_probs[9])
                    # print(np.exp(-neg_likelihoods) / np.exp(-neg_likelihoods).sum())
                    # ref_16_strategy = ref_agents[16].get_strategy(ts([obs]), recurrent_hidden_states,
                    #                                               ts([[0.0] if done else [1.0]])).detach().squeeze().numpy()
                    # ref_17_strategy = ref_agents[17].get_strategy(ts([obs]), recurrent_hidden_states,
                    #                                               ts([[0.0] if done else [1.0]])).detach().squeeze().numpy()
                    # ref_18_strategy = ref_agents[18].get_strategy(ts([obs]), recurrent_hidden_states,
                    #                                               ts([[0.0] if done else [1.0]])).detach().squeeze().numpy()
                    # ref_19_strategy = ref_agents[19].get_strategy(ts([obs]), recurrent_hidden_states,
                    #                                               ts([[0.0] if done else [1.0]])).detach().squeeze().numpy()
                    # print(self.name, 16, ref_16_strategy, ref_16_strategy[action])
                    # print(self.name, 17, ref_17_strategy, ref_17_strategy[action])
                    # print(self.name, 18, ref_18_strategy, ref_18_strategy[action])
                    # print(self.name, 19, ref_17_strategy, ref_19_strategy[action])
                # print(time.time() - st)
                self.main_conn.send((action, strategy))
