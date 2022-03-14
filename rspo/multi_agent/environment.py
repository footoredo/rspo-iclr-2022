import gym
import numpy as np

import multiprocessing as mp

from argparse import Namespace

from .utils import *
from stable_baselines3.common.running_mean_std import RunningMeanStd

import copy


class Environment(mp.Process):
    def __init__(self, env_id, logger, args: Namespace, env, agents, act_sizes, act_recover_fns, main_conn, obs_shm,
                 act_shm, obs_locks, act_locks, norm_obs=False):
        super(Environment, self).__init__()

        self.env_id = env_id
        self.logger = logger

        self.args = args
        self.seed = reseed(args.seed, "env-{}".format(self.env_id))
        self.num_steps = args.num_steps
        self.num_envs = args.num_processes
        self.num_agents = args.num_agents
        self.episode_steps = args.episode_steps
        self.reward_norm = args.reward_normalization
        self.batch_size = self.num_steps * self.num_envs
        self.num_env_steps = args.num_env_steps // args.num_processes
        self.reseed_step = args.reseed_step // args.num_processes if args.reseed_step is not None else None
        self.reseed_z = args.reseed_z
        self.ref = args.use_reference
        self.dtype = np.float32
        self.act_sizes = act_sizes

        self.env = env
        self.agents = agents

        self.main_conn = main_conn
        self.obs_shm = obs_shm
        self.act_shm = act_shm
        self.obs_locks = obs_locks
        self.act_locks = act_locks
        self.act_recover_fns = act_recover_fns

        self.norm_obs = norm_obs
        self.obs_rms = dict()
        self.epsilon = 1e-8
        self.clip_obs = 10.

        self.np_random = np.random.RandomState(seed=self.seed)

    def reseed(self, step, z):
        _seed = None
        for iz in range(z):
            _seed = self.np_random.tomaxint() & ((1 << 32) - 1)
            self.env.seed(_seed)
        self.log("reseed with seed {}".format(_seed))
        return _seed

    def log(self, msg):
        self.logger("Environment-{}: {}".format(self.env_id, msg))

    def write(self, place, obs, reward, normalized_reward, done, bad_mask):
        obs = np.array(obs, dtype=self.dtype)
        # self.log("obs len {}".format(obs.shape[0]))
        # print(obs.shape[0], len(place))
        np.copyto(place[:obs.shape[0]], obs)
        place[obs.shape[0]] = float(reward)
        place[obs.shape[0] + 1] = float(normalized_reward)
        place[obs.shape[0] + 2] = float(done)
        place[obs.shape[0] + 3] = float(bad_mask)

    def _normalize_obs(self, obs, obs_rms):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_obs(self, obs):
        new_obs = dict()
        for agent in self.agents:
            obs_ = np.array([copy.deepcopy(obs[agent])])
            obs_rms = self.obs_rms[agent]
            if self.norm_obs:
                obs_rms.update(obs_)
                obs_ = self._normalize_obs(obs_, obs_rms)
            new_obs[agent] = obs_[0]
        # print(obs, new_obs)
        return new_obs

    def run(self):
        if self.num_env_steps == 0:
            return
        env = self.env
        args = self.args
        # ref = self.ref
        ref = False
        reset_every = args.num_steps
        # print(env.seed)
        env.seed(self.seed)
        # acquire_all_locks(self.obs_locks)

        reward_filters = {agent: Identity() for agent in self.agents}
        if self.reward_norm:
            reward_filters = {agent: RewardFilter(reward_filters[agent], shape=(), gamma=args.gamma, clip=False)
                              for agent in self.agents}

        self.obs_rms = {agent: RunningMeanStd(shape=env.observation_spaces[agent].shape) for agent in self.agents}

        if self.reseed_step is not None and 0 >= self.reseed_step:
            self.reseed(0, self.reseed_z)

        last_seed = None
        if ref:
            last_seed = self.reseed(0, 1)
            self.np_random.seed(last_seed)

        init_obs = env.reset()
        init_obs = self.normalize_obs(init_obs)
        # self.log(init_obs)

        obs_places = []
        obs_lens = []
        offset = 0
        item_size = np.zeros(1, dtype=self.dtype).nbytes
        actions = dict()
        infos = dict()

        num_episodes = self.num_env_steps // args.episode_steps

        for agent in self.agents:
            actions[agent] = 0

            obs_space = self.env.observation_spaces[agent]
            assert isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1
            obs_len = obs_space.shape[0]
            full_len = obs_len + 4  # reward, normalized reward, done, bad mask
            place = np.frombuffer(self.obs_shm.buf[offset + item_size * full_len * self.env_id:
                                                   offset + item_size * full_len * (self.env_id + 1)],
                                  dtype=self.dtype)
            obs_places.append(place)
            obs_lens.append(obs_len)
            self.write(place, init_obs[agent], 0., 0., 0., 0.)
            # np.copyto(place[:obs_len], init_obs[agent])
            # self.log("#{} - obs for {}: {}".format(0, agent, init_obs[agent]))
            offset += item_size * full_len * self.num_envs

        # release_all_locks(self.obs_locks)

        self.main_conn.recv()
        done = False
        step = 0
        finished_episodes = 0
        while True:
            # self.log(step)
            self.np_random.tomaxint()  # flush state for 1 step
            release_all_locks(self.obs_locks)

            if self.reseed_step is not None and step + 1 == self.reseed_step:
                self.reseed(step + 1, self.reseed_z)

            if done:
                # self.log("done")
                # acquire_all_locks(self.act_locks)
                if args.reject_sampling:
                    if self.main_conn.recv():
                        break
                else:
                    acquire_all_locks(self.act_locks)
                    finished_episodes += 1
                    # self.log(finished_episodes)
                    if finished_episodes >= num_episodes:
                        break
                obs = env.reset()
                obs = self.normalize_obs(obs)
                for agent in self.agents:
                    reward_filters[agent].reset()
                rewards = {agent: 0. for agent in self.agents}
                dones = {agent: False for agent in self.agents}
                # self.log([(self.obs_rms[agent].mean, self.obs_rms[agent].var) for agent in self.agents])
            else:
                # self.log(len(self.act_locks))
                acquire_all_locks(self.act_locks)
                act_pos = self.env_id * sum(self.act_sizes) * item_size
                for i, agent in enumerate(self.agents):
                    _action = copy.deepcopy(np.frombuffer(self.act_shm.buf[act_pos: act_pos + self.act_sizes[i] * item_size],
                                            dtype=self.dtype))
                    actions[agent] = self.act_recover_fns[i](_action)
                    # self.log(_action)
                    act_pos += self.act_sizes[i] * item_size
                    # print(np.isnan(actions[agent]))
                    # self.log("step {} from {} - act {}".format(step, agent, actions[agent]))
                obs, rewards, dones, infos = env.step(actions)
                obs = self.normalize_obs(obs)
            # release_all_locks(self.act_locks)
            # acquire_all_locks(self.obs_locks)

            # if ref and (step + 1) % reset_every == 0:
            #     c = (step + 1) // reset_every
            #     if c % 2 == 0:
            #         last_seed = self.reseed(step + 1, 1)
            #         self.np_random.seed(last_seed)
            #     else:
            #         self.env.seed(last_seed)

            # self.log("to write")
            not_done = False
            # self.log("step {}, done {}".format(step, dones))
            for i, agent in enumerate(self.agents):
                # self.log("{}, {}".format(i, agent))
                # self.log("step {} - obs for {}: {}, {}, {}".format(i + 1, agent, obs[agent], rewards[agent], dones[agent]))
                # print(infos, type(agent))
                bad_mask = 0.0 if type(infos[agent]) is dict and 'bad_transition' in infos[agent].keys() else 1.0
                self.write(obs_places[i], obs[agent], rewards[agent], reward_filters[agent](rewards[agent]), dones[agent], bad_mask)
                not_done = not_done or not dones[agent]
            done = not not_done

            # if self.env_id == 0:
            #     self.log("step: {}, done: {}".format(step, done))

            step += 1

        release_all_locks(self.obs_locks)
