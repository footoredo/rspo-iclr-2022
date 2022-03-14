import os
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from rspo.multi_agent.utils import jointplot, mkdir2


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, num_refs=0, num_value_refs=0, use_rnd=False):
        self.original_obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.original_rewards = torch.zeros(num_steps, num_processes, 1)
        self.rnd = 1 if use_rnd else 0
        self.rewards = torch.zeros(num_steps, num_processes, 1 + num_refs * 3 + self.rnd)
        self.num_refs = num_refs
        self.num_value_refs = num_value_refs
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1 + num_value_refs * 2 + self.rnd)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1 + num_refs * 3 + self.rnd)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.dice_deps = torch.zeros(num_steps, num_processes, 1)
        self.interpolate_masks = torch.zeros(num_steps, num_processes, num_refs * 2 + 1 + self.rnd)
        self.mmds = torch.zeros(num_steps, num_processes, 1)
        self.action_loss_coef = 1.0
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.original_obs = self.original_obs.to(device)
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.original_rewards = self.original_rewards.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.dice_deps = self.dice_deps.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.mmds = self.bad_masks.to(device)

    def insert(self, obs, original_obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, original_rewards, masks, bad_masks):
        self.original_obs[self.step + 1].copy_(original_obs)
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        # print(action_log_probs)
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step].copy_(value_preds)
        # print(self.rewards[self.step], rewards)
        # print(self.rewards[self.step].size(), rewards.size())
        self.rewards[self.step].copy_(rewards)
        self.original_rewards[self.step].copy_(original_rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.original_obs[0].copy_(self.original_obs[-1])
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def plot_joint_plot(self, episode_steps, agent_name):
        num_refs = self.num_refs
        num_episodes = self.rewards.size()[0] // episode_steps * self.rewards.size()[1]
        returns = np.zeros(num_episodes)
        likelihoods = np.zeros((num_episodes, self.num_refs))
        episode_cnt = 0
        for i in range(self.rewards.size()[0] // episode_steps):
            for j in range(self.rewards.size()[1]):
                for k in range(episode_steps):
                    returns[episode_cnt] += self.rewards[i * episode_steps + k, j, 0].item()
                    likelihoods[episode_cnt] += self.rewards[i * episode_steps + k, j,
                                                1 + num_refs * 2:].detach().numpy()
                episode_cnt += 1

        for cni in range(num_refs):
            if cni < 2:
                continue
            # save_dir = mkdir2(self.save_dir, "ref-{}".format(cni))
            # save_path = os.path.join(save_dir, "{}.png".format(self.cnt))
            # save_path = None
            # jointplot(cn[si][:, cni], return_batch[:, 0].numpy()[si],
            #           save_path=save_path, title="agent-{} ref-{}".format(self.agent_name, cni))
            jointplot(likelihoods[:, cni], returns,
                      save_path=None, title="agent-{} ref-{}".format(agent_name, cni))

    # Call this after compute_returns
    def compute_interpolate_masks(self, thresholds: torch.Tensor, alphas, use_filters, use_rnd, rnd_alpha, full_intrinsic=False):
        # print(use_reward)
        shape = list(self.rewards.size()[:2])
        if self.num_refs == 0:
            masks = [alphas[0] * torch.ones(shape + [1])]
            if use_rnd:
                masks.append(rnd_alpha * torch.ones(shape + [1]))
            interpolate_masks = torch.cat(masks, dim=2)
        else:
            if not any(use_filters):
                extrinsic_mask = torch.ones(shape + [1])
                prediction_masks = torch.ones(shape + [self.num_refs])
                exploration_masks = torch.ones(shape + [self.num_refs])
            else:
                likelihoods = self.returns[:-1, :, 1 + self.rnd + self.num_refs * 2:]
                failed_mask = torch.gt(thresholds, likelihoods)
                if use_filters[0]:
                    extrinsic_mask = 1. - (torch.any(failed_mask, dim=2, keepdim=True)).float()
                else:
                    extrinsic_mask = torch.ones(shape + [1])
                if use_filters[1]:
                    prediction_masks = 1. - failed_mask.float()
                else:
                    prediction_masks = torch.ones(shape + [self.num_refs])
                if use_filters[2]:
                    exploration_masks = failed_mask.float()
                else:
                    exploration_masks = torch.ones(shape + [self.num_refs])

            # print(extrinsic_mask.size(), prediction_masks.size(), exploration_masks.size())
            # print(self.num_refs, extrinsic_mask.size(), prediction_masks.size(), exploration_masks.size(), failed_mask.size(), self.returns.size())
            masks = [alphas[0] * extrinsic_mask]
            if use_rnd:
                masks.append(rnd_alpha * torch.ones(shape + [1]))
            masks += [alphas[1] * prediction_masks, alphas[2] * exploration_masks]
            print("eff:", extrinsic_mask.mean())
            interpolate_masks = torch.cat(masks, dim=2)
        # print(interpolate_masks.size(), interpolate_masks[0, 0], self.rewards[0, 0])
        # print(interpolate_masks.size(), self.interpolate_masks.size())
        # print(interpolate_masks)
        self.interpolate_masks.copy_(interpolate_masks)

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        likelihood_gamma,
                        use_proper_time_limits=True):
        # assert not use_proper_time_limits, "Not Implemented"
        gamma2 = likelihood_gamma
        num_refs = self.num_refs

        rnd = self.rnd

        if False:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                                          gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                num_value_refs = self.num_value_refs
                # assert num_refs == num_value_refs
                # print(self.value_preds)
                # print(self.rewards.size(0))
                for step in reversed(range(self.rewards.size(0))):
                    # print(self.rewards.size(), self.value_preds.size())
                    if num_refs == num_value_refs:
                        delta = self.rewards[step, :, :1 + rnd + num_refs * 2] + \
                                gamma * self.value_preds[step + 1, :, :1 + rnd + num_refs * 2] * self.masks[step + 1] - \
                                self.value_preds[step, :, :1 + rnd + num_refs * 2]
                    else:
                        delta = self.rewards[step, :, :1 + rnd + num_refs * 2]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    if use_proper_time_limits:
                        gae = gae * self.bad_masks[step + 1]
                    if num_refs == num_value_refs:
                        self.returns[step, :, :1 + rnd + num_refs * 2] = gae + self.value_preds[step]
                    else:
                        self.returns[step, :, :1 + rnd + num_refs * 2] = gae

                    # delta = self.rewards[step, :, :1] + \
                    #         gamma * self.value_preds[step + 1, :, :1] * self.masks[step + 1] - \
                    #         self.value_preds[step, :, :1]
                    # gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    # self.returns[step, :, :1] = gae + self.value_preds[step, :, :1]
                    # self.returns[step, :, 1:1 + num_refs] = self.rewards[step, :, 1:1 + num_refs]

                    self.returns[step, :, 1 + rnd + num_refs * 2:] = \
                        gamma2 * torch.mul(self.returns[step + 1, :, 1 + rnd + num_refs * 2:], self.masks[step + 1]) + \
                        self.rewards[step, :, 1 + rnd + num_refs * 2:]
            else:
                if use_proper_time_limits:
                    raise NotImplementedError
                # print("GAEGEAGEAGEAEG")
                # gamma = torch.tensor([gamma] + [0.] * (self.reward_dim - 1), dtype=torch.float)
                # self.returns[-1] = next_value
                self.returns[-1].zero_()
                if num_refs == self.num_value_refs:
                    self.returns[-1, :, :1 + rnd + num_refs * 2] = next_value[:, :1 + rnd + num_refs * 2]
                else:
                    self.returns[-1, :, :1 + rnd] = next_value[:, :1 + rnd]
                # print(next_value, self.rewards[-1])
                # print(self.rewards.size(0))
                for step in reversed(range(self.rewards.size()[0])):
                    # print(self.rewards[step], self.masks[step + 1])
                    # print(self.returns[step + 1].size())
                    # print(self.rewards[step].size())
                    # print(self.rewards[step, :, 1:].size())
                    # print(self.returns[step, :, 0].size())
                    # print(torch.square(self.rewards[step, :, 1:] - self.returns[step, :, 0]).size())
                    # print(torch.mul(self.returns[step + 1, :, 1:], self.masks[step + 1]).size())
                    # print(torch.mul(self.returns[step + 1], self.masks[step + 1]).size())
                    self.returns[step, :, :1 + rnd+ num_refs * 2] = \
                        gamma * torch.mul(self.returns[step + 1, :, :1 + rnd + num_refs * 2], self.masks[step + 1]) + \
                        self.rewards[step, :, :1 + rnd + num_refs * 2]
                    self.returns[step, :, 1 + rnd + num_refs * 2:] = \
                        gamma2 * torch.mul(self.returns[step + 1, :, 1 + rnd + num_refs * 2:], self.masks[step + 1]) + \
                        self.rewards[step, :, 1 + rnd + num_refs * 2:]
                    # self.returns[step, :, 1:] = torch.square(
                    #     self.rewards[step, :, 1:] - self.returns[step, :, :1]) + gamma2 * torch.mul(
                    #     self.returns[step + 1, :, 1:], self.masks[step + 1])
                    # if any(self.rewards[step] > 0.):
                    #     print(self.rewards[step])
                    #     print(self.returns[step])
            if num_refs > 0:
                for step in range(1, self.rewards.size()[0]):
                    self.returns[step, :, 1 + rnd + num_refs * 2:] = \
                        self.returns[step - 1, :, 1 + rnd + num_refs * 2:] * self.masks[step] + \
                        self.returns[step, :, 1 + rnd + num_refs * 2:] * (1 - self.masks[step])
        # if dice_lambda is not None:
        #     self.compute_dice_deps(dice_lambda)

    # def compute_dice_deps(self, dice_lambda):
    #     weighted_cumsum = self.action_log_probs[0]
    #     # print(self.action_log_probs[0])
    #     for t in range(self.action_log_probs.size()[0]):
    #         if t > 0:
    #             weighted_cumsum = dice_lambda * weighted_cumsum * self.masks[t] + self.action_log_probs[t]
    #         deps_exclusive = weighted_cumsum - self.action_log_probs[t]
    #         self.dice_deps[t] = magic_box(weighted_cumsum) - magic_box(deps_exclusive)
    #
    #     # print(advantages.size(), full_deps.size())

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               episode_steps=1):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        assert mini_batch_size % episode_steps == 0

        def step_view(data):
            data_shape = data.size()[2:]
            # print(data.size(), mini_batch_size)
            # print(data_shape)
            # print(mini_batch_size, batch_size, num_mini_batch)
            if episode_steps > 1:
                return data[:num_steps].reshape(-1, episode_steps, num_processes, *data_shape).transpose(1, 2). \
                    reshape(-1, episode_steps, *data_shape)
            else:
                return data[:num_steps].view(-1, *data_shape)

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size // episode_steps)),
            mini_batch_size // episode_steps,
            drop_last=True)

        obs_step = step_view(self.obs)
        recurrent_hidden_states = step_view(self.recurrent_hidden_states)
        actions = step_view(self.actions)
        value_preds = step_view(self.value_preds)
        returns = step_view(self.returns)
        masks = step_view(self.masks)
        # print(episode_steps, masks.size())
        action_log_probs = step_view(self.action_log_probs)
        interpolate_masks = step_view(self.interpolate_masks)
        rewards = step_view(self.original_rewards)
        mmds = step_view(self.mmds)

        if advantages is not None:
            advantages = step_view(advantages)

        while True:
            for indices in sampler:
                obs_batch = obs_step[indices]
                recurrent_hidden_states_batch = recurrent_hidden_states[indices]
                actions_batch = actions[indices]
                value_preds_batch = value_preds[indices]
                return_batch = returns[indices]
                masks_batch = masks[indices]
                old_action_log_probs_batch = action_log_probs[indices]
                interpolate_masks_batch = interpolate_masks[indices]
                rewards_batch = rewards[indices]
                mmds_batch = mmds[indices]

                if advantages is None:
                    adv_targ = None
                else:
                    adv_targ = advantages[indices]

                yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                      masks_batch, old_action_log_probs_batch, adv_targ, interpolate_masks_batch, rewards_batch, \
                      mmds_batch

    def recurrent_generator(self, advantages, num_mini_batch, episode_steps=1):
        assert episode_steps == 1
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            interpolate_masks_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                interpolate_masks_batch.append(self.interpolate_masks[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            interpolate_masks_batch = torch.stack(interpolate_masks_batch, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                  old_action_log_probs_batch, adv_targ, interpolate_masks_batch
