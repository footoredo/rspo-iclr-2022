import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from rspo.multi_agent.utils import flat_view, net_add, ggrad, tsne, cluster, barred_argmax, \
    extract_trajectory, displot, jointplot, mkdir2


D_MAP = "SLRDU"
DX = [-1., 1., 0., 0.]
DY = [0., 0., -1., 1.]


class PPO():
    def __init__(self,
                 agent_name,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_grad_norm=True,
                 use_clipped_value_loss=True,
                 task=None,
                 direction=None,
                 save_dir=None,
                 args=None,
                 is_ref=False):

        self.agent_name = agent_name
        self.actor_critic = actor_critic

        self.ref = args.ppo_use_reference
        self.ref_agent = None

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.reward_prediction_loss_coef = args.reward_prediction_loss_coef
        # print(self.reward_prediction_loss_coef)

        self.max_grad_norm = max_grad_norm
        self.clip_grad_norm = clip_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.task = task
        self.direction = direction

        self.is_ref = is_ref

        if is_ref:
            self.optimizer = None
        else:
            # self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, betas=(0.5, 0.999))
            self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
            self.reward_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.cnt = 0
        self.args = args

        self.save_dir = mkdir2(save_dir, "ppo") if save_dir is not None and not is_ref else None

        # if not is_ref:
        #     print("PPO")

    def update(self, rollouts):
        # st = time.time()
        self.cnt += 1
        # advantages = rollouts.returns[:-1]
        num_refs = rollouts.num_refs
        num_value_refs = rollouts.num_value_refs
        rnd = 1 if self.args.use_rnd else 0
        if num_refs == num_value_refs:
            advantages = rollouts.returns[:-1, :, :1 + rnd + num_refs * 2] - rollouts.value_preds[:-1]
        else:
            advantages = rollouts.returns[:-1, :, :1 + rnd] - rollouts.value_preds[:-1, :, :1 + rnd]
        # print(advantages.size())
        # print(advantages.mean(dim=[0, 1], keepdims=True).size())
        advantages = (advantages - advantages.mean(dim=[0, 1], keepdims=True)) / (
            advantages.std(dim=[0, 1], keepdims=True) + 1e-5)  # [batch, step, n]
        # advantages[:, :, 0] = (advantages[:, :, 0] - advantages[:, :, 0].mean(dim=[0, 1], keepdims=True)) / (
        #     advantages[:, :, 0].std(dim=[0, 1], keepdims=True) + 1e-5)  # [batch, step, n]
        # advantages[:, :, 1:] = (advantages[:, :, 1:] - advantages[:, :, 0].mean(dim=[0, 1], keepdims=True)) / (
        #     advantages[:, :, 1:].std(dim=[0, 1], keepdims=True) + 1e-5)  # [batch, step, n]

        # def down_sample(_o, _a):
        #     # b = int(1e9 + 7)
        #     h = 0
        #     for x in _o.numpy()[1:]:
        #         ix = int(x > 0.)
        #         h = h * 2 + ix
        #     h = h * 5 + _a.item()
        #     return h
        #
        # trajectories = extract_trajectory(rollouts)
        # print(len(trajectories), len(trajectories[0]))
        # hash_set = set()
        # found_cnt = [0, 0]
        #
        # S = 8
        # SS = S * (2 ** 6) * 5
        #
        # exp_rew = np.zeros((S * 5, S * 5))
        # # vis_cnt = np.zeros((S * 5, S * 5), dtype=np.int)
        # vis_cnt = 0
        #
        # for trajectory in trajectories:
        #     found = -1
        #     for step, data in enumerate(trajectory):
        #         obs, act, rew = data
        #         h = down_sample(obs, act)
        #         if step < S and h not in hash_set:
        #             hash_set.add(h)
        #         if rew > 0.:
        #             if rew > 1.5:
        #                 found = 1
        #             else:
        #                 found = 0
        #             # break
        #         for i, d1 in enumerate(trajectory[:min(S, step)]):
        #             o1, a1, _ = d1
        #             h1 = down_sample(o1, a1)
        #             for j, d2 in enumerate(trajectory[:min(S, step)]):
        #                 o2, a2, _ = d2
        #                 h2 = down_sample(o2, a2)
        #                 exp_rew[h1, h2] += rew
        #                 # vis_cnt[h1, h2] += 1
        #                 vis_cnt += 1

        # exp_rew /= vis_cnt
        # w, v = np.linalg.eigh(exp_rew)
        # for i in range(w.shape[0]):
        #     print(w[i])
        #     for j in range(S):
        #         vs = np.zeros(5)
        #         for a in range(5):
        #             h = j * 5 + a
        #             vs[a] = v[h, i] / 0.1
        #         # print(np.exp(vs) / np.exp(vs).sum(), end='->')
        #         print(D_MAP[np.argmax(vs)], end='->')
        #     print()
        #     for j in range(S):
        #         vs = np.zeros(5)
        #         for a in range(5):
        #             h = j * 5 + a
        #             vs[a] = -v[h, i] / 0.1
        #         # print(np.exp(vs) / np.exp(vs).sum(), end='->')
        #         print(D_MAP[np.argmax(vs)], end='->')
        #     print()
        #
        # print(len(hash_set), found_cnt, len(trajectories))

        value_loss_epoch = 0
        action_loss_epoch = 0
        rnd_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0
        reward_prediction_loss_epoch = 0

        cnt = [0, 0]

        # the_generator = rollouts.feed_forward_generator(advantages, mini_batch_size=1)
        # the_sample = next(the_generator)
        # the_obs = the_sample[0]
        obss = []
        directions = []
        # for i in range(4):
        #     for j in range(4):
        #         if i != j:
        #             obs = [0., 0., 0., DX[i], DY[i], DX[j], DY[j]]
        #             obss.append(obs)
        #             directions.append((i + 1, j + 1))
        # for i in range(20):
        #     x1 = np.random.randn()
        #     x2 = np.random.randn()
        #     if np.sign(x1) == np.sign(x2):
        #         x2 = -x2
        #     directions.append((1 if x1 < 0. else 2, 1 if x2 < 0. else 2))
        #     obss.append([0., 0., 0., x1, 0., x2, 0.])

        # print(directions)

        # the_obs = torch.tensor(obss)
        # the_strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
        # the_fingerprint = the_strategy.argmax(-1)
        # print(the_fingerprint)
        # print(directions)
        # print(the_obs.size())
        # print(the_obs)
        # print(the_strategy)
        fgs = []
        advs = []
        nears = []
        fgmax = []
        grads = []

        episode_steps = 1

        last_centers = None

        ppo_epoch = self.ppo_epoch
        num_mini_batch = self.num_mini_batch
        num_mini_batch_needed = num_mini_batch

        dipg = self.args.dipg and self.args.use_reference
        # print(dipg)

        # if self.cnt <= 1:
        #     ppo_epoch = 50
        #     num_mini_batch = 32
        #     num_mini_batch_needed = 1
        #     print(11)

        interpolate_time = 0.

        for e in range(ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, num_mini_batch, episode_steps=episode_steps)

            mini_batch_cnt = 0
            for sample in data_generator:
                # the_strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()

                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, interpolate_masks_batch, rewards_batch, mmds_batch = sample

                batch_size = masks_batch.view(-1).size()[0]
                half_batch_size = batch_size // 2

                # for z in range(batch_size // 32):
                #     near = 0
                #     for i in range(32):
                #         if obs_batch[z][i][3:5].norm(2) < 0.11:
                #             cnt[0] += 1
                #             near = 1
                #             break
                #             # print(0, obs_batch[0], D_MAP[actions_batch.item()], adv_targ.item())
                #         if obs_batch[z][i][5:7].norm(2) < 0.11:
                #             cnt[1] += 1
                #             near = 2
                #             break
                #     nears.append(near)
                #     z_mask.append(near != 2)

                obs_batch = obs_batch.view(batch_size, -1)
                recurrent_hidden_states_batch = recurrent_hidden_states_batch.view(batch_size, -1)
                actions_batch = actions_batch.view(batch_size, -1)
                value_preds_batch = value_preds_batch.view(batch_size, -1)
                return_batch = return_batch.view(batch_size, -1)
                masks_batch = masks_batch.view(batch_size, -1)
                old_action_log_probs_batch = old_action_log_probs_batch.view(batch_size, -1)
                interpolate_masks_batch = interpolate_masks_batch.view(batch_size, -1)
                rewards_batch = rewards_batch.view(batch_size, -1)
                adv_targ = adv_targ.view(batch_size, -1)[:, :1 + num_refs * 2]
                mmds_batch = mmds_batch.view(batch_size, -1)
                # print(adv_targ)

                # print(return_batch.size(), value_preds_batch.size(), adv_targ.size())

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy_all, _, dists = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                if self.args.use_rnd:
                    _, rnd_values, rnd_predictions = self.actor_critic.get_reward_prediction(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                    )
                    rnd_loss = ((rnd_values - rnd_predictions).pow(2).sum(-1) / 2).mean()
                else:
                    rnd_loss = torch.tensor(0.0)

                # probs = self.actor_critic.get_strategy(obs_batch, recurrent_hidden_states_batch, masks_batch)
                #
                # if num_refs > 0:
                #     cross_entropy_all = -(torch.exp(action_log_probs) * ref_rewards).sum()
                # else:
                #     cross_entropy_all =

                # test_obs = torch.tensor(obss, requires_grad=True, dtype=torch.float)
                # test_obs = torch.tensor([[0., 0., 0., 0., 0., 0., 0.]], requires_grad=True, dtype=torch.float)
                # test_strategy = self.actor_critic.get_strategy(test_obs, None, None)
                # for z in range(test_obs.size()[0]):
                #     print("\nobs {}, 0direction:{}, 1direction:{}\n------------------".format(z, D_MAP[directions[z][0]], D_MAP[directions[z][1]]))
                #     print(test_obs[z].detach())
                #     for d in range(5):
                #         print("  direction-{}:".format(D_MAP[d]))
                #         obs_grad = torch.autograd.grad(test_strategy[z][d], test_obs, create_graph=True)[0][z]
                #         print(obs_grad.detach())
                # print(test_strategy)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss_all = -torch.min(surr1, surr2)

                if dipg:
                    dipg_loss_all = action_log_probs * mmds_batch
                else:
                    dipg_loss_all = torch.zeros_like(action_loss_all)

                on_d = []
                fgs = []
                acs = []

                if self.args.guided_updates is None or self.cnt <= self.args.guided_updates:
                    z_mask = []
                    MICRO_BATCH_SIZE = 1

                    tg = flat_view(ggrad(self.actor_critic, action_loss_all.mean()), self.actor_critic).detach()
                    net_add(self.actor_critic, -tg)
                    strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
                    tfg = strategy - the_strategy

                    net_add(self.actor_critic, tg)

                    gs = []

                    for z in range(batch_size // MICRO_BATCH_SIZE):
                        partial_action_loss = action_loss_all[z * MICRO_BATCH_SIZE: (z + 1) * MICRO_BATCH_SIZE].mean()
                        g = flat_view(ggrad(self.actor_critic, partial_action_loss), self.actor_critic).detach()
                        gs.append(g.numpy())
                        take = True
                        if MICRO_BATCH_SIZE == 1:
                            # if return_batch[z] > 0.:
                            #     print(obs_batch[z], actions_batch[z].item(), return_batch[z].item())
                            acs.append("action-{}".format(actions_batch[z].item()))
                            take = return_batch[z].item() > 0. and actions_batch[z].item() == 1
                        # gs.append((g / g.norm(2)).numpy())
                        # print(g.norm(2))
                        # print(g)
                        # for p in self.actor_critic.parameters():
                        #     print(p.size())
                        # print(g[-35:])
                        # print(self.actor_critic.dist.linear.weight)
                        net_add(self.actor_critic, -g)
                        # print(self.actor_critic.dist.linear.weight)
                        strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
                        # print(strategy)
                        # print(the_strategy)
                        fg = strategy - the_strategy
                        # gs.append(fg.view(-1).numpy())
                        # print(fg.argmax(-1))
                        if take:
                            fgs.append(fg.view(-1).numpy())
                        # print(fg)
                        # print(self.direction)
                        # print(fg.argmax(-1))
                        z_cnt = [0, 0]
                        t_cnt = 0
                        barred_fg = barred_argmax(fg)
                        for i in range(fg.size()[0]):
                            if barred_fg[i] == directions[i][0]:
                                z_cnt[0] += 1
                            elif barred_fg[i] == directions[i][1]:
                                z_cnt[1] += 1
                            if fg[i].argmax() == tfg[i].argmax():
                                t_cnt += 1

                        # print(z_cnt, fg.size()[0], t_cnt)
                        if take:
                            on_d.append(z_cnt[self.direction] > z_cnt[1 - self.direction])
                            # if actions_batch[z].item() == 1:
                            #     print(barred_argmax(fg), z_cnt)
                        if self.direction is not None:
                            agree = z_cnt[self.direction] > z_cnt[1 - self.direction]
                            # print(len(directions))
                            # agree = z_cnt[self.direction] < len(directions) // 4
                            # agree = True
                            z_mask.extend([agree] * MICRO_BATCH_SIZE)
                        else:
                            z_mask.extend([1.] * MICRO_BATCH_SIZE)
                        net_add(self.actor_critic, g)

                        grads.append(g)

                    def dis(a, b):
                        return np.square(a - b).sum()

                    # clusters = cluster(fgs, 2)
                    # c1 = 0
                    # c2 = 0
                    # for i in range(len(fgs)):
                    #     b = clusters.labels_[i] == 0
                    #     if b == on_d[i]:
                    #         c1 += 1
                    #     else:
                    #         c2 += 1
                    # print(c1, c2)
                        # print(clusters.labels_[i], on_d[i])
                    # print(clusters.labels_)
                    # centers = clusters.cluster_centers_
                    # if last_centers is not None:
                    #     for i1 in range(2):
                    #         for i2 in range(2):
                    #             print(dis(centers[i1], last_centers[i2]), end=' ')
                    #         print()
                    # print()
                    # last_centers = centers
                    tsne(fgs, on_d)
                else:
                    z_mask = 1.

                action_loss_all = torch.mul(action_loss_all, torch.tensor(z_mask, dtype=torch.float))
                interpolate_mask = torch.zeros((batch_size, 1 + num_refs))
                interpolate_mask[:, 0] = 1.

                selected_indices = None
                if self.args.use_reference:
                    choose_size = int(batch_size * 0.5)
                    criterion = return_batch[:, 1 + num_refs:]
                    # si = list(filter(lambda x: return_batch[x, 0] > 0., range(batch_size)))
                    si = list(range(batch_size))
                    cn = criterion.numpy()
                    # plot = self.args.plot_joint_plot
                    plot = False
                    if e == 0 and mini_batch_cnt == 0 and plot:
                        # si = list(range(batch_size))
                        # displot([cn[i] for i in filter(lambda x: return_batch[x, 0] > 0., range(batch_size))])
                        # displot(return_batch[:, 0])
                        # print(cn.shape[1])
                        for cni in range(num_refs):
                            if cni < 16:
                                continue
                            save_dir = mkdir2(self.save_dir, "ref-{}".format(cni))
                            save_path = os.path.join(save_dir, "{}.png".format(self.cnt))
                            save_path = None
                            # jointplot(cn[si][:, cni], return_batch[:, 0].numpy()[si],
                            #           save_path=save_path, title="agent-{} ref-{}".format(self.agent_name, cni))
                            jointplot(cn[si][:, cni], return_batch[:, 0].numpy()[si],
                                      save_path=save_path, title="agent-{} ref-{}".format(self.agent_name, cni))
                    # else:
                    #     print("yes")
                    pool = cn[si]
                    # from sklearn.mixture import GaussianMixture
                    # mixture = GaussianMixture(n_components=2).fit(pool.reshape(-1, 1))
                    # print(mixture.means_.flatten())
                    # print(mixture.weights_.flatten())
                    # if mixture.means_.max() < 150.:
                    #     lim = mixture.means_.mean() + 60.
                    # else:
                    #     lim = min(mixture.means_.mean(), 150.)
                    # print("lim:", lim)
                    lim = self.args.likelihood_threshold
                    if type(lim) == list:
                        lim = np.array(lim)
                    # criterion = torch.square(return_batch[:, 1:] - return_batch[:, 0])

                    # indices = [torch.argsort(criterion[:, i]) for i in range(return_batch.size()[1] - 1)]
                    # # print(criterion[indices[0][batch_size - choose_size]][0])
                    # ranks = [torch.argsort(ind) for ind in indices]
                    # s = torch.stack(ranks, dim=1).min(dim=1).values

                    # print(return_batch[:5, 1], s[:5])

                    # topk = torch.topk(s, choose_size)
                    # indices = topk.indices

                    # indices = list(filter(lambda x: ranks[0][x] >= choose_size and
                    #                                 ranks[1][x] >= choose_size, range(batch_size)))
                    # print(len(indices))

                    # indices = list(filter(lambda x: all(cn[x] > lim) or return_batch[x, 0] < 0., range(batch_size)))
                    # print(advantages.size())
                    # print(1)
                    if self.args.interpolate_rewards:
                        indices = list(range(batch_size))
                        interpolate_mask = interpolate_masks_batch
                        # st = time.time()
                        # interpolate_mask = []
                        # for i in range(batch_size):
                        #     if all(cn[i] > lim):
                        #         interpolate_mask.append([1.] + [0.] * num_refs)
                        #     else:
                        #         interpolate_mask.append([0.] + (cn[i] <= lim).astype(np.float).tolist())
                        # interpolate_mask = torch.Tensor(interpolate_mask)
                        # interpolate_time += time.time() - st
                        # print(time.time() - st)
                    elif not self.args.use_likelihood_reward_cap:
                        indices = list(filter(lambda x: all(cn[x] > lim), range(batch_size)))
                        # print(len(indices), batch_size)
                    else:
                        indices = list(range(batch_size))
                    # print(2)
                    # indices = list(filter(lambda x: all(cn[x] > lim) or return_batch[x, 0].item() < 0., range(batch_size)))

                    # indices = list(range(batch_size))

                    selected_indices = indices

                    # action_loss_all = action_loss_all[indices]
                    # print(action_loss_all.size())

                # adv_targ = torch.mul(adv_targ, torch.tensor(z_mask, dtype=torch.float).unsqueeze(-1))
                # print(obs_batch[0], dists.probs[0])

                # if adv_targ.item() > 0. and int(obs_batch[0][0] + 0.5) == 31:

                # if near != 1:
                #     mini_batch_cnt += 1
                #     # print(mini_batch_cnt)
                #     if mini_batch_cnt >= self.num_mini_batch:
                #         break
                #     continue
                    # print(1, obs_batch[0], D_MAP[actions_batch.item()], adv_targ.item())

                tmp_num_refs = num_refs * 2 if num_refs == num_value_refs else 0
                if self.use_clipped_value_loss:
                    # print("CLIP")
                    # value_preds_batch = value_preds_batch[:, :1]
                    # values = values[:, :1]
                    # return_batch = return_batch[:, :1]
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    # print(value_pred_clipped.size())
                    value_losses = (values - return_batch[:, :1 + tmp_num_refs]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch[:, :1 + tmp_num_refs]).pow(2)
                    value_loss_all = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    # print(1)
                    value_loss_all = 0.5 * (return_batch[:, :1 + tmp_num_refs] - values).pow(2)

                if self.args.use_reward_predictor:
                    # num_actions = self.actor_critic.base.num_actions
                    # one_hot_actions_batch = torch.nn.functional.one_hot(actions_batch, num_classes=num_actions).squeeze(1)
                    prediction = self.actor_critic.get_reward_prediction(obs_batch, recurrent_hidden_states_batch,
                                                                         masks_batch, actions_batch)
                    reward_prediction, random_net_value, random_net_prediction = prediction
                    reward_prediction_loss_all = 0.5 * (rewards_batch * self.args.reward_prediction_multiplier - reward_prediction).pow(2)
                    # reward_prediction_loss_all = 0.5 * (return_batch[:, 0] * 5. - reward_prediction).pow(2)
                    # reward_prediction_loss_all += 0.5 * (random_net_prediction - random_net_value).pow(2)
                else:
                    reward_prediction_loss_all = None

                if not self.args.interpolate_rewards:
                    value_loss_all = value_loss_all[:, :1]
                # print(value_loss_all[:5])
                # print(obs_batch[0])
                # print(value_loss_all.size())

                # with torch.no_grad():
                #     fg = torch.autograd.grad(action_loss, self.actor_critic.parameters(), create_graph=True, allow_unused=True)
                #     fgs.append(fg)

                if selected_indices is not None:
                    # print(self.agent_name, len(selected_indices), batch_size)
                    value_loss_all = value_loss_all[selected_indices]
                    action_loss_all = action_loss_all[selected_indices]
                    # action_loss_all = -action_loss_all
                    # for i in selected_indices:
                    #     action_loss_all[i] = -0.1 * action_loss_all[i]
                    dist_entropy_all = dist_entropy_all[selected_indices]

                    if self.args.use_reward_predictor:
                        reward_prediction_loss_all = reward_prediction_loss_all[selected_indices]

                # print(action_loss_all.size(), interpolate_mask.size())
                if self.args.interpolate_rewards:
                    _action_loss_all = torch.mul(action_loss_all, interpolate_mask).sum(dim=-1)
                    # print(interpolate_mask[0])
                    # print(interpolate_mask[:, 0].sum())
                    # print((action_loss_all[:, 0].detach() - _action_loss_all.detach()).square().sum())
                    weight = max(interpolate_mask.sum().item(), 1)
                    # print(action_loss_all.size()[0], weight)
                    action_loss_all = _action_loss_all * (action_loss_all.size()[0] / weight)

                reward_prediction_loss = torch.tensor(0.)

                value_loss = value_loss_all.mean()
                action_loss = action_loss_all.mean()
                dist_entropy = dist_entropy_all.mean()

                if self.args.use_reward_predictor:
                    reward_prediction_loss = reward_prediction_loss_all.mean()
                    # print("!!!")]

                dipg_loss = dipg_loss_all.mean()

                if self.task is None:
                    action_loss_mask = 1.

                    if self.args.use_reward_predictor:
                        self.reward_optimizer.zero_grad()
                        (reward_prediction_loss * self.reward_prediction_loss_coef).backward()
                    self.optimizer.zero_grad()
                    # self.value_loss_coef = 0.
                    loss = value_loss * self.value_loss_coef + rollouts.action_loss_coef * action_loss_mask * \
                           action_loss - dist_entropy * self.entropy_coef
                    if dipg:
                        loss += dipg_loss * self.args.dipg_alpha
                    if self.args.use_rnd:
                        loss += rnd_loss
                    loss.backward()

                    total_norm = 0.
                    for name, p in self.actor_critic.named_parameters():
                        if name.startswith("base.predictor") or name.startswith("base.random_net") or name.startswith("base.random_net_predictor"):
                            continue
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    grad_norm_epoch += total_norm

                    if self.clip_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                 self.max_grad_norm)
                    self.optimizer.step()

                    if self.args.use_reward_predictor:
                        self.reward_optimizer.step()
                elif self.task[:4] == "grad":
                    if return_batch[0].item() > 0.5:
                        g = flat_view(ggrad(self.actor_critic, action_loss)).detach()
                        # print(g.norm(2))
                        net_add(self.actor_critic, -g)
                        strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
                        # print(the_strategy.size())
                        fg = strategy - the_strategy
                        fgmax.append(fg.argmax())
                        fgs.append(fg.reshape(-1).numpy())
                        advs.append(adv_targ.mean().item())
                        nears.append(["no", "near-1", "near-2"][near])
                        net_add(self.actor_critic, g)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                rnd_loss_epoch += rnd_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                reward_prediction_loss_epoch += reward_prediction_loss.item()

                # strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
                # fg = []
                # for i in range(strategy.size()[0]):
                #     top2 = strategy[i].topk(2)
                #     if top2.values[0] - top2.values[1] < 0.1:
                #         fg.append(-1)
                #     else:
                #         fg.append(top2.indices[0].item())
                #
                # # print(fg)
                # fgs.append((fg, e, mini_batch_cnt))

                mini_batch_cnt += 1
                # print(mini_batch_cnt)
                if mini_batch_cnt >= num_mini_batch_needed:
                    break

        # import joblib
        # joblib.dump(fgs, "data/policy-fingerprints/fixed-init.run-{}.update-{}.data".format(self.args.reseed_z, self.cnt))

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # import pandas as pd
        #
        # df = pd.DataFrame(dict(x=fgmax, y=nears))
        # sns.displot(data=df, x="x", y="y")
        # plt.show()
        #
        # print(cnt)

        # mean_g = torch.stack(grads).mean(0)
        # for i, g in enumerate(grads):
        #     if nears[i] != 0:
        #         print(g.norm(2), g @ mean_g / g.norm(2) / mean_g.norm(2), nears[i])
        # print("interpolate_time:", interpolate_time)

        if type(self.task) == str:
            if self.task[:3] == "cnt":
                import joblib
                joblib.dump(cnt, "{}.obj".format(self.task))
            elif self.task[:4] == "grad":
                pass
                # n = len(fgs)
                # dim = fgs[0].shape[0]
                # c = 2
                # ids = [[] for _ in range(c)]
                # xs = [np.zeros(dim) for _ in range(c)]
                #
                # def norm(vec):
                #     return np.linalg.norm(vec)
                #
                # for i in range(n):
                #     j = np.random.choice(c)
                #     ids[j].append(i)
                #     xs[j] += fgs[i] / norm(fgs[i])
                #
                # for j in range(c):
                #     xs[j] /= len(ids[j])
                #
                # for n_iter in range(20):
                #     ids = [[] for _ in range(c)]
                #     new_xs = [np.zeros(dim) for _ in range(c)]
                #     # print(xs)
                #     for i in range(n):
                #         nearest_dis = -5.
                #         nearest_j = 0
                #         for j in range(c):
                #             dis = fgs[i] @ xs[j] / (norm(fgs[i]) * norm(xs[j]))
                #             # print(dis, norm(fgs[i]), norm(xs[j]))
                #             if dis > nearest_dis:
                #                 nearest_dis = dis
                #                 nearest_j = j
                #         ids[nearest_j].append(i)
                #         new_xs[nearest_j] += fgs[i] / norm(fgs[i])
                #     for j in range(c):
                #         xs[j] = new_xs[j] / len(ids[j])
                #
                #     avg = [0. for _ in range(c)]
                #     for j in range(c):
                #         print(len(ids[j]))
                #         for i in ids[j]:
                #             avg[j] += int(nears[i] == "near-2") / len(ids[j])
                #     print(avg)
                # # print(len(fgs), fgs[0])
                # tsne(fgs, nears)
                # import joblib
                # joblib.dump((fgs, advs), "{}.obj".format(self.task))

        # mean_fg = []
        # for i, g in enumerate(fgs[0]):
        #     if g is None:
        #         mean_fg.append(None)
        #     else:
        #         mean_g = torch.zeros_like(g)
        #         for fg in fgs:
        #             mean_g += fg[i]
        #         mean_fg.append(mean_g / len(fgs))
        # fgs.append(tuple(mean_fg))
        #
        # print("generating fingerprints")
        # data_generator = rollouts.feed_forward_generator(advantages, 1)
        # fingerprints = []
        # for i, fg in enumerate(fgs):
        #     for p, g in zip(self.actor_critic.parameters(), fg):
        #         if g is not None:
        #             p.data += 1e-4 * g
        #
        #     fingerprint = []
        #
        #     for sample in data_generator:
        #         obs_batch, recurrent_hidden_states_batch, actions_batch, \
        #         value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
        #         adv_targ = sample
        #
        #         if i == 0:
        #             for j, obs in enumerate(list(obs_batch)):
        #                 if obs[0] < 0.01 and (obs[3] * obs[5] < 0 or obs[4] * obs[6] < 0):
        #                     print(j, obs)
        #
        #         # Reshape to do in a single forward pass for all steps
        #         values, action_log_probs, dist_entropy, _, dist = self.actor_critic.evaluate_actions(
        #             obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
        #
        #         fingerprint.append(dist.probs)
        #         break
        #
        #     # print(len(fingerprint), fingerprint[0].size())
        #     fingerprint = torch.cat(fingerprint, dim=0).view(-1)
        #     fingerprints.append(fingerprint.detach().numpy())
        #
        #     for p, g in zip(self.actor_critic.parameters(), fg):
        #         if g is not None:
        #             p.data -= 1e-4 * g
        #
        # np.array(fingerprints).dump("grads.15.fingerprint.data")

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        rnd_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates
        reward_prediction_loss_epoch /= num_updates

        # print(fgs[-1])

        # print(self.actor_critic.dist.linear.weight, self.actor_critic.dist.linear.bias)
        # print(self.actor_critic.dist.linear.weight)

        # print(time.time() - st)

        return value_loss_epoch, action_loss_epoch, rnd_loss_epoch, dist_entropy_epoch, grad_norm_epoch, reward_prediction_loss_epoch
