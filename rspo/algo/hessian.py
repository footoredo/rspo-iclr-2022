import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from rspo.multi_agent.utils import flat_view, net_add, ggrad, get_hessian, extract_trajectory, calc_sim


D_MAP = "SLRDU"
DX = [-1., 1., 0., 0.]
DY = [0., 0., -1., 1.]


class Hessian():
    def __init__(self,
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
                 args=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.clip_grad_norm = clip_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.task = task
        self.direction = direction

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, betas=(0., 0.999))

        self.cnt = 0
        self.args = args
        self.last_direc = None

    def update(self, rollouts):
        self.cnt += 1
        # advantages = rollouts.returns[:-1]
        advantages = rollouts.returns[:-1] # - rollouts.value_preds[:-1]
        # advantages = (advantages - advantages.mean()) / (
        #     advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0

        fgs = []
        advs = []
        nears = []
        fgmax = []
        grads = []

        episode_steps = 1

        ppo_epoch = self.ppo_epoch
        num_mini_batch = self.num_mini_batch
        num_mini_batch_needed = num_mini_batch

        ppo = False

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
                        adv_targ = sample

                batch_size = value_preds_batch.view(-1).size()[0]

                if not ppo:
                    _, action_log_probs, _, _, _ = self.actor_critic.evaluate_actions(
                        obs_batch.view(batch_size, -1), None, None, actions_batch.view(batch_size, -1))
                    action_log_probs = action_log_probs.view(batch_size // episode_steps, episode_steps, -1)

                    hessians = []
                    net = self.actor_critic

                    def get_obj(_i, _r):
                        return torch.div(torch.mul(pi[:, _i], _r), pi[:, _i].detach())

                    log_prob = torch.zeros_like(action_log_probs[:, 0])
                    for t in range(episode_steps):
                        step_rewards = return_batch[:, t] - return_batch[:, t + 1] if t < episode_steps - 1 else \
                            return_batch[:, t]
                        log_prob += action_log_probs[:, t]
                        prob = torch.exp(log_prob)
                        hessians.append(get_hessian(net, torch.div(torch.mul(prob, step_rewards), prob.detach()).sum()).detach())

                    hessian = torch.stack(hessians, dim=0).sum(dim=0)
                    w, v = np.linalg.eigh(hessian)
                    # print(w)

                    if self.last_direc is not None:
                        sims = []
                        for i in range(5):
                            sims.append(calc_sim(self.last_direc, v[:, -i - 1]))
                        print(sims)
                        sims = np.array(sims)
                        ci = np.abs(sims).argmax()
                        s = np.sign(sims[ci])
                        sim = sims[ci] * s
                        direc = v[:, -ci - 1] * s
                    else:
                        direc = -v[:, -1]
                        sim = 0.
                    if self.last_direc is None:
                        self.last_direc = direc
                    else:
                        self.last_direc = self.last_direc * 0.9 + direc * 0.1
                    print("sim:", sim)
                    net_add(net, 1e-1 * torch.tensor(self.last_direc))

                if ppo:
                    obs_batch = obs_batch.view(batch_size, -1)
                    recurrent_hidden_states_batch = recurrent_hidden_states_batch.view(batch_size, -1)
                    actions_batch = actions_batch.view(batch_size, -1)
                    value_preds_batch = value_preds_batch.view(batch_size, -1)
                    return_batch = return_batch.view(batch_size, -1)
                    masks_batch = masks_batch.view(batch_size, -1)
                    old_action_log_probs_batch = old_action_log_probs_batch.view(batch_size, -1)
                    adv_targ = adv_targ.view(batch_size, -1)

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, _, dists = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)

                    ratio = torch.exp(action_log_probs -
                                      old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss_all = -torch.min(surr1, surr2)

                    action_loss = action_loss_all.mean()

                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                     value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    # with torch.no_grad():
                    #     fg = torch.autograd.grad(action_loss, self.actor_critic.parameters(), create_graph=True, allow_unused=True)
                    #     fgs.append(fg)

                    if self.task is None:
                        action_loss_mask = 1.

                        self.optimizer.zero_grad()
                        (value_loss * self.value_loss_coef + action_loss_mask * action_loss -
                         dist_entropy * self.entropy_coef).backward()

                        total_norm = 0.
                        for p in self.actor_critic.parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        grad_norm_epoch += total_norm

                        if self.clip_grad_norm:
                            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                     self.max_grad_norm)
                        self.optimizer.step()


                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item()

                mini_batch_cnt += 1
                # print(mini_batch_cnt)
                if mini_batch_cnt >= num_mini_batch_needed:
                    break

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch
