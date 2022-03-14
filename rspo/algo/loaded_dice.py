import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from rspo.utils import magic_box
from rspo.multi_agent.utils import mkdir2


class LoadedDiCE():
    def __init__(self,
                 actor_critic,
                 dice_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 dice_lambda,
                 episode_steps,
                 task,  # "MIS", "eigen", None
                 clip_param=0.2,
                 max_grad_norm=0.5,
                 clip_grad_norm=True,
                 lr=None,
                 eps=None,
                 use_clipped_value_loss=True,
                 save_dir="/tmp/loaded_dice"):

        self.actor_critic = actor_critic

        self.epoch = dice_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.dice_lambda = dice_lambda
        self.episode_steps = episode_steps
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.clip_grad_norm = clip_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.task = task
        self.save_dir = save_dir
        self.lr = lr
        self.eps = eps

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

        self.evecs = None
        self.evals = None
        self.evec = None
        self.eval = None
        self.last_gradient = None
        self.update_counter = 0

    def update(self, rollouts):
        self.update_counter += 1
        advantages = rollouts.returns[:-1] #- rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0

        def flat_view(data):
            return data.view(-1, *data.size()[2:])

        def calc_grad(f, create_graph=False):
            return torch.autograd.grad(f, self.actor_critic.parameters(), allow_unused=True, create_graph=create_graph)

        def flat_grad(grad):
            grads = []
            for gg in grad:
                if gg is not None:
                    grads.append(gg.reshape(-1))
            return torch.cat(grads)

        def hv(fg, vec):
            return flat_grad(calc_grad(fg @ vec, True)).detach()

        def power_method(gradients):
            fg = flat_grad(gradients)
            delta_matrix = torch.zeros((fg.size()[0], fg.size()[0]))

            def _hv(vec, use_dm=True):
                v1 = hv(fg, vec)
                if use_dm:
                    v1 += delta_matrix @ vec
                return v1

            evals = []
            evecs = []

            while True:
                v = torch.rand(fg.size())
                v = v / v.norm(2)
                for i in range(200):
                    v = _hv(v)
                    v = v / v.norm(2)
                nv = (_hv(v, False) / v).detach().numpy()
                print(nv)
                print(nv.std())
                if nv.std() < 0.1:
                    eval = nv.mean()
                    evec = v.detach().numpy().reshape(-1, 1)
                    evals.append(eval)
                    evecs.append(evec)
                    # if len(evecs) > 1:
                    #     for i in range(len(evecs) - 1):
                    #         print(evecs[i].T @ evec)
                    print("lambda_{}: {}".format(len(evals), eval))
                    delta_matrix -= eval * (evec @ evec.T)
                else:
                    break

            # print(evals)
            # print(evecs)

        def lanczos_method(gradients):
            fg = flat_grad(gradients)
            n = fg.size()[0]
            fg = fg.view(1, n)
            # print(fg)

            v = torch.randn((n, 1))
            v /= v.norm(2)
            vs = [v]
            wp = hv(fg, v).view((n, 1))
            a = wp.transpose(0, 1) @ v
            w = wp - a * v

            eps = 1e-3
            m = 200
            T = np.zeros((m, m))
            T[0, 0] = a

            for j in range(1, m):
                b = w.norm(2)
                T[j - 1, j] = b
                T[j, j - 1] = b
                lv = v
                if b > eps or b < -eps:
                    v = w / b
                else:
                    print("resample!")
                    v = torch.randn((n, 1))
                    for vv in vs:
                        v -= (v.transpose(0, 1) @ vv) * vv
                    v /= v.norm(2)
                wp = hv(fg, v).view((n, 1))
                a = wp.transpose(0, 1) @ v
                w = wp - a * v - b * lv

                # dv = torch.zeros_like(v)
                # for q in vs:
                #     # print(v.T @ q)
                #     dv += (v.T @ q) * q
                # v -= dv

                vs.append(v)
                # print(a, b)

                T[j, j] = a

            evals, evecs = np.linalg.eigh(T)

            V = torch.cat(vs, 1).detach().numpy()
            # print(V.shape)
            qs = []
            new_evals = []
            new_evecs = []
            for i in range(m):
                z = V @ evecs[:, i]
                z /= np.linalg.norm(z)

                test = hv(fg, torch.tensor(z).float()).numpy() / z / evals[i] - 1
                if np.linalg.norm(test) < 0.1:
                    # return evals[i], z
                    orth = 0.
                    for q in qs:
                        orth = max(orth, np.abs(z.T @ q))

                    if orth < 1:
                        qs.append(z)
                        new_evals.append(evals[i])
                        new_evecs.append(z)

                        # print(z.shape, z)
                        # print(evals[i], hv(fg, torch.tensor(z).float()).numpy() / z)
                    # else:
                    #     print(orth)
            return new_evals, new_evecs

        def add_grad(grad, direction):
            s = 0
            with torch.no_grad():
                for p, g in zip(self.actor_critic.parameters(), grad):
                    if g is not None:
                        length = g.view(-1).size()[0]
                        p.data += torch.tensor(direction[s:s + length].reshape(p.data.size())).float()
                        s += length

        def test_evec(evec, gradients):
            nevec = evec / evec.norm(2)
            test_vec = hv(flat_grad(gradients), nevec)
            p_eval = test_vec.norm(2)
            sign_i = torch.abs(nevec).argmax().item()
            sign = 1 if test_vec[sign_i] * nevec[sign_i] > 0 else -1
            test_vec /= test_vec.norm(2)
            random_vec = torch.randn(*self.evec.shape)
            random_vec /= random_vec.norm(2)
            test_vec2 = hv(flat_grad(gradients), random_vec)
            test_vec2 /= test_vec2.norm(2)
            return sign * p_eval.item(), (test_vec - sign * nevec).norm(2).item(), (test_vec2 - sign * random_vec).norm(2).item()

        if self.task == "eigen":
            epoch = 1
            num_mini_batch = 1
        else:
            epoch = self.epoch
            num_mini_batch = self.num_mini_batch
        for e in range(epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, num_mini_batch, None, self.episode_steps)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, num_mini_batch, None, self.episode_steps)

            reward = 0.
            cnt_sample = 0
            for sample in data_generator:
                cnt_sample += 1
                if cnt_sample > num_mini_batch:
                    break
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                # print(obs_batch[0], actions_batch[0], adv_targ[0])

                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(
                    flat_view(obs_batch), flat_view(recurrent_hidden_states_batch), flat_view(masks_batch),
                    flat_view(actions_batch))

                # print("obs:", obs_batch[0, 0], obs_batch[0, 1])
                # print(actions_batch.size())
                # print("act:", actions_batch[0, 0])
                # print(action_log_probs.size())
                action_log_probs = action_log_probs.reshape(old_action_log_probs_batch.size())

                empty_mask = (1 - masks_batch).type(torch.ByteTensor)
                # print(action_log_probs.size(), masks_batch.size())
                action_log_probs[empty_mask] = 1.0

                weighted_cumsum = torch.zeros_like(action_log_probs)
                weighted_cumsum[:, 0] = action_log_probs[:, 0]
                for t in range(1, action_log_probs.size()[1]):
                    weighted_cumsum[:, t] = self.dice_lambda * weighted_cumsum[:, t - 1] + action_log_probs[:, t]
                deps_exclusive = weighted_cumsum - action_log_probs
                full_deps = magic_box(weighted_cumsum) - magic_box(deps_exclusive)

                action_loss = -(adv_targ * full_deps).mean()
                # print(action_loss.size())

                if self.use_clipped_value_loss:
                    values = values.view(value_preds_batch.size())
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                # print(value_loss.size())

                gradients = calc_grad(action_loss, create_graph=True)
                reward += return_batch[:, 0].mean().item()
                # print("reward:", reward)

                total_norm = 0.
                for g in gradients:
                    if g is not None:
                        param_norm = g.norm(2)
                        total_norm += param_norm ** 2
                total_norm = total_norm ** (1. / 2)
                grad_norm_epoch += total_norm.item()

                if self.task == "MIS":
                    # print("!")
                    self.optimizer.zero_grad()
                    # (value_loss * self.value_loss_coef + total_norm -
                    #  dist_entropy * self.entropy_coef).backward()
                    (-dist_entropy).backward()

                    self.optimizer.step()
                elif self.task == "gradient":
                    gradient = flat_grad(gradients)
                    gradient /= gradient.norm(2)
                    if self.last_gradient is not None:
                        print("similarity:", self.last_gradient @ gradient)
                    self.last_gradient = gradient
                elif self.task == "eigen":
                    # lanczos_method(gradients)
                    if self.evec is None:
                        evals, evecs = lanczos_method(gradients)
                        evals = np.array(evals)
                        evecs = np.array(evecs)
                        # import joblib
                        # joblib.dump((evals, evecs), os.path.join(mkdir2(self.save_dir, "update-{}".format(self.update_counter)), "eigen.data"))
                        if self.evec is not None:
                            for evec in evecs:
                                dis = self.evec @ evec
                                print(dis)
                        # self.evecs = evecs
                        # self.evals = evals
                        # self.eval = evals[0]
                        # self.evec = evecs[0]
                        dists = []
                        m = 5
                        for i in range(m):
                            print(evals[i], evecs[i])

                            add_grad(gradients, evecs[i])
                            _, _, _, _, dist = self.actor_critic.evaluate_actions(
                                flat_view(obs_batch), flat_view(recurrent_hidden_states_batch), flat_view(masks_batch),
                                flat_view(actions_batch))
                            dists.append(dist.logits)
                            add_grad(gradients, -evecs[i])

                        np.set_printoptions(precision=4)
                        sim = np.zeros((m, m))
                        for i in range(m):
                            for j in range(i + 1):
                                dis = (dists[i] - dists[j]).norm(2)
                                sim[i, j] = sim[j, i] = dis
                        print(sim)

                        self.evec = evecs[0]
                        self.eval = evals[0]
                        self.evecs = evecs
                        self.evals = evals
                        # self.task = None

                        # print(eval, evec)
                        # self.evecs = evec
                    else:
                        fg = flat_grad(gradients)
                        te = torch.tensor(self.evec).float()
                        new_eval = (te @ hv(fg, te)).detach().item()

                        eval = torch.tensor(new_eval, requires_grad=True)
                        evec = torch.tensor(self.evec, requires_grad=True, dtype=torch.float)

                        # optimizer = optim.Adam((eval, evec), lr=self.lr, eps=self.eps)

                        lr = 1e-2

                        print(test_evec(evec, gradients))
                        p_evec = torch.tensor(self.evec, dtype=torch.float)
                        for _ in range(200):
                            nevec = evec / evec.norm(2)
                            h = hv(fg, nevec)
                            loss = (h / h.norm(2) + nevec).norm(2) - nevec @ p_evec  # lambda < 0

                            # optimizer.zero_grad()
                            # loss.backward()
                            # optimizer.step()
                            g = torch.autograd.grad(loss, evec, create_graph=True)[0]
                            # print(g.norm(2), g.max(), g.min())
                            evec.data -= lr * g

                            if (_ + 1) % 50 == 0:
                                print(_, test_evec(evec, gradients))

                        self.eval, _, _ = test_evec(evec, gradients)
                        self.evec = evec.detach().numpy()
                        for i in range(5):
                            print(self.evec @ self.evecs[i])
                    alpha = 1e-1
                    add_grad(gradients, -alpha * self.evec)

                    gradients = calc_grad(action_loss, create_graph=True)
                    total_norm = 0.
                    for g in gradients:
                        if g is not None:
                            param_norm = g.norm(2)
                            total_norm += param_norm ** 2
                    total_norm = total_norm ** (1. / 2)
                    if total_norm > 0.1:
                        print("Eigen stop. Start SGD.")
                        self.task = None

                    #
                    # test_vec = hv(flat_grad(gradients), torch.tensor(self.evec).float()).detach().numpy() / self.evec
                    # random_vec = np.random.randn(*self.evec.shape)
                    # random_vec /= np.linalg.norm(random_vec)
                    # test_vec2 = hv(flat_grad(gradients), torch.tensor(random_vec).float()).detach().numpy() / random_vec
                    # print(self.eval, np.mean(test_vec), np.std(test_vec), np.std(test_vec2))

                    # for i, evec in enumerate(self.evecs):
                    #     test_vec = hv(flat_grad(gradients), torch.tensor(evec).float()).detach().numpy() / evec
                    #     random_vec = np.random.randn(*evec.shape)
                    #     random_vec /= np.linalg.norm(random_vec)
                    #     test_vec2 = hv(flat_grad(gradients), torch.tensor(random_vec).float()).detach().numpy() / random_vec
                    #     print(i, self.evals[i], np.mean(test_vec), np.std(test_vec), np.std(test_vec2))

                    # s = 0
                    # with torch.no_grad():
                    #     for p, g in zip(self.actor_critic.parameters(), gradients):
                    #         if g is not None:
                    #             length = g.view(-1).size()[0]
                    #             p.data += torch.tensor(self.evec[s:s + length].reshape(p.data.size())).float()
                    #             s += length
                else:
                    self.optimizer.zero_grad()
                    (value_loss * self.value_loss_coef + action_loss -
                     dist_entropy * self.entropy_coef).backward()

                    if self.clip_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                 self.max_grad_norm)

                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            if e == 0:
                print("reward:", reward / num_mini_batch)

        num_updates = epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch
