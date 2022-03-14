import time
import json

import pandas as pd

import wandb

from rspo.arguments import get_args
from multiprocessing import shared_memory

from functools import partial

from rspo.multi_agent import Agent, RefAgent, Environment


from rspo.multi_agent.utils import *

import logging
import multiprocessing as mp

from copy import deepcopy

import math

# import mujoco_py


def train_in_turn(n_agents, i, n_iter):
    return n_iter % n_agents == i


def train_simultaneously(n_agents, i, n_iter):
    return True


def no_train(n_agents, i, n_iter):
    return False


def _run(args, logger):
    run_name = get_timestamp()

    if args.use_wandb:
        wandb_config = {
            "project": args.wandb_project,
            "entity": CONFIDENTIAL["wandb"]["username"],
            "group": args.wandb_group,
            "name": run_name,
            "config": args
        }

        wandb.init(**wandb_config)

    result = dict()

    num_agents = args.num_agents
    num_envs = args.num_processes

    # if args.use_reference:
    #     args.num_env_steps *= 2

    agents = []
    main_agent_conns = []
    envs = []
    main_env_conns = []

    _make_env = partial(make_env, args.env_name, args.episode_steps, args.env_config)
    env = _make_env()

    input_structures = env.input_structures
    if args.use_reference and args.ref_config is not None:
        ref_config = json.load(open(args.ref_config))
    else:
        ref_config = None

    ref_agents = []
    num_refs_all = []
    if args.use_reference:
        for i, agent in enumerate(env.agents):
            obs_space = env.observation_spaces[agent]
            act_space = env.action_spaces[agent]

            if ref_config is not None:
                ref_load_dirs = []
                ref_load_steps = []
                ref_num_refs = []
                ref_load_agents = []
                for ref in ref_config[str(env.agents[i])]:
                    ref_load_dirs.append(ref["load_dir"])
                    ref_load_steps.append(ref["load_step"])
                    ref_num_refs.append(ref["num_refs"])
                    ref_load_agents.append(ref["load_agent"])
            else:
                ref_load_dirs = args.ref_load_dir
                ref_load_steps = args.ref_load_step
                ref_num_refs = args.ref_num_ref
                if type(ref_load_dirs) != list:
                    ref_load_dirs = [ref_load_dirs]
                if type(ref_load_steps) != list:
                    ref_load_steps = [ref_load_steps] * len(ref_load_dirs)
                if type(ref_num_refs) != list:
                    ref_num_refs = [ref_num_refs] * len(ref_load_dirs)
                ref_load_agents = [env.agents[i]] * len(ref_load_dirs)
            ref_agents_i = []
            # print(ref_load_steps, args.ref_use_ref)
            for ld, ls, nr, la in zip(ref_load_dirs, ref_load_steps, ref_num_refs, ref_load_agents):
                if not args.no_load_refs:
                    ref_agent, _ = get_agent(la, args, obs_space, input_structures[agent], act_space, None, n_ref=nr, is_ref=True)
                    obs_rms, trajectories = load_actor_critic(ref_agent, ld, la if type(la) == str else env.agents[la],
                                                              ls, load_trajectories=args.dipg)
                    # print(ld, obs_rms)
                    ref_agent.obs_rms = obs_rms
                    ref_agent.trajectories = trajectories
                else:
                    ref_agent = None
                ref_agents_i.append(ref_agent)
            ref_agents.append(ref_agents_i)
            num_refs_all.append(len(ref_agents_i))
            # print(num_refs)

    save_dir = mkdir2(args.save_dir, run_name)
    json.dump(vars(args), open(os.path.join(save_dir, "config.json"), "w"), indent=2)

    obs_locks = []
    act_locks = []
    ref_locks = []

    for i in range(num_envs):
        _obs_locks = []
        _act_locks = []
        for j in range(num_agents):
            # __obs_locks = []
            # __act_locks = []
            # for k in range(1 + num_refs_all[j]):
            #     __obs_locks.append(mp.Event())
            #     # _obs_locks[-1].set()
            #     __act_locks.append(mp.Event())
            # # _act_locks[-1].set()
            # _obs_locks.append(__obs_locks)
            # _act_locks.append(__act_locks)
            _obs_locks.append(mp.Event())
            _act_locks.append(mp.Event())
        obs_locks.append(_obs_locks)
        act_locks.append(_act_locks)

    # for i in range(num_agents):
    #     _ref_locks = []
    #     for k in range(num_refs_all[i]):
    #         _ref_locks.append(mp.Event())
    #     ref_locks.append(_ref_locks)

    assert num_agents == len(env.agents)
    obs_buffer_size = 0
    item_size = np.zeros(1, dtype=np.float32).nbytes
    # print(item_size)
    obs_indices = []

    for i, agent in enumerate(env.agents):
        # print(env.observation_spaces[agent].shape)
        next_obs_buffer_size = obs_buffer_size + item_size * num_envs * (env.observation_spaces[agent].shape[0] + 4)
        # print(next_obs_buffer_size - obs_buffer_size)
        obs_indices.append((obs_buffer_size, next_obs_buffer_size))
        obs_buffer_size = next_obs_buffer_size

    act_sizes = []
    act_recover_fns = []
    for agent in env.agents:
        act_sizes.append(get_action_size(env.action_spaces[agent], in_buffer=True))
        act_recover_fns.append(get_action_recover_fn(env.action_spaces[agent]))

    sum_act_sizes = sum(act_sizes)

    obs_shm = shared_memory.SharedMemory(create=True, size=obs_buffer_size)
    act_shm = shared_memory.SharedMemory(create=True, size=num_envs * sum_act_sizes * item_size)
    ref_shms = []

    # print("input_structures:", input_structures)

    # print(len(env.agents))
    # print(env.agents)

    train_fn = no_train
    if args.train:
        if args.train_in_turn:
            train_fn = train_in_turn
        else:
            train_fn = train_simultaneously

    data_queue = mp.Queue() if args.use_wandb else None

    for i, agent in enumerate(env.agents):
        conn1, conn2 = mp.Pipe()
        main_agent_conns.append(conn1)

        obs_space = env.observation_spaces[agent]
        act_space = env.action_spaces[agent]

        ref_shm = None

        if args.use_reference:
            pass
            # num_actions = act_space.n

            # print(num_refs_all[i], args.num_processes, num_actions, item_size)
            # ref_shm = shared_memory.SharedMemory(create=True, size=num_refs_all[i] * args.num_processes * num_actions * item_size)
            # ref_processes_i = []
            # for j, ref_agent in enumerate(ref_agents[i]):
            #     ref_process = RefAgent(agent=ref_agent, agent_id=i, ref_id=j, num_refs=num_refs_all[i],
            #                            num_actions=num_actions, args=args, obs_shm=obs_shm,
            #                            buffer_start=obs_indices[i][0], buffer_end=obs_indices[i][1],
            #                            ref_shm=ref_shm, obs_locks=[locks[i][1 + j] for locks in obs_locks],
            #                            act_locks=[locks[i][1 + j] for locks in act_locks],
            #                            ref_locks=[locks[j] for locks in ref_locks])
            #     ref_processes_i.append(ref_process)
            #     # ref_process.start()
            #
            #     # while True:
            #     #     obs = list(map(float, input().split()))
            #     #     obs = torch.tensor(obs, dtype=torch.float)
            #     #     print(ref_agent.get_strategy(obs, None, None))
            # ref_shms.append(ref_shm)

        # print("123123132")
        num_refs = None if args.num_refs is None else args.num_refs[i]
        thread_limit = args.parallel_limit // num_agents
        # thread_limit = None
        ap = Agent(i, env.agents[i], thread_limit=thread_limit, logger=logger.info, args=args,
                   obs_space=obs_space,
                   input_structure=input_structures[agent],
                   act_space=act_space, act_sizes=act_sizes, main_conn=conn2,
                   obs_shm=obs_shm, buffer_start=obs_indices[i][0], buffer_end=obs_indices[i][1],
                   obs_locks=[locks[i] for locks in obs_locks], act_shm=act_shm, ref_shm=None,
                   act_locks=[locks[i] for locks in act_locks], ref_locks=None,
                   use_attention=args.use_attention,
                   save_dir=save_dir,
                   train=partial(train_fn, args.num_agents, i),
                   num_refs=num_refs,
                   reference_agent=ref_agents[i] if args.use_reference else None,
                   data_queue=data_queue,
                   norm_obs=args.obs_normalization,
                   norm_reward=args.reward_normalization)
        # print("123123123", i)
        agents.append(ap)
        ap.start()

    for i in range(num_envs):
        conn1, conn2 = mp.Pipe()
        main_env_conns.append(conn1)
        ev = Environment(i, logger.info, args, env=_make_env(), agents=env.agents, main_conn=conn2,
                         act_sizes=act_sizes,
                         act_recover_fns=act_recover_fns,
                         obs_shm=obs_shm,
                         obs_locks=obs_locks[i],
                         act_shm=act_shm,
                         act_locks=act_locks[i])
        envs.append(ev)
        ev.start()

    for conn in main_agent_conns:
        conn.send(None)

    for conn in main_env_conns:
        conn.send(None)

    num_updates = args.num_env_steps // args.num_steps // args.num_processes

    if args.use_wandb:
        agent_finish_count = 0
        while agent_finish_count < num_agents:
            data = data_queue.get()
            # print(data)
            if data is None:
                agent_finish_count += 1
            else:
                agent_name = "agent-{}".format(data["agent"])
                iteration = data["iteration"]
                log_data = dict(iteration=iteration)
                for key, value in data.items():
                    if key != "agent" and key != "iteration":
                        log_data[agent_name + "/" + key] = value
                wandb.log(log_data, step=iteration)

    if args.reject_sampling:
        for i in range(num_updates):
            while True:
                # This happens every batch (times num_envs)
                finish = True
                for j in range(num_agents):
                    cf = main_agent_conns[j].recv()
                    # print(j, cf)
                    finish = finish and cf
                    # print(j, finish)
                for j in range(num_agents):
                    main_agent_conns[j].send(finish)
                for j in range(num_envs):
                    main_env_conns[j].send(finish and i == num_updates - 1)
                if finish:
                    break

    for ev in envs:
        ev.join()

    obs_shm.close()
    obs_shm.unlink()

    statistics = dict()
    for i, agent in enumerate(env.agents):
        statistics[agent] = main_agent_conns[i].recv()

    result["statistics"] = statistics

    import joblib
    joblib.dump(statistics, os.path.join(save_dir, "statistics.obj"))

    if args.plot:
        plot_statistics(statistics, "reward")
        if args.use_reference:
            plot_statistics(statistics, "dist_penalty")
        # plot_statistics(statistics, "grad_norm")
        # plot_statistics(statistics, "value_loss")
        # plot_statistics(statistics, "action_loss")
        # plot_statistics(statistics, "dist_entropy")

    is_simple = args.env_name[:6] == "simple"
    is_gw = args.env_name[-2:] == "gw"
    is_mujoco = args.env_name == "half-cheetah" or args.env_name == "hopper" or args.env_name == "humanoid" or \
        args.env_name == "ant" or args.env_name == "walker2d" or args.env_name == "swimmer"

    close_to = [0, 0, 0, 0, 0]
    gather_count = np.zeros((5, 5), dtype=int)
    eat_fruit_cnt = [0, 0]
    hurt_cnt = [0, 0]
    attack_cnt = 0
    first_gather_count = np.zeros((5, 5), dtype=int)
    gathered = False
    sum_reward = np.zeros(num_agents)
    max_meet = args.episode_steps
    actions_after_meeting = np.zeros((max_meet, 2, 5), dtype=float)
    cnt_after_meeting = np.zeros(max_meet, dtype=int)
    cnt_escalation_move = np.zeros(4, dtype=int)

    healthy_cnt = 0

    reach_1 = 0
    reach_key = 0
    reach_2 = 0

    prisoners_dilemma_statistics = [{
        "initial_strategy": np.zeros(2),
        "strategy_matrix": np.zeros((2, 2, 2)),
        "cnt_matrix": np.zeros((2, 2))
    } for _ in range(2)]

    # half-cheetah
    half_cheetah_statistics = {
        "positions_run": [],
        "positions": [],
        "front_upright": 0,
        "back_upright": 0,
        "normal": 0,
        "reversed": 0
    }

    simple_more_reach_cnt = None
    simple_more_reach_steps = None
    if args.env_name == "simple-more":
        simple_more_reach_cnt = np.zeros(env.aec_env.world.num_targets, dtype=int)
        simple_more_reach_steps = np.zeros(env.aec_env.world.num_targets, dtype=int)

    episode_steps = args.episode_steps
    if args.test_episode_steps is not None:
        episode_steps = args.test_episode_steps
    # episode_steps = 1000
    env.max_frames = episode_steps

    gif_images = []

    if args.play:
        if args.env_name == "stag-hunt-gw" or args.env_name == "escalation-gw":
            env.env.symmetry_plan = None
        for i in range(num_agents):
            main_agent_conns[i].send(args.render)
        env.seed(np.random.randint(10000))
        obs = env.reset()
        meet_count = 0
        meet = False
        if args.gif:
            if is_mujoco:
                import mujoco_py
                env.env.viewer = mujoco_py.MjViewer(env.env.sim)
                # img = env.env.sim.render(width=256, height=256, depth=False, mode='offscreen', camera_name='track')
            gif_images.append(env.render(mode='rgb_array'))
        if args.render:
            if args.env_name == "stag-hunt-gw":
                # print(obs)
                for agent in env.env.agents:
                    print(agent.agent_id, agent.pos, agent.collective_return)
                print("monster", env.env.stag_pos)
                print("plant1", env.env.hare1_pos)
                print("plant2", env.env.hare2_pos)
                env.env.text_render()
                print('---------------')
            if args.env_name == "escalation-gw":
                for agent in env.env.agents:
                    print(agent.agent_id, agent.pos, agent.collective_return)
                print("escalation", env.env.escalation_pos)
                print("coop length", env.env.coop_length)
                env.env.text_render()
                print('---------------')
        dones = {agent: False for agent in env.agents}
        num_games = 0
        images = []
        is_first = True

        half_cheetah_data = {
            "x": [],
            "z": [],
            "joint": []
        }

        st = None
        while True:
            actions = dict()
            strats = dict()
            # st = time.time()
            for i, agent in enumerate(env.agents):
                main_agent_conns[i].send((obs[agent], dones[agent]))
                actions[agent], strats[agent] = main_agent_conns[i].recv()
                # print(agent, actions[agent])
            # print(time.time() - st)

            if args.env_name == "escalation-gw":
                if env.env.coop_length > 0 and not env.env.escalation_over:
                    meet_count = env.env.coop_length - 1
                    cnt_after_meeting[meet_count] += 1
                    actions_after_meeting[meet_count][0] += strats[0].probs.detach().numpy()
                    actions_after_meeting[meet_count][1] += strats[1].probs.detach().numpy()

            if args.env_name == "prisoners-dilemma":
                if is_first:
                    prisoners_dilemma_statistics[0]["initial_strategy"] = strats[0]
                    prisoners_dilemma_statistics[1]["initial_strategy"] = strats[1]
                    is_first = False
                else:
                    def get_action(one_hot):
                        if one_hot[0] > 0.5:
                            return 0
                        else:
                            return 1
                    pre_actions = (get_action(obs[0][1:3]), get_action(obs[0][3:5]))
                    # print(pre_actions)
                    prisoners_dilemma_statistics[0]["strategy_matrix"][pre_actions] += strats[0]
                    prisoners_dilemma_statistics[1]["strategy_matrix"][pre_actions] += strats[1]
                    prisoners_dilemma_statistics[0]["cnt_matrix"][pre_actions] += 1
                    prisoners_dilemma_statistics[1]["cnt_matrix"][pre_actions] += 1

            obs, rewards, dones, infos = env.step(actions)
            for i, agent in enumerate(env.agents):
                sum_reward[i] += rewards[agent]
            if args.env_name == "stag-hunt-gw":
                # for i in range(2):
                #     print(env.env.agents[i].reward_this_turn)
                #     if env.env.agents[i].reward_this_turn == env.env.defect:
                #         eat_fruit_cnt[i] += 1
                #     elif env.env.agents[i].reward_this_turn == env.env.gore:
                #         hurt_cnt[i] += 1
                #     elif env.env.agents[i].reward_this_turn == env.env.coop:
                #         attack_cnt[i] += 1
                if env.env.agents[0].pos[0] == env.env.agents[1].pos[0] and \
                        env.env.agents[0].pos[1] == env.env.agents[1].pos[1]:
                    gather_count[env.env.agents[0].pos[0]][env.env.agents[0].pos[1]] += 1
                    if not gathered:
                        first_gather_count[env.env.agents[0].pos[0]][env.env.agents[0].pos[1]] += 1
                        gathered = True

            if args.env_name == "escalation-gw":
                if rewards[0] > 0 and rewards[1] > 0:
                    meet = True
                    if env.env.coop_length == 1:
                        delta_pos = env.env.escalation_pos - env.env.agents[0].pos
                        if delta_pos[0] == 0:
                            if delta_pos[1] == -1:
                                cnt_escalation_move[0] += 1
                            elif delta_pos[1] == 1:
                                cnt_escalation_move[1] += 1
                        elif delta_pos[0] == -1:
                            cnt_escalation_move[2] += 1
                        elif delta_pos[0] == 1:
                            cnt_escalation_move[3] += 1
                    # meet_count += 1
                else:
                    meet = False
            if args.env_name == "simple-key":
                if env.aec_env.world.turn_reach_1:
                    reach_1 += 1
                if env.aec_env.world.turn_reach_key:
                    reach_key += 1
                if env.aec_env.world.turn_reach_2:
                    reach_2 += 1
            if args.env_name == "simple-more":
                if env.aec_env.world.turn_touched:
                    simple_more_reach_cnt[env.aec_env.world.touched] += 1
                    simple_more_reach_steps[env.aec_env.world.touched] += env.aec_env.world.steps
            if args.env_name == "half-cheetah":
                half_cheetah_statistics["positions_run"].append(deepcopy(env.env.sim.data.qpos))
                angle = env.env.sim.data.qpos[2]
                # print(angle)
                angle_limit = math.pi / 6
                if -angle_limit < angle <= angle_limit:
                    half_cheetah_statistics["normal"] += 1
                elif angle_limit < angle <= math.pi - angle_limit:
                    half_cheetah_statistics["front_upright"] += 1
                elif angle_limit - math.pi < angle <= -angle_limit:
                    half_cheetah_statistics["back_upright"] += 1
                else:
                    half_cheetah_statistics["reversed"] += 1
                # print(env.env.get_body_com('bfoot'), env.env.get_body_com('ffoot'))
                joints = ['bfoot', 'ffoot']
                lengths = [0.094, 0.07]
                for i, joint in enumerate(joints):
                    pos = env.env.data.get_geom_xpos(joint)
                    angle = env.env.data.get_joint_qpos(joint)
                    # print(env.env.data.get_joint_xaxis(joint))
                    # print(angle)
                    # pos = pos + lengths[i] * 2 * math.sin(angle)
                    half_cheetah_data["x"].append(pos[0])
                    half_cheetah_data["z"].append(pos[2])
                    half_cheetah_data["joint"].append(joint)
            # if args.generate_gif:
            #     gif_images.append(env.env.render(mode='rgb_array'))
            if args.render:
                if args.env_name == "stag-hunt-gw":
                    for i in range(args.num_agents):
                        print(obs[i], actions[i])
                    for agent in env.env.agents:
                        print(agent.agent_id, agent.pos, agent.collective_return)
                    print("monster", env.env.stag_pos)
                    print("plant1", env.env.hare1_pos)
                    print("plant2", env.env.hare2_pos)
                    # print(env.steps, env.max_frames, dones)
                    env.env.text_render()
                    print('---------------')
                elif args.env_name == "escalation-gw":
                    for i in range(args.num_agents):
                        print(obs[i], actions[i])
                    for agent in env.env.agents:
                        print(agent.agent_id, agent.pos, agent.collective_return)
                    print("escalation", env.env.escalation_pos)
                    print("coop length", env.env.coop_length)
                    env.env.text_render()
                    print('---------------')
                else:
                    env.render()
                    # input()
                    time.sleep(0.1)
            elif args.gif:
                if env.steps % 1 == 0:
                    image = env.render(mode='rgb_array')
                    # print(image)
                    images.append(image)

            not_done = False
            for agent in env.agents:
                not_done = not_done or not dones[agent]

            if args.env_name == "escalation-gw":
                if env.env.escalation_over:
                    # not_done = False
                    meet_count = 0
                    # actions_after_meeting[0][actions[0]] += 1
                    # actions_after_meeting[1][actions[1]] += 1

            if is_mujoco:
                if hasattr(env.env, "is_healthy"):
                    healthy_cnt += 1 if env.env.is_healthy else 0

            if not not_done:
                num_games += 1
                if args.env_name == "stag-hunt-gw":
                    eat_fruit_cnt[0] += env.env.hare1_num
                    eat_fruit_cnt[1] += env.env.hare2_num
                    hurt_cnt[0] += env.env.gore1_num
                    hurt_cnt[1] += env.env.gore2_num
                    attack_cnt += env.env.coop_num

                # print(num_games)
                # print(infos[env.agents[0]])
                # if is_simple:
                #     close_to[np.argmin(infos[env.agents[0]])] += 1
                if num_games >= args.num_games_after_training or args.num_games_after_training == -1:
                    break
                obs = env.reset()
                dones = {agent: False for agent in env.agents}
                gathered = False
                if args.env_name == "half-cheetah":
                    # half_cheetah_statistics["positions"].append(deepcopy(half_cheetah_statistics["positions_run"]))
                    half_cheetah_statistics["positions_run"] = []

        if args.gif:
            from rspo.multi_agent.utils import save_gif
            save_gif(images, os.path.join("/tmp", "plays.gif"), 15)

    for i, agent in enumerate(agents):
        main_agent_conns[i].send(None)
        agent.join()
        # print(agent)

    # print(close_to)
    # import seaborn as sns
    # sns.heatmap(gather_count)
    # plt.show()
    print("reward:", sum_reward / args.num_games_after_training)

    play_statistics = dict(rewards=sum_reward / args.num_games_after_training,
                           episode_steps=episode_steps)

    if is_mujoco:
        print("healthy_rate", healthy_cnt / args.num_games_after_training / episode_steps)

    if args.env_name == "stag-hunt-gw":
        print("eat_fruit_cnt:", eat_fruit_cnt)
        print("hurt_cnt:", hurt_cnt)
        print("attack_cnt:", attack_cnt)
        play_statistics["gather_count"] = gather_count
        play_statistics["first_gather_count"] = first_gather_count

    if args.env_name == "escalation-gw":
        actions_after_meeting /= np.maximum(cnt_after_meeting, 1).reshape(max_meet, 1, 1)
        play_statistics["actions_after_meeting"] = actions_after_meeting
        play_statistics["cnt_after_meeting"] = cnt_after_meeting

        if args.use_wandb:
            for i in range(args.episode_steps):
                wandb.run.summary["reach-{}".format(i + 1)] = cnt_after_meeting[i]
    # if args.env_name[:6] == "simple":
    #     play_statistics["close_to"] = close_to
    #     result["close_to"] = close_to
    if args.env_name == "simple-key":
        play_statistics["reach_1"] = reach_1
        play_statistics["reach_key"] = reach_key
        play_statistics["reach_2"] = reach_2

    if args.env_name == "simple-more":
        play_statistics["reach_cnt"] = simple_more_reach_cnt
        play_statistics["reach_steps"] = simple_more_reach_steps / np.maximum(simple_more_reach_cnt, 1)
        play_statistics["total_reach"] = simple_more_reach_cnt.sum()

    if args.env_name == "prisoners-dilemma":
        np.set_printoptions(precision=3, suppress=True)
        for i in range(2):
            prisoners_dilemma_statistics[i]["strategy_matrix"] /= \
                np.maximum(1., prisoners_dilemma_statistics[i]["cnt_matrix"]).reshape(2, 2, 1)
            play_statistics["initial_strategy_{}".format(i)] = prisoners_dilemma_statistics[i]["initial_strategy"]
            play_statistics["strategy_matrix_{}".format(i)] = prisoners_dilemma_statistics[i]["strategy_matrix"]
            play_statistics["cnt_matrix_{}".format(i)] = prisoners_dilemma_statistics[i]["cnt_matrix"] / args.num_games_after_training

    if args.play and args.env_name == "half-cheetah":
        # np.set_printoptions(precision=3, suppress=True)
        # half_cheetah_statistics["average_pos"] /= args.num_games_after_training
        # print("average_pos:", half_cheetah_statistics["average_pos"])

        situations = ["normal", "reversed", "front_upright", "back_upright"]

        for situation in situations:
            play_statistics[situation] = half_cheetah_statistics[situation]
            if args.use_wandb:
                wandb.run.summary[situation] = half_cheetah_statistics[situation]

        play_statistics["positions"] = half_cheetah_statistics["positions"]

        plot_joint = False

        if plot_joint:
            df = pd.DataFrame(data=half_cheetah_data)
            sns.lineplot(x="x", y="z", hue="joint", data=df.query("x > 80 and x < 100"))
            plt.show()

        plot = False

        if plot:
            data = {
                "step": [],
                "which_pos": [],
                "which_run": [],
                "pos": []
            }
            for i, positions_run in enumerate(half_cheetah_statistics["positions"]):
                for j, position in enumerate(positions_run):
                    for k in range(position.shape[0]):
                        data["step"].append(j)
                        data["which_pos"].append(f"pos_{k}")
                        data["which_run"].append(f"run_{i}")
                        data["pos"].append(position[k])

            df = pd.DataFrame(data=data)
            sns.lineplot(x="step", y="pos", hue="which_run", data=df.query("which_pos == 'pos_2'"))
            plt.show()

    if args.play:
        joblib.dump(play_statistics, os.path.join(save_dir, "play_statistics.obj"))
        show_play_statistics(args.env_name, play_statistics)

    # if args.generate_gif:
    #     if len(gif_images) > 0:
    #         import imageio
    #         imageio.mimsave("play.gif", gif_images)

    # print(result["statistics"][env.agents[0]]["likelihood"])
    # print(result["statistics"][env.agents[1]]["likelihood"])

    result["play_statistics"] = play_statistics
    result["save_dir"] = save_dir

    act_shm.close()
    act_shm.unlink()

    return result


def run(args):
    log_level_mapping = {
        "info": logging.INFO,
        "error": logging.ERROR
    }
    logger = mp.log_to_stderr()
    logger.setLevel(log_level_mapping[args.log_level])
    return _run(args, logger)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
